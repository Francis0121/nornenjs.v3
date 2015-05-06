
#include <cstdio>
#include <helper_math.h>

typedef unsigned int  uint;
typedef unsigned char uchar;
typedef unsigned char VolumeType;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;
texture<float4,  1, cudaReadModeElementType> texture_float_1D;

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{

    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}
__device__ unsigned char myMAX(unsigned char a, unsigned char b)
{
	if(a >= b)
		return a;
	else
		return b;
}

__device__
float3 mul(const float *M, const float3 &v)
{
   float3 r;
   
   r.x = v.x * M[0] + v.y * M[1] + v.z * M[2];
   r.y = v.x * M[4] + v.y * M[5] + v.z * M[6];
   r.z = v.x * M[8] + v.y * M[9] + v.z * M[10];
   
   return r;
}

__device__
float4 mul(const float *M, const float4 &v)
{
	float4 r;

	r.x = v.x * M[0] + v.y * M[1] + v.z * M[2]  + v.w * M[3];
	r.y = v.x * M[4] + v.y * M[5] + v.z * M[6]  + v.w * M[7];
	r.z = v.x * M[8] + v.y * M[9] + v.z * M[10] + v.w * M[11];	
	r.w = 1.0f;
	
	return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}
extern "C" {
__global__ void TF2d_kernel(float4* TF2d_k, int TFSize)
	{
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

        if ((x >= TFSize) || (y >= TFSize)) return;

        float4 temp;
        float4 result = {0.0};
        float4 sum = {0.0f};

        int nx, ny, diff;
        if(x>y){
        		diff = x-y;
        		ny = x;
        		nx = y;
        }
        else if(y>x){
        		diff = y-x;
        		nx = x;
        		ny = y;
        }
        else{
        		diff=1;
        		nx = ny = x;
        		sum.w = 0.0f;
        }

        for(int i=nx; i<ny; i++){
        		temp = tex1D(texture_float_1D, i);

        		temp.x *= temp.w;
        		temp.y *= temp.w;
        		temp.z *= temp.w;

        		sum.x += temp.x;
        		sum.y += temp.y;
        		sum.z += temp.z;
        		sum.w += temp.w;
        }

        result.x = sum.x / diff; //* (newAlpha/sum.w);
        result.y = sum.y / diff; //* (newAlpha/sum.w);
        result.z = sum.z / diff; //* (newAlpha/sum.w);
        result.w = sum.w / diff;

        TF2d_k[TFSize*y + x].x = result.x;
        TF2d_k[TFSize*y + x].y = result.y;
        TF2d_k[TFSize*y + x].z = result.z;
        TF2d_k[TFSize*y + x].w = result.w;
	}
}
extern "C" {
__global__ void block_volume(unsigned char* image_p,
							 unsigned char* dest_p,
							 int srcWidth,
							 int srcHeight,
							 int srcDepth,
							 int desWidth,
							 int desHeight,
							 int desDepth){


				unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
                unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y;

                if (tx >= desWidth || ty >= desHeight) return;

				for(int i=0; i<desDepth; i++){
                		dest_p[i*desWidth*desHeight + ty*desHeight + tx] = 0;
                		unsigned char tempmax=0;

                		for(int z=i*4; z<=i*4+4; z++)
                			for(int y=ty*4; y<=ty*4+4; y++)
                				for(int x=tx*4; x<=tx*4+4; x++){
                					if(z>=srcDepth || y>=srcHeight || x>=srcWidth )
                						continue;
                						tempmax = myMAX(tempmax, image_p[z*srcWidth*srcHeight + y*srcHeight + x]);
                				}
                		dest_p[i*desWidth*desHeight + ty*desHeight + tx] = tempmax;
                	}

		}

}
extern "C" {
__global__ void render_kernel_volume(uint *d_output, 
								     float *d_invViewMatrix,
								     float4* TF2d_k,
								     unsigned int imageW,
								     unsigned int imageH,
								     float brightness,
								     float transferScaleX,
								     float transferScaleY,
								     float transferScaleZ,
								     unsigned int quality
								 )
{
	
		const int maxSteps = 500;
		const float tstep = 0.01f;
		const float opacityThreshold = 0.95f;
		const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
		const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);
	 
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	 
		if ((x >= imageW) || (y >= imageH)) return;
	 
		float u = (x / (float) imageW)*2.0f-1.0f;
		float v = (y / (float) imageH)*2.0f-1.0f;
	 
		Ray eyeRay;
		eyeRay.o = make_float3(mul(d_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
		eyeRay.d = normalize(make_float3(u, v, -2.0f));
		eyeRay.d = mul(d_invViewMatrix, eyeRay.d);
	 
		float tnear, tfar;
		int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
	 
		if (!hit) return;
	 
		if (tnear < 0.0f) tnear = 0.0f; 
	 
		float4 sum = make_float4(0.0f);
		float t = tnear;
		float3 pos = eyeRay.o + eyeRay.d * tnear;
		float3 step = eyeRay.d*tstep;
	 
		for (float i=0; i<maxSteps; i++){

				float sample = tex3D(tex,pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
				float sample_next = tex3D(tex, pos.x*0.5f+0.5+(step.x*0.5), pos.y*0.5f+0.5f +(step.y*0.5),  pos.z*0.5f+0.5f+(step.z*0.5));

				float4 col=make_float4(0.0f);

     			col.w = TF2d_k[(256*(int)(sample*256)) + (int)sample_next*256].w;
     			col.x = TF2d_k[(256*(int)(sample*256)) + (int)sample_next*256].x;
     			col.y = TF2d_k[(256*(int)(sample*256)) + (int)sample_next*256].y;
     			col.z = TF2d_k[(256*(int)(sample*256)) + (int)sample_next*256].z;

     			if(quality == 1){
     			
					float3 nV = {0.0, 0.0, 0.0};
					float3 lV = {0.0, 0.0, 0.0};
	
					lV.x = eyeRay.d.x;
					lV.y = eyeRay.d.y;
					lV.z = eyeRay.d.z;
	
					float x_plus = tex3D(tex, pos.x*0.5f+0.5+(step.x*0.5), pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
					float x_minus = tex3D(tex,pos.x*0.5f+0.5-(step.x*0.5), pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
	
					float y_plus = tex3D(tex, pos.x*0.5f+0.5, pos.y*0.5f+0.5f +(step.y*0.5), pos.z*0.5f+0.5f);
					float y_minus = tex3D(tex, pos.x*0.5f+0.5, pos.y*0.5f+0.5f-(step.y*0.5),pos.z*0.5f+0.5f);
	
					float z_plus = tex3D(tex, pos.x*0.5f+0.5, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f+(step.z*0.5));
					float z_minus = tex3D(tex, pos.x*0.5f+0.5, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f-(step.z*0.5));
	
					nV.x = (x_plus - x_minus)/2.0f;
					nV.y = (y_plus - y_minus)/2.0f;
					nV.z = (z_plus - z_minus)/2.0f;

					float NL = 0.0f;
					NL = lV.x*nV.x + lV.y*nV.y + lV.z*nV.z;
	
					if(NL < 0.0f) NL = 0.0f;
					float localShading = 0.2 + 0.8*NL;
	
					col *= localShading;
     			}
				col.x *= col.w;
				col.y *= col.w;
				col.z *= col.w;
				
				sum = sum + col*(1.0f - sum.w);
     
				if (sum.w > opacityThreshold)
					break;
					
				t += (tstep*0.5);

				if (t > tfar) break;

				pos += (step*0.5);

		}
		sum *= brightness;
		sum.w=0.0;
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
	}
}
extern "C" {
__global__ void render_kernel_MIP(uint *d_output, 
                                  float *d_invViewMatrix,
                                  float4* TF2d_k,
                                  unsigned int imageW,
                                  unsigned int imageH,
                                  float brightness,
                                  float transferScaleX,
                                  float transferScaleY,
                                  float transferScaleZ,
                                  unsigned int quality)
{
	
		const int maxSteps = 500;
		const float tstep = 0.01f;
		const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
		const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);
	 
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	 
		if ((x >= imageW) || (y >= imageH)) return;
	 
		float u = (x / (float) imageW)*2.0f-1.0f;
		float v = (y / (float) imageH)*2.0f-1.0f;
	 
		Ray eyeRay;
		eyeRay.o = make_float3(mul(d_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
		eyeRay.d = normalize(make_float3(u, v, -2.0f));
		eyeRay.d = mul(d_invViewMatrix, eyeRay.d);
	 
		float tnear, tfar;
		int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
	 
		if (!hit) return;
	 
		if (tnear < 0.0f) tnear = 0.0f; 
	 
		float4 sum = make_float4(0.0f);
		float t = tnear;
		float3 pos = eyeRay.o + eyeRay.d * tnear;
		float3 step = eyeRay.d*tstep;
		float max = 0.0f; 
		for (float i=0; i<maxSteps; i++){
				
				float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
				if(sample >= max) 
					max = sample;
					
				t += (tstep*0.5);

			   if (t > tfar) break;

				pos += (step*0.5);
			
		}
		sum.x = max;
		sum.y = max;
		sum.z = max;
		sum.w = 0;
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
	}
}
extern "C" {
__global__ void render_kernel_MPR(uint *d_output,
                                  float *d_invViewMatrix,
                                  float4* TF2d_k,
                                  unsigned int imageW,
                                  unsigned int imageH,
                                  float brightness,
                                  float transferScaleX,
                                  float transferScaleY,
                                  float transferScaleZ,
                                  unsigned int quality)
	{
		const int maxSteps = 500;
		const float tstep = 0.01f;
		const float opacityThreshold = 0.95f;
		const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
		const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

		uint x = blockIdx.x*blockDim.x + threadIdx.x;
		uint y = blockIdx.y*blockDim.y + threadIdx.y;

		if ((x >= imageW) || (y >= imageH)) return;

		float u = (x / (float) imageW)*2.0f-1.0f;
		float v = (y / (float) imageH)*2.0f-1.0f;

		// calculate eye ray in world space
		Ray eyeRay;
		eyeRay.o = make_float3(mul(d_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
		eyeRay.d = normalize(make_float3(u, v, -2.0f));
		eyeRay.d = mul(d_invViewMatrix, eyeRay.d);

		// find intersection with box
		float tnear, tfar;
		int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

		if (!hit) return;

		if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

		// march along ray from front to back, accumulating color
		float4 sum = make_float4(0.0f);
		float t = tnear;
		float3 pos = eyeRay.o + eyeRay.d * tnear;
		float3 step = eyeRay.d*tstep;

		float max = 0.0f;

		float sample = tex3D(tex, pos.x*0.5f+0.5f+transferScaleX, pos.y*0.5f+0.5f+transferScaleY, pos.z*0.5f+0.5f+transferScaleZ);

		sum.x = sample;
		sum.y = sample;
		sum.z = sample;
		sum.w = 0;
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
	}
}

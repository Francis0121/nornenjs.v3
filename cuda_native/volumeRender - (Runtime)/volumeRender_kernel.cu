/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_blockArray = 0;
cudaArray *d_transferFuncArray;

typedef unsigned char VolumeType;
//typedef unsigned short VolumeType;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;         // 3D texture
texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex_block;         // 3D texture
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
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
float3 cudaNormalize(float3 a){
	float3 temp={a.x, a.y, a.z};
	float sum = sqrt((float)(a.x*a.x + a.y*a.y + a.z*a.z));

	if(sum == 0){
		temp.x = 0;
		temp.y = 0;
		temp.z = 0;
	}else{
		temp.x /= sum;
		temp.y /= sum;
		temp.z /= sum;
	}

	return temp;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
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
__device__ uchar rgbaFloatToChar(float rgba)
{
	rgba = __saturatef(rgba);   // clamp to [0.0, 1.0]
	return (uchar(rgba*255));
}
__global__ void makeBlock_kernel(unsigned char* image_p, unsigned char* dest_p, cudaExtent blockSize, cudaExtent volumeSize)
{
	int tx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx >= blockSize.width || ty >= blockSize.height) return;

	for(int i=0; i<blockSize.depth; i++){
		dest_p[i*blockSize.width*blockSize.height + ty*blockSize.height + tx] = 0;
		unsigned char tempmax=0;

		for(int z=i*4; z<=i*4+4; z++)
			for(int y=ty*4; y<=ty*4+4; y++)
				for(int x=tx*4; x<=tx*4+4; x++){
					if(z>=volumeSize.depth || y>=volumeSize.height || x>=volumeSize.width )
						continue;
					tempmax = myMAX(tempmax, image_p[z*volumeSize.width*volumeSize.height + y*volumeSize.height + x]);
				}
		dest_p[i*blockSize.width*blockSize.height + ty*blockSize.height + tx] = tempmax;
	}
}
__global__ void
d_render(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale)
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
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
	float4 temp =make_float4(0.0f);
	//uint4 sum = make_uint4(0);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d * tnear;
    float3 step = eyeRay.d*tstep;
	float max = 0.0f; 
    for (float i=0; i<maxSteps; i++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates

	   // float block_den = tex3D(tex_block, (pos.x*0.5f+0.5f), (pos.y*0.5f+0.5f), (pos.z*0.5f+0.5f));
		//uint density = __float2uint_rn(block_den*256);
		/*temp.w = block_den;
		temp.x = block_den;
		temp.y = block_den;
		temp.z = block_den;
		uint density =  ((unsigned int)(temp.w*255)<<24) | ((unsigned int)(temp.z*255)<<16) | ((unsigned int)(temp.y*255)<<8) | (unsigned int)(temp.x*255);*/
	//	if(block_den >= max) 
//				max = block_den;*/
		//if(((density >> 16) &255) < 4) { //빈공간 도약 - PALLET_START~PALLET_END까지만 그리기 때문에
		//	
		//}
		//else{
			float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
	       
			// lookup in transfer function texture
			float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);
	      
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

			//nV = cudaNormalize(nV);

			float NL = 0.0f;
			NL = lV.x*nV.x + lV.y*nV.y + lV.z*nV.z;

			if(NL < 0.0f) NL = 0.0f;
			float localShading = 0.2 + 0.8*NL;
			
			//col*=localShading;
			// pre-multiply alpha
			col.x *= col.w;
			col.y *= col.w;
			col.z *= col.w;
			// "over" operator for front-to-back blending
			sum = sum + col*(1.0f - sum.w);

			// exit early if opaque
			if (sum.w > opacityThreshold)
				break;

			t += (tstep*0.5);

			if (t > tfar) break;

			pos += (step*0.5);
		//}
	}
	/*sum.x = max;
	sum.y = max;
	sum.z = max;
	sum.w = 0;*/
    sum *= brightness;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
	
}

extern "C"
void* make_blockVolume(void* image, cudaExtent blockSize, cudaExtent volumeSize)
{
	unsigned int vsize = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(VolumeType);
	unsigned int bsize = blockSize.width * blockSize.height * blockSize.depth * sizeof(VolumeType);

	unsigned char *dest; //cpu에 보낼 블락 data
	unsigned char *dest_p; //gpu에서 사용할 블락 데이터
	unsigned char *image_p; //볼륨 데이터

	dest = new unsigned char[bsize/sizeof(VolumeType)]; //64*64*57
	memset((void*)dest, 0, bsize);

	cudaMalloc((void**)&image_p, vsize); 
	cudaMemcpy(image_p, image, vsize, cudaMemcpyHostToDevice); 

	cudaMalloc((void**)&dest_p, bsize);

	dim3 Db = dim3(16, 16);
	dim3 Dg = dim3(4, 4);

	makeBlock_kernel<<<Dg, Db>>>(image_p, dest_p, blockSize, volumeSize);

	cudaMemcpy(dest, dest_p, bsize, cudaMemcpyDeviceToHost);
	/*for(int i=0; i<64*64*47; i++)
	{
		printf("%d\n",dest[i]);
	}*/
	cudaFree(image_p);
	cudaFree(dest_p);

	return dest;

}
extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
    tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}
extern "C"
void initBlockTexture(void *h_volume_block, int x, int y, int z)
{
	cudaExtent block_Size = make_cudaExtent(x, y, z);
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors( cudaMalloc3DArray(&d_blockArray, &channelDesc, block_Size) );

    // copy data to 3D array
    cudaMemcpy3DParms myParams = {0};
	myParams.srcPtr   = make_cudaPitchedPtr(h_volume_block, block_Size.width*sizeof(VolumeType), block_Size.width, block_Size.height);
    myParams.dstArray = d_blockArray;
    myParams.extent   = block_Size;
    myParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors( cudaMemcpy3D(&myParams) );

    // set texture parameters
    tex_block.normalized = true;                      // access with normalized texture coordinates
    tex_block.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex_block.channelDesc = channelDesc;
	tex_block.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates
    tex_block.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex_block, d_blockArray, channelDesc));            
} 
extern "C"
void initCuda(void *h_volume, cudaExtent volumeSize)
{
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeBorder;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeBorder;
    // tex.addressMode[2] = cudaAddressModeBorder;
    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

	float4 transferFunc[256];
   
	 for(int i=0; i<=80; i++){    //alpha
		 transferFunc[i].w = 0.0f;
		 transferFunc[i].x = 0.0f;
		 transferFunc[i].y = 0.0f;
		 transferFunc[i].z = 0.0f;
	}
	for(int i=80+1; i<=100; i++){
		transferFunc[i].w = (1.0 / (100-80)) * ( i - 80);
		transferFunc[i].x = (1.0 / (100-80)) * ( i - 80);
		transferFunc[i].y = (1.0 / (100-80)) * ( i - 80);
		transferFunc[i].z = (1.0 / (100-80)) * ( i - 80);
	}
	for(int i=100+1; i<256; i++){
		transferFunc[i].w =1.0f;
		transferFunc[i].x =1.0f;
		transferFunc[i].y =1.0f;
		transferFunc[i].z =1.0f;
	}
	//-------------------------------------------------------------------
	// create transfer function texture
  //  float4 transferFunc[] =
  //  {
  //     /* {  0.0, 0.0, 0.0, 0.0, },
  //      {  1.0, 0.0, 0.0, 1.0, },
  //      {  1.0, 0.5, 0.0, 1.0, },
  //      {  1.0, 1.0, 0.0, 1.0, },
  //      {  0.0, 1.0, 0.0, 1.0, },
  //      {  0.0, 1.0, 1.0, 1.0, },
  //      {  0.0, 0.0, 1.0, 1.0, },
  //      {  1.0, 0.0, 1.0, 1.0, },
  //      {  0.0, 0.0, 0.0, 0.0, },*/

		//{  0.0, 0.0, 0.0, 0.0, },
  //      {  0.0, 0.0, 0.0, 1.0, },
  //      {  0.0, 0.0, 0.1, 0.2, },
  //      {  0.3, 0.4, 0.5, 0.6, },
  //      {  0.7, 0.8, 0.9, 1.0, },
  //      {  1.0, 1.0, 1.0, 1.0, },
  //      {  1.0, 1.0, 1.0, 1.0, },
  //      {  1.0, 1.0, 1.0, 1.0, },
  //      {  1.0, 1.0, 1.0, 0.0, },
  //  };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray;
    checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1));
    checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));
}

extern "C"
void freeCudaBuffers()
{
    checkCudaErrors(cudaFreeArray(d_volumeArray));
    checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                   float density, float brightness, float transferOffset, float transferScale)
{
    d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, density,
                                      brightness, transferOffset, transferScale);
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_

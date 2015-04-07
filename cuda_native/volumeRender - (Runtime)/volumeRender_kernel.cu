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
cudaArray *d_transferFuncArray1 = 0;
typedef unsigned char VolumeType;
//typedef unsigned short VolumeType;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;         // 3D texture
texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex_block;         // 3D texture
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture
texture<float4, 1, cudaReadModeElementType>         transferTex1; // 1D transfer function texture
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

	   // float block_den = tex3D(tex_block, (pos.x*0.5f+0.5f), (pos.y*0.5f+0.5f), (pos.z*0.5f+0.5f))*65535;
		//float3 advanced = {0.0f,0.0f.0.0f};
		//uint density = __float2uint_rn(block_den*256);
		/*temp.w = block_den;
		temp.x = block_den;
		temp.y = block_den;
		temp.z = block_den;
		uint density =  ((unsigned int)(temp.w*255)<<24) | ((unsigned int)(temp.z*255)<<16) | ((unsigned int)(temp.y*255)<<8) | (unsigned int)(temp.x*255);*/
	   //	if(block_den >= max) 
       //				max = block_den;*/
	   //if((int)block_den < 80) { //빈공간 도약 - PALLET_START~PALLET_END까지만 그리기 때문에
		  // int3 nowPos= {(pos.x*0.5f+0.5f), (pos.y*0.5f+0.5f), (pos.z*0.5f+0.5f)};
		  // int3 advpos;
		  // do{
				//pos += (step*0.5);
			
		  // }
		
	    //
	    //}
		//else{
			float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
	       // float sample_next = tex3D(tex, pos.x*0.5f+0.5+(step.x*0.5), pos.y*0.5f+0.5f +(step.y*0.5), pos.z*0.5f+0.5f+(step.z*0.5));
			
			// lookup in transfer function texture
			float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);
			//float4 col={0.0};
			//float diff;
			//if(sample<=sample_next){
				// diff=sample_next-sample;
				// float4 col= (tex1D(transferTex1, (sample_next-transferOffset)*transferScale) - tex1D(transferTex1, (sample-transferOffset)*transferScale)) / diff;
				
			//}
			//else if(sample>sample_next){
			//	diff=sample-sample_next;
			///	col= (tex1D(transferTex1, (sample-transferOffset)*transferScale) - tex1D(transferTex1, (sample_next-transferOffset)*transferScale)) / diff;
			//}
			//float4 col = tex3D(transferTex1,sample,sample_next,0);


			//float3 nV = {0.0, 0.0, 0.0};
			//float3 lV = {0.0, 0.0, 0.0};

			//lV.x = eyeRay.d.x;
			//lV.y = eyeRay.d.y;
			//lV.z = eyeRay.d.z;
			//
			//float x_plus = tex3D(tex, pos.x*0.5f+0.5+(step.x*0.5), pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
			//float x_minus = tex3D(tex,pos.x*0.5f+0.5-(step.x*0.5), pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);

			//float y_plus = tex3D(tex, pos.x*0.5f+0.5, pos.y*0.5f+0.5f +(step.y*0.5), pos.z*0.5f+0.5f);
			//float y_minus = tex3D(tex, pos.x*0.5f+0.5, pos.y*0.5f+0.5f-(step.y*0.5),pos.z*0.5f+0.5f);

			//float z_plus = tex3D(tex, pos.x*0.5f+0.5, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f+(step.z*0.5));
			//float z_minus = tex3D(tex, pos.x*0.5f+0.5, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f-(step.z*0.5));

			//nV.x = (x_plus - x_minus)/2.0f;
			//nV.y = (y_plus - y_minus)/2.0f;
			//nV.z = (z_plus - z_minus)/2.0f;

			////nV = cudaNormalize(nV);

			//float NL = 0.0f;
			//NL = lV.x*nV.x + lV.y*nV.y + lV.z*nV.z;

			//if(NL < 0.0f) NL = 0.0f;
			//float localShading = 0.2 + 0.8*NL;
			
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




//struct OTF_2D* getPre_integration(){
//
//	
//	for(int x=0; x<256; x++){
//		for(int y=0; y<256; y++){
//
//			float4 result;
//			float4 temp={0.0f};
//
//			if(y > x){
//				for(int i=x; i<y; i++){
//					temp.x = transferFunc[i].x;
//					temp.y = transferFunc[i].y;
//					temp.z = transferFunc[i].z;
//					temp.w = transferFunc[i].w;
//					
//					float diff = i-x;
//
//					if(diff == 0.0f)
//						diff = 1.0f;
//
//					temp.w = 1.0f-pow(1-temp.w, 1/diff);
//
//					result.x += (1-result.w)*temp.x*temp.w;
//					result.y += (1-result.w)*temp.y*temp.w;
//					result.z += (1-result.w)*temp.z*temp.w;
//					result.w += (1-result.w)*temp.w;
//				}
//			}
//			else if(x > y){
//				for(int i=y; i<x; i++){
//					temp.x = transferFunc[i].x;
//					temp.y = transferFunc[i].y;
//					temp.z = transferFunc[i].z;
//					temp.w = transferFunc[i].w;
//
//					float diff = i-y;
//
//					if(diff == 0.0f)
//						diff = 1.0f;
//
//					temp.w = 1.0f-pow(1-temp.w, 1/diff);
//
//					result.x += (1-result.w)*temp.x*temp.w;
//					result.y += (1-result.w)*temp.y*temp.w;
//					result.z += (1-result.w)*temp.z*temp.w;
//					result.w += (1-result.w)*temp.w;
//				}
//			}
//			else {
//				result.x = 255.0f;
//				result.y = 255.0f;
//				result.z = 255.0f;
//				result.w = 0.0f;
//			}
//			OTF_2D[256*x + y].sum_R = result.x;
//			OTF_2D[256*x + y].sum_G = result.y;
//			OTF_2D[256*x + y].sum_B = result.z;
//			OTF_2D[256*x + y].sum_a = result.w;
//		}
//	}
//	return OTF_2D;
//}
extern "C"
{
	void FreeGPUVolArray(void)
	{
		cudaFreeArray(d_volumeArray);
		cudaFreeArray(d_blockArray);
	}
 
	void FreeGPUTFArray(void)
	{
		cudaFreeArray(d_TFArray);
	}
 
	void FreeGPUEtcArray(void)
	{
		cudaFreeArray(d_AverageArray);
		cudaFreeArray(d_SigmaArray);
	}
}
 
 
void initTFTexture(int width, float4 *h_data)   
{
	if(d_TFArray != 0)
		cudaFreeArray(d_TFArray);
 
	uint size = width*sizeof(float)*4;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	checkCudaErrors(cudaMallocArray(&d_TFArray, &channelDesc, width)); 
	
	checkCudaErrors(cudaMemcpyToArray(d_TFArray, 0, 0, h_data, size, cudaMemcpyHostToDevice));
 
    tex_TF.addressMode[0] = cudaAddressModeClamp;
    tex_TF.addressMode[1] = cudaAddressModeClamp;
    tex_TF.filterMode = cudaFilterModePoint;
    tex_TF.normalized = false;    // access with integer texture coordinates
	checkCudaErrors(cudaBindTextureToArray(tex_TF, d_TFArray, channelDesc));
 
}
 
 
void initVolume(const ushort *h_volume, cudaExtent volumeSize, int bytePerVoxel)
{
	if(d_volumeArray != NULL) {
		cudaFreeArray(d_volumeArray);
		d_volumeArray=NULL;
	}
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(bytePerVoxel*8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors( cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize, 0) );
 
    // copy data to 3D array
	int x = volumeSize.width;
	int y = volumeSize.height;
    cudaMemcpy3DParms myParams = {0};
    myParams.srcPtr   = make_cudaPitchedPtr((void*)h_volume, x*bytePerVoxel, x, y);
    myParams.dstArray = d_volumeArray;
    myParams.extent   = volumeSize;
    myParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors( cudaMemcpy3D(&myParams) );
 
    // set texture parameters
    tex_volume.normalized = false;                      // access with normalized texture coordinates
    tex_volume.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex_volume.channelDesc = channelDesc;
	tex_volume.addressMode[0] = cudaAddressModeBorder;   // wrap texture coordinates
    tex_volume.addressMode[1] = cudaAddressModeBorder;
    tex_volume.addressMode[2] = cudaAddressModeBorder;
 
 
    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex_volume, d_volumeArray, channelDesc));
}
 
 
void initAvgVolume(const float *h_volume, cudaExtent volumeSize, int bytePerVoxel)
{
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(bytePerVoxel*8, 0, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors( cudaMalloc3DArray(&d_AverageArray, &channelDesc, volumeSize, 0) );
 
    // copy data to 3D array
	int x = volumeSize.width;
	int y = volumeSize.height;
    cudaMemcpy3DParms myParams = {0};
    myParams.srcPtr   = make_cudaPitchedPtr((void*)h_volume, x*bytePerVoxel, x, y);
    myParams.dstArray = d_AverageArray;
    myParams.extent   = volumeSize;
    myParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors( cudaMemcpy3D(&myParams) );
 
    // set texture parameters
    tex_average.normalized = false;                      // access with normalized texture coordinates
    tex_average.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex_average.channelDesc = channelDesc;
	tex_average.addressMode[0] = cudaAddressModeBorder;   // wrap texture coordinates
    tex_average.addressMode[1] = cudaAddressModeBorder;
    tex_average.addressMode[2] = cudaAddressModeBorder;
 
 
    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex_average, d_AverageArray, channelDesc));
}
 
 
void initSigVolume(const float *h_volume, cudaExtent volumeSize, int bytePerVoxel)
{
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(bytePerVoxel*8, 0, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors( cudaMalloc3DArray(&d_SigmaArray, &channelDesc, volumeSize, 0) );
 
    // copy data to 3D array
	int x = volumeSize.width;
	int y = volumeSize.height;
    cudaMemcpy3DParms myParams = {0};
    myParams.srcPtr   = make_cudaPitchedPtr((void*)h_volume, x*bytePerVoxel, x, y);
    myParams.dstArray = d_SigmaArray;
    myParams.extent   = volumeSize;
    myParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors( cudaMemcpy3D(&myParams) );
 
    // set texture parameters
    tex_sigma.normalized = false;                      // access with normalized texture coordinates
    tex_sigma.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex_sigma.channelDesc = channelDesc;
	tex_sigma.addressMode[0] = cudaAddressModeBorder;   // wrap texture coordinates
    tex_sigma.addressMode[1] = cudaAddressModeBorder;
    tex_sigma.addressMode[2] = cudaAddressModeBorder;
 
 
    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex_sigma, d_SigmaArray, channelDesc));
}
 
 
void initBlockTexture(const ushort *h_volume_block, cudaExtent blockSize, int bytePerVoxel)
{
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(bytePerVoxel*8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors( cudaMalloc3DArray(&d_blockArray, &channelDesc, blockSize, 0) );
 
    // copy data to 3D array
	int x = blockSize.width;
	int y = blockSize.height;
    cudaMemcpy3DParms myParams = {0};
    myParams.srcPtr   = make_cudaPitchedPtr((void*)h_volume_block, x*bytePerVoxel, x, y);
    myParams.dstArray = d_blockArray;
    myParams.extent   = blockSize;
    myParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors( cudaMemcpy3D(&myParams) );
 
    // set texture parameters
    tex_block.normalized = false;                      // access with normalized texture coordinates
    tex_block.filterMode = cudaFilterModePoint;      // linear interpolation
    tex_block.channelDesc = channelDesc;
	tex_block.addressMode[0] = cudaAddressModeBorder;   // wrap texture coordinates
    tex_block.addressMode[1] = cudaAddressModeBorder;
    tex_block.addressMode[2] = cudaAddressModeBorder;
 
 
    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex_block, d_blockArray, channelDesc));            
} 
 
 
 
__device__ void GetRayBound(float *t, float3 sdot, float3 start, cudaExtent volumeSize){
 
	const float EPS = 0.00001; // epsilon
	// [0,0,0] ~ [255,255,224] box
	// eye : sdot
	// direction : start
	// get t1, t2
	float kx[2]={-20000,20000}, ky[2]={-20000,20000}, kz[2]={-20000,20000};
	// sdot.x + kx[0] * start.x = 0;
	if( fabs((float)start.x) > EPS) {
		kx[0] = (0 - sdot.x) / start.x;
		kx[1] = (volumeSize.width - sdot.x) / start.x;
		if( kx[0] > kx[1] ) { // in > out
			float temp = kx[0];
			kx[0] = kx[1];
			kx[1] = temp;
		}
	}
 
	if( fabs((float)start.y) > EPS){
		ky[0] = (0 - sdot.y) / start.y;
		ky[1] = (volumeSize.height - sdot.y) / start.y;
		if( ky[0] > ky[1] ) { // in > out
			float temp = ky[0];
			ky[0] = ky[1];
			ky[1] = temp;
		}
	}
 
	if( fabs((float)start.z) > EPS){
		kz[0] = (0 - sdot.z) / start.z;
		kz[1] = (volumeSize.depth - sdot.z) / start.z;
		if( kz[0] > kz[1] ) { // in > out
			float temp = kz[0];
			kz[0] = kz[1];
			kz[1] = temp;
		}
	}
 
	float kin = max(max(kx[0], ky[0]), kz[0]);
	float kout = min(min(kx[1], ky[1]), kz[1]);
 
	t[0] = kin + 0.05f;
	t[1] = kout - 0.05f;
 
}
 
__device__ ushort myMAX(ushort a, ushort b)
{
	if(a >= b)
		return a;
	else 
		return b;
}
 
__device__ float3 cudaNormalize(float3 a){
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
 
 
__device__ float GetSum(float Average, float Sigma, int start, int end, float* probability_k)
{
	//start=start*16.0f;
	//end=end*16.0f;
	//Average = Average-200.0f;
	float startz = (start - Average)/Sigma;
	float endz = (end - Average)/Sigma;
	float pi = 3.141592f, e = 2.718f;
	float p1=0.0f, p2=0.0f;
 
	if(startz > 5.0f)
		p1 = 1.0f;
	else if(startz < -5.0f)
		p1 = 0.0f;
	//else if(startz < -10.0f)
	//	p1 = 0.5f;
	else if(startz >= 0.0f)
		p1 = 0.5f + probability_k[(int)(startz*100)];
	else 
		p1 = 0.5f - probability_k[-(int)(startz*100)];
 
	if(endz > 5.0f)
		p2 = 1.0f;
	else if(endz < -5.0f)
		p2 = 0.0f;
	//else if(endz < -10.0f)
	//	p2 = 0.5f;
	else if(endz >= 0.0f)
		p2 = 0.5f + probability_k[(int)(endz*100)];
	else 
		p2 = 0.5f - probability_k[-(int)(endz*100)];
 
	if(endz == startz)
		endz = startz+0.1f; //debug code
 
	
	return ((1.0f/((endz-startz)*sqrt(2.0f*pi)))*((1.0f/pow(sqrt(e), startz*startz))-
		(1.0f/pow(sqrt(e), endz*endz))) + (-startz*1.0f/(endz-startz))*(p2-p1) + 1.0f*(1-p2));
}
 
 
__global__ void makeBlock_kernel(ushort* image_p, ushort* dest_p, cudaExtent blockSize, cudaExtent volumeSize)
{
	int tx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx >= blockSize.width || ty >= blockSize.height) return;
 
	for(int i=0; i<blockSize.depth; i++){
		dest_p[i*blockSize.width*blockSize.height + ty*blockSize.height + tx] = 0;
		ushort tempmax=0;
 
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
 
 
__global__ void cuda_kernel(uchar *surface, int width, int height, cudaExtent volumeSize, float3 sdot, 
							float3 vDir, float3 vXcross, float3 vYcross, float zResolution, float blockResolution)
{
    int tx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
 
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (tx >= width || ty >= height) return;
 
	sdot = sdot + (tx-width/2)*vXcross + (ty-height/2)*vYcross;
 
	float t[2] = {0.0f, 1000.0f};
	GetRayBound(t, sdot, vDir, volumeSize); //t1, t2받아오기
 
	float4 intensity = {0.0f};
	float alpha = 0.0f;
	bool bShading=false;
	bool bSkipping = false;
 
	for(float i=t[0]; i<t[1]; i+=1.0f){
 
		float3 render={0.0f, 0.0f, 0.0f};
		render = sdot + i*vDir;
 
		float block_den = tex3D(tex_block, (int)(render.x*blockResolution), (int)(render.y*blockResolution), 
								int(render.z*blockResolution))*65535;
		float3 advanced  = {0.0f, 0.0f, 0.0f};
		if((int)block_den < alpha_start) { 
			int3 nowPos = {(int)(render.x*blockResolution), (int)(render.y*blockResolution), 
							(int)(render.z*blockResolution)};
			int3 advPos;
			do {
				i += 1.0f;
				advanced = sdot + i*vDir;
				advPos.x = (int)(advanced.x*blockResolution);
				advPos.y = (int)(advanced.y*blockResolution);
				advPos.z = (int)(advanced.z*blockResolution);
 
			} while ( nowPos.x == advPos.x &&
					  nowPos.y == advPos.y &&
					  nowPos.z == advPos.z);
			i -= 1.0f;
			bShading=true;
			bSkipping=true;
			continue;
		}
 
		float den = tex3D(tex_volume, render.x, render.y, render.z)*65535;
		//float den_next = tex3D(tex_volume, render.x+startvec.x, render.y+startvec.y, render.z+startvec.z)*4095; //next voxel
	
		float4 samplecolor = tex1D(tex_TF, den);
		//float4 samplecolor = tex3D(tex_TF2d, den, den_next, 0); //pre-integral 
 
		if(bSkipping){
			float3 prevpos = sdot +(i-1)*vDir;
			float den_prev = tex3D(tex_volume, prevpos.x, prevpos.y, prevpos.z)*65535;
			float4 prevcolor = tex1D(tex_TF, den_prev);
		
			samplecolor +=  (1.0f-samplecolor.w)*prevcolor*prevcolor.w;
		}
		bSkipping=false;
 
		if(samplecolor.w < 0.01f) {} else
		if(samplecolor.w > 0.001f && bShading){
			//------------------------------------------------------------------------
			//shading1 - local - NL을 뽑아내자
			//float shading1 = 0.0f;
			float3 nV = {0.0, 0.0, 0.0};
			float3 lV = {0.0, 0.0, 0.0};
 
			lV = vDir;
 
			float x_plus = tex3D(tex_volume, render.x+1, render.y, render.z)*65535;
			float x_minus = tex3D(tex_volume, render.x-1, render.y, render.z)*65535;
 
			float y_plus = tex3D(tex_volume, render.x, render.y+1, render.z)*65535;
			float y_minus = tex3D(tex_volume, render.x, render.y-1, render.z)*65535;
 
			float z_plus = tex3D(tex_volume, render.x, render.y, render.z+1)*65535;
			float z_minus = tex3D(tex_volume, render.x, render.y, render.z-1)*65535;
 
			nV.x = (x_plus - x_minus);
			nV.y = (y_plus - y_minus);
			nV.z = (z_plus - z_minus)*(float)zResolution;
 
			nV = cudaNormalize(nV);
 
			float NL = 0.0f;
			NL = lV.x*nV.x + lV.y*nV.y + lV.z*nV.z;
 
			if(NL < 0.0f) NL = 0.0f;
 
			float localShading = 0.3 + 0.7*NL;
 
			samplecolor.x *= localShading;
			samplecolor.y *= localShading;
			samplecolor.z *= localShading;
		} else
		{
			const float fCutPlaneShading = 0.0f;
			samplecolor = samplecolor*fCutPlaneShading;
		
		}
		bShading = true;
 
		intensity.x += (1.0f-alpha)*samplecolor.x*samplecolor.w;
		intensity.y += (1.0f-alpha)*samplecolor.y*samplecolor.w;
		intensity.z += (1.0f-alpha)*samplecolor.z*samplecolor.w;
		alpha += (1.0f-alpha)*samplecolor.w;
 
		if(alpha > 0.95f)
			break;
 
	}
 
	surface[(ty*width + tx)*3 + 0] = intensity.x;
	surface[(ty*width + tx)*3 + 1] = intensity.y;
	surface[(ty*width + tx)*3 + 2] = intensity.z;
}
 
__global__ void cuda_kernel_AO(uchar *surface, int width, int height, cudaExtent volumeSize, float3 sdot, 
							float3 vDir, float3 vXcross, float3 vYcross, float zResolution, float blockResolution,
							float* probability_k, float3 factor)
{
    int tx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
 
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (tx >= width || ty >= height) return;
 
	sdot = sdot + (tx-width/2)*vXcross + (ty-height/2)*vYcross;
 
	float t[2] = {0.0f, 1000.0f};
	GetRayBound(t, sdot, vDir, volumeSize); //t1, t2받아오기
 
	float4 intensity = {0.0f};
	float alpha = 0.0f;
	bool bShading=false;
	bool bSkipping = false;
 
	for(float i=t[0]; i<t[1]; i+=1.0f){
 
		float3 render={0.0f, 0.0f, 0.0f};
		render = sdot + i*vDir;
 
		float block_den = tex3D(tex_block, (int)(render.x*blockResolution), (int)(render.y*blockResolution), 
								int(render.z*blockResolution))*65535;
		float3 advanced  = {0.0f, 0.0f, 0.0f};
		if((int)block_den < alpha_start) { 
			int3 nowPos = {(int)(render.x*blockResolution), (int)(render.y*blockResolution), 
							(int)(render.z*blockResolution)};
			int3 advPos;
			do {
				i += 1.0f;
				advanced = sdot + i*vDir;
				advPos.x = (int)(advanced.x*blockResolution);
				advPos.y = (int)(advanced.y*blockResolution);
				advPos.z = (int)(advanced.z*blockResolution);
 
			} while ( nowPos.x == advPos.x &&
					  nowPos.y == advPos.y &&
					  nowPos.z == advPos.z);
			i -= 1.0f;
			bShading=true;
			bSkipping=true;
			continue;
		}
 
		float den = tex3D(tex_volume, render.x, render.y, render.z)*65535;
		//float den_next = tex3D(tex_volume, render.x+startvec.x, render.y+startvec.y, render.z+startvec.z)*4095; //next voxel
	
		float4 samplecolor = tex1D(tex_TF, den);
		//float4 samplecolor = tex3D(tex_TF2d, den, den_next, 0); //pre-integral 
 
		if(bSkipping){
			float3 prevpos = sdot +(i-1)*vDir;
			float den_prev = tex3D(tex_volume, prevpos.x, prevpos.y, prevpos.z)*65535;
			float4 prevcolor = tex1D(tex_TF, den_prev);
		
			samplecolor +=  (1.0f-samplecolor.w)*prevcolor*prevcolor.w;
		}
		bSkipping=false;
 
		if(samplecolor.w < 0.01f) {} else
		if(samplecolor.w > 0.001f && bShading){
			//------------------------------------------------------------------------
			//shading1 - local - NL을 뽑아내자
			//float shading1 = 0.0f;
			float3 nV = {0.0, 0.0, 0.0};
			float3 lV = {0.0, 0.0, 0.0};
 
			lV = vDir;
 
			float x_plus = tex3D(tex_volume, render.x+1, render.y, render.z)*65535;
			float x_minus = tex3D(tex_volume, render.x-1, render.y, render.z)*65535;
 
			float y_plus = tex3D(tex_volume, render.x, render.y+1, render.z)*65535;
			float y_minus = tex3D(tex_volume, render.x, render.y-1, render.z)*65535;
 
			float z_plus = tex3D(tex_volume, render.x, render.y, render.z+1)*65535;
			float z_minus = tex3D(tex_volume, render.x, render.y, render.z-1)*65535;
 
			nV.x = (x_plus - x_minus);
			nV.y = (y_plus - y_minus);
			nV.z = (z_plus - z_minus)*(float)zResolution;
 
			nV = cudaNormalize(nV);
 
			float NL = 0.0f;
			NL = lV.x*nV.x + lV.y*nV.y + lV.z*nV.z;
 
			if(NL < 0.0f) NL = 0.0f;
 
			float localShading = 0.3 + 0.7*NL;
			//------------------------------------------------------------------------
			//shading2 - global
			nV *= 5.f;
			float Sigma = tex3D(tex_sigma, min(render.x-nV.x, (float)volumeSize.width), 
				min(render.y-nV.y, (float)volumeSize.height), min(render.z-nV.z, (float)volumeSize.depth));
			float Average = tex3D(tex_average, min(render.x-nV.x, (float)volumeSize.width), 
				min(render.y-nV.y, (float)volumeSize.height), min(render.z-nV.z, (float)volumeSize.depth));	
 
			//samplecolor = tex1D(tex_TF, Average);
 
			float sum = GetSum(Average, Sigma, alpha_start, alpha_end, probability_k); //1400, 2100 - alpha starat, end
 
			//if(x_plus > den && y_plus > den && z_plus > den )
			//	sum = sum/1.5f;
 
			float shading2 = 1.0f - min(max((sum*2.0f - 0.5f), 0.0f), 1.0f); //global shding value 조정
			//shading2 = 1.0f-shading2;
 
			float shading = factor.x + factor.y*shading2*shading2 + factor.z*NL; //factor1,2,3
 
			samplecolor.x *= shading2;
			samplecolor.y *= shading2;
			samplecolor.z *= shading2;
		} else
		{
			const float fCutPlaneShading = 0.0f;
			samplecolor = samplecolor*fCutPlaneShading;
		
		}
		bShading = true;
 
		intensity.x += (1.0f-alpha)*samplecolor.x*samplecolor.w;
		intensity.y += (1.0f-alpha)*samplecolor.y*samplecolor.w;
		intensity.z += (1.0f-alpha)*samplecolor.z*samplecolor.w;
		alpha += (1.0f-alpha)*samplecolor.w;
 
		if(alpha > 0.95f)
			break;
 
	}
 
	surface[(ty*width + tx)*3 + 0] = intensity.x;
	surface[(ty*width + tx)*3 + 1] = intensity.y;
	surface[(ty*width + tx)*3 + 2] = intensity.z;
}
 
ushort* make_blockVolume(ushort* image, cudaExtent blockSize, cudaExtent volumeSize)
{
	unsigned int vsize = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(ushort);
	unsigned int bsize = blockSize.width * blockSize.height * blockSize.depth * sizeof(ushort);
 
	ushort *dest, *dest_p, *image_p;
 
	dest = new ushort[bsize/sizeof(ushort)];
	memset((void*)dest, 0, bsize);
 
	cudaMalloc((void**)&image_p, vsize);
	cudaMemcpy(image_p, image, vsize, cudaMemcpyHostToDevice);
 
	cudaMalloc((void**)&dest_p, bsize);
 
	dim3 Db = dim3(32, 32);		// block dimensions are fixed to be 512 threads
    dim3 Dg = dim3((blockSize.width+Db.x-1)/Db.x, (blockSize.height+Db.y-1)/Db.y);
 
	makeBlock_kernel<<<Dg, Db>>>(image_p, dest_p, blockSize, volumeSize);
 
	cudaMemcpy(dest, dest_p, bsize, cudaMemcpyDeviceToHost);
 
	cudaFree(image_p);
	cudaFree(dest_p);
 
	return dest;
 
}
 
 
void Run_Kernel(uchar* surface, const int imgsize[2], cudaExtent volumeSize, ushort* pVol,
				float zResolution, float blockResolution, const float *ViewingPoint)
{
	printf("-GPU render : Basic\n");
	//---------------------------------------------------------------
	//시점, 카메라 각도 설정
	float3 volCenter = {volumeSize.width/2.0f, volumeSize.height/2.0f, volumeSize.depth/2.0f};
	float3 sdot={ViewingPoint[0], ViewingPoint[1], ViewingPoint[2]}, vUp={0.0f, 0.0f, 1.0f};
	float3 frontView = {volumeSize.width/2.f, volumeSize.height, volumeSize.depth/2.f};
	
	float3 vDir, vXCross, vYcross, front;
 
	front = frontView-volCenter;
	front = normalize(front);
 
	vDir = volCenter-sdot;
	vDir = normalize(vDir);
 
	float3 temp_z = {0.f, vDir.y, vDir.z};
	temp_z = normalize(temp_z);
	if(dot(front, temp_z) < 0.f)
		vUp.z = -1.0f;
	
	vXCross = cross(vUp, vDir);
	vXCross = normalize(vXCross);
 
	vYcross = cross(vDir, vXCross);
	vYcross = normalize(vYcross);
	//---------------------------------------------------------------
 
	uchar* surface_k;
	cudaMalloc((void**)&surface_k, imgsize[0]*imgsize[1]*3*sizeof(uchar));
	cudaMemset(surface_k, 0, imgsize[0]*imgsize[1]*3*sizeof(uchar));
	cudaMemcpy(surface_k, surface, imgsize[0]*imgsize[1]*3*sizeof(uchar), cudaMemcpyHostToDevice);
 
    //dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Db = dim3(32, 32);		// block dimensions are fixed to be 512 threads
    dim3 Dg = dim3((imgsize[0]+Db.x-1)/Db.x, (imgsize[1]+Db.y-1)/Db.y);
 
    cuda_kernel<<<Dg,Db>>>(surface_k, imgsize[0], imgsize[1], volumeSize, sdot, vDir, 
		vXCross, vYcross, zResolution, blockResolution);
    if (cudaGetLastError() != cudaSuccess)
        printf("cuda_kernel() failed to launch error = %d\n", cudaGetLastError());    
 
	cudaMemcpy(surface, surface_k, imgsize[0]*imgsize[1]*3*sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaFree(surface_k);
}
 
 
void Run_Kernel_AO(uchar* surface, const int imgsize[2], cudaExtent volumeSize, ushort* pVol,
				float zResolution, float blockResolution, float probability[310], float factor[3], const float *ViewingPoint)
{
	printf("-GPU render : AO \n");
	//---------------------------------------------------------------
	//시점, 카메라 각도 설정
	float3 volCenter = {volumeSize.width/2.0f, volumeSize.height/2.0f, volumeSize.depth/2.0f};
	float3 sdot={ViewingPoint[0], ViewingPoint[1], ViewingPoint[2]}, vUp={0.0f, 0.0f, 1.0f};
	float3 frontView = {volumeSize.width/2.f, volumeSize.height, volumeSize.depth/2.f};
	
	float3 vDir, vXCross, vYcross, front;
 
	front = frontView-volCenter;
	front = normalize(front);
 
	vDir = volCenter-sdot;
	vDir = normalize(vDir);
 
	float3 temp_z = {0.f, vDir.y, vDir.z};
	temp_z = normalize(temp_z);
	if(dot(front, temp_z) < 0.f)
		vUp.z = -1.0f;
	
	vXCross = cross(vUp, vDir);
	vXCross = normalize(vXCross);
 
	vYcross = cross(vDir, vXCross);
	vYcross = normalize(vYcross);
	//---------------------------------------------------------------
 
	float* probability_k;
	cudaMalloc((void**)&probability_k, 310*sizeof(float));
	cudaMemset(probability_k, 0, 310*sizeof(float));
	cudaMemcpy(probability_k, probability, 310*sizeof(float), cudaMemcpyHostToDevice);
	
	float3 factor3 ={factor[0], factor[1], factor[2]};
 
	uchar* surface_k;
	cudaMalloc((void**)&surface_k, imgsize[0]*imgsize[1]*3*sizeof(uchar));
	cudaMemset(surface_k, 0, imgsize[0]*imgsize[1]*3*sizeof(uchar));
	cudaMemcpy(surface_k, surface, imgsize[0]*imgsize[1]*3*sizeof(uchar), cudaMemcpyHostToDevice);
 
    //dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Db = dim3(32, 32);		// block dimensions are fixed to be 512 threads
    dim3 Dg = dim3((imgsize[0]+Db.x-1)/Db.x, (imgsize[1]+Db.y-1)/Db.y);
 
    cuda_kernel_AO<<<Dg,Db>>>(surface_k, imgsize[0], imgsize[1], volumeSize, sdot, vDir, 
		vXCross, vYcross, zResolution, blockResolution, probability_k, factor3);
 
    if (cudaGetLastError() != cudaSuccess)
        printf("cuda_kernel() failed to launch error = %d\n", cudaGetLastError());
    
	cudaMemcpy(surface, surface_k, imgsize[0]*imgsize[1]*3*sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaFree(surface_k);
}
 
 
void initTF2dTexture(float4 *h_volume, int x, int y, int z)
{
	cudaExtent Size = make_cudaExtent(x, y, z);
    // create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    checkCudaErrors( cudaMalloc3DArray(&d_TF2dArray, &channelDesc, Size, 0) );
 
    // copy data to 3D array
    cudaMemcpy3DParms myParams = {0};
    myParams.srcPtr   = make_cudaPitchedPtr((void*)h_volume, x*sizeof(float4), x, y);
    myParams.dstArray = d_TF2dArray;
    myParams.extent   = Size;
    myParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors( cudaMemcpy3D(&myParams) );
 
    // set texture parameters
    tex_TF2d.normalized = false;                      // access with normalized texture coordinates
    tex_TF2d.filterMode = cudaFilterModePoint;      // linear interpolation
    tex_TF2d.channelDesc = channelDesc;
	tex_TF2d.addressMode[0] = cudaAddressModeBorder;   // wrap texture coordinates
    tex_TF2d.addressMode[1] = cudaAddressModeBorder;
    tex_TF2d.addressMode[2] = cudaAddressModeBorder;
 
 
    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex_TF2d, d_TF2dArray, channelDesc));
}
 
 
 
__global__ void TF2d_kernel(float4* TF2d_k, int TFSize)
{
	int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
 
	if(x>=TFSize || y>=TFSize)
		return;
 
	//float4 result;				//1번 방법 - pre-integral : OTF 뾰족하게 해도 한겹만 나오게 할수있다.
	//float4 temp = {0.0f};
	//
	//if(y > x){
	//	for(int i=x; i<y; i++){
	//		temp = tex1D(tex_TF, i);
 
	//		float diff = i-x;
 
	//		if(diff == 0.0f)
	//			diff = 1.0f;
 
	//		temp.w = 1.0f-pow(1-temp.w, 1/diff);
 
	//		result.x += (1-result.w)*temp.x*temp.w;
	//		result.y += (1-result.w)*temp.y*temp.w;
	//		result.z += (1-result.w)*temp.z*temp.w;
	//		result.w += (1-result.w)*temp.w;
	//	}
	//}
	//else if(x > y){
	//	for(int i=y; i<x; i++){
	//		temp = tex1D(tex_TF, i);
 
	//		float diff = i-y;
 
	//		if(diff == 0.0f)
	//			diff = 1.0f;
 
	//		temp.w = 1.0f-pow(1-temp.w, 1/diff);
 
	//		result.x += (1-result.w)*temp.x*temp.w;
	//		result.y += (1-result.w)*temp.y*temp.w;
	//		result.z += (1-result.w)*temp.z*temp.w;
	//		result.w += (1-result.w)*temp.w;
	//	}
	//}
	//else {
	//	result.x = 255.0f;
	//	result.y = 255.0f;
	//	result.z = 255.0f;
	//	result.w = 0.0f;
	//}
 
	float4 temp;					//2번 방법 - 1번방법보다 물결무늬가 덜 생긴다 : summed 2d table
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
		temp = tex1D(tex_TF, i);
 
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
 
 
void init_TF2d(int TFSize)
{
	int size = TFSize*TFSize;
	float4* TF2d_k;
	cudaMalloc((void**)&TF2d_k, size*sizeof(float4));
	cudaMemset(TF2d_k, 0, size*sizeof(float4));
 
	dim3 Db = dim3( 16, 16 ); 
    dim3 Dg = dim3( 256, 256 );
 
 
	TF2d_kernel<<<Dg, Db>>>(TF2d_k, TFSize); //pre-integral OTF init kernel - threads 4096*4096
 
	float4* TF2d;
	TF2d = new float4[size];
	memset(TF2d, 0, size*sizeof(float4));
 
	cudaMemcpy(TF2d, TF2d_k, size*sizeof(float4), cudaMemcpyDeviceToHost);
 
	cudaFree(TF2d_k);
 
	initTF2dTexture(TF2d, TFSize, TFSize, 1);
 
	delete[] TF2d;
 
 
}
 
extern "C"
void GPU_Render(uchar *image, int imgsize[2], ushort* pVol, int dim[3], 
				TF *transfer, int tf_size, double zResolution, bool &bInitVol, bool &bInitTF, float *ViewingPoint)
{
	float4 *tf_cuda;
	if(!bInitTF){
		printf("-init TF texture memory - GPU\n");
		tf_cuda = new float4[tf_size];
		for(int i=0; i<tf_size; i++){
			tf_cuda[i].x = transfer[i].R;
			tf_cuda[i].y = transfer[i].G;
			tf_cuda[i].z = transfer[i].B;
			tf_cuda[i].w = transfer[i].alpha;
		}
		initTFTexture(tf_size, tf_cuda);
	}
 
	cudaExtent volume_dim_block, volume_dim;
	float blockResolution = 0.25f;
	volume_dim = make_cudaExtent(dim[0], dim[1], dim[2]);
	volume_dim_block = make_cudaExtent(dim[0]*blockResolution, dim[1]*blockResolution, dim[2]*blockResolution);
 
	ushort *pVol_block;
	if(!bInitVol){
		printf("-init Volume texture memory - GPU\n");
		pVol_block = make_blockVolume(pVol, volume_dim_block, volume_dim);
 
		initVolume(pVol, volume_dim , sizeof(ushort));
		initBlockTexture(pVol_block, volume_dim_block, sizeof(ushort));
	}
 
	Run_Kernel(image, imgsize, volume_dim, pVol, (float)zResolution, blockResolution, ViewingPoint);
 
	if(!bInitVol){
		delete[] pVol_block;
		bInitVol = true;
	}
	if(!bInitTF){
		delete[] tf_cuda;
		bInitTF = true;
	}
	
}
 
extern "C"
void GPU_Render_AO(uchar *image, int imgsize[2], ushort* pVol, int dim[3], 
				TF *transfer, int tf_size, double zResolution, bool &bInitVol, bool &bInitTF,
				float *Avg, float *Sig, bool &m_bInitAvgSig, float probability[310], float factor[3],
				float *ViewingPoint)
{
	float4 *tf_cuda;
	if(!bInitTF){
		printf("-init TF texture memory - GPU\n");
		tf_cuda = new float4[tf_size];
		for(int i=0; i<tf_size; i++){
			tf_cuda[i].x = transfer[i].R;
			tf_cuda[i].y = transfer[i].G;
			tf_cuda[i].z = transfer[i].B;
			tf_cuda[i].w = transfer[i].alpha;
		}
		initTFTexture(tf_size, tf_cuda);
	}
 
	cudaExtent volume_dim_block, volume_dim;
	float blockResolution = 0.25f;
	volume_dim = make_cudaExtent(dim[0], dim[1], dim[2]);
	volume_dim_block = make_cudaExtent(dim[0]*blockResolution, dim[1]*blockResolution, dim[2]*blockResolution);
 
	ushort *pVol_block;
	if(!bInitVol){
		printf("-init Volume texture memory - GPU\n");
		pVol_block = make_blockVolume(pVol, volume_dim_block, volume_dim);
 
		initVolume(pVol, volume_dim , sizeof(ushort));
		initBlockTexture(pVol_block, volume_dim_block, sizeof(ushort));
	}
	if(!m_bInitAvgSig && Avg != NULL && Sig != NULL){
		initAvgVolume(Avg, volume_dim, sizeof(float));
		initSigVolume(Sig, volume_dim, sizeof(float));
	}
 
	Run_Kernel_AO(image, imgsize, volume_dim, pVol, (float)zResolution, blockResolution, probability, factor, ViewingPoint);
 
	if(!bInitVol){
		delete[] pVol_block;
		bInitVol = true;
	}
	if(!bInitTF){
		delete[] tf_cuda;
		bInitTF = true;
	}	
}
 
__global__ void cuda_kernel_test(ushort *new_vol_k, ushort *vol_k, int3 dim3, float *gaussianMask_k, int maskSize)
{
	int tx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
 
	if (tx >= dim3.x-2 || ty >= dim3.y-2) return;
	if (tx <= 2 || ty <= 2) return;
 
	int size = maskSize/2;
 
	for(int tz=1; tz<dim3.z; tz++)
	{
		double sum = 0.f;
		for(int i=-size; i<size+1; i++)
		{
			for(int j=-size; j<size+1; j++)
			{
				for(int k=-size; k<size+1; k++)
				{
					int z=k+1, y=j+1, x=i+1;
					sum += vol_k[(tz+k)*dim3.x*dim3.y + (ty+j)*dim3.y + (tx+i)]*gaussianMask_k[z*maskSize*maskSize + y*maskSize + x];
				}
			}
		}
		new_vol_k[tz*dim3.x*dim3.y + ty*dim3.y + tx] = (ushort)sum;
	}
 
}
 
extern "C"
void testSmoothFilter(ushort* pVol, int *dim)
{
	printf("-GPU testSmoothFilter \n");
 
	float fSigma=0.8f;
	float gaussianMask[27];
	int maskSize = 3;
	int allocSize= maskSize*maskSize*maskSize;
 
	float sum=0.f;
	for(int i=0; i<maskSize; i++)
	{
		float z = fabs((float)i-1.f);
		for(int j=0; j<maskSize; j++)
		{
			float y = fabs((float)j-1.f);
			for(int k=0; k<maskSize; k++)
			{
				float x = fabs((float)k-1.f); 
				float fDist = x+y+z;
				sum += gaussianMask[k*maskSize*maskSize + j*maskSize + i] = 
					exp(-(fDist*fDist)/(2.f*fSigma*fSigma))/(sqrtf(2.f*PI)*fSigma);
			}
		}
	}
	for(int i=0; i<maskSize; i++)
	{
		for(int j=0; j<maskSize; j++)
		{
			for(int k=0; k<maskSize; k++)
			{
				gaussianMask[k*maskSize*maskSize + j*maskSize + i] /= sum;
			}
		}
	}
 
	//printf("%f %f %f\n", gaussianMask[0], gaussianMask[1], gaussianMask[2]);
	//printf("%f %f %f\n", gaussianMask[3], gaussianMask[4], gaussianMask[5]);
	//printf("%f %f %f\n\n", gaussianMask[6], gaussianMask[7], gaussianMask[8]);
 
	//printf("%f %f %f\n", gaussianMask[9], gaussianMask[10], gaussianMask[11]);
	//printf("%f %f %f\n", gaussianMask[12], gaussianMask[13], gaussianMask[14]);
	//printf("%f %f %f\n\n", gaussianMask[15], gaussianMask[16], gaussianMask[17]);
 
	//printf("%f %f %f\n", gaussianMask[18], gaussianMask[19], gaussianMask[20]);
	//printf("%f %f %f\n", gaussianMask[21], gaussianMask[22], gaussianMask[23]);
	//printf("%f %f %f\n\n", gaussianMask[24], gaussianMask[25], gaussianMask[26]);
 
	float* gaussianMask_k;
	cudaMalloc((void**)&gaussianMask_k, allocSize*sizeof(float));
	cudaMemset(gaussianMask_k, 0, allocSize*sizeof(float));
	cudaMemcpy(gaussianMask_k, gaussianMask, allocSize*sizeof(float), cudaMemcpyHostToDevice);
 
	ushort *pVol_k, *new_pVol_k;
	int vol_size = dim[0]*dim[1]*dim[2];
	int3 vol_dim3 = {dim[0], dim[1], dim[2]};
 
	cudaMalloc((void**)&pVol_k, vol_size*sizeof(ushort));
	cudaMemset(pVol_k, 0, vol_size*sizeof(ushort));
	cudaMemcpy(pVol_k, pVol, vol_size*sizeof(ushort), cudaMemcpyHostToDevice);
 
	cudaMalloc((void**)&new_pVol_k, vol_size*sizeof(ushort));
	cudaMemset(new_pVol_k, 0, vol_size*sizeof(ushort));
 
	dim3 Db = dim3(32, 32);		// block dimensions are fixed to be 512 threads
    dim3 Dg = dim3((dim[0]+Db.x-1)/Db.x, (dim[1]+Db.y-1)/Db.y);
 
    cuda_kernel_test<<<Dg,Db>>>(new_pVol_k, pVol_k, vol_dim3, gaussianMask_k, maskSize);
 
    if (cudaGetLastError() != cudaSuccess)
        printf("cuda_kernel() failed to launch error = %d\n", cudaGetLastError());
    
	memset(pVol, 0, sizeof(ushort)*vol_size);
	cudaMemcpy(pVol, new_pVol_k, vol_size*sizeof(ushort), cudaMemcpyDeviceToHost);
 
	cudaFree(pVol_k);
	cudaFree(new_pVol_k);
	cudaFree(gaussianMask_k);
 
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
	float4 transferFunc1[256]={0.0f};
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
	
	transferFunc1[0].w= transferFunc[0].w;
	transferFunc1[0].x= transferFunc[0].x * transferFunc[0].w;
	transferFunc1[0].y= transferFunc[0].y * transferFunc[0].w;
	transferFunc1[0].z= transferFunc[0].z * transferFunc[0].w;		
	
	for(int i=1; i<256; i++)
	{		
		

		transferFunc1[i].w += transferFunc1[i-1].w + transferFunc[i].w;
		transferFunc1[i].x += transferFunc1[i-1].x + transferFunc[i].x * transferFunc[i].w;
		transferFunc1[i].y += transferFunc1[i-1].y + transferFunc[i].y * transferFunc[i].w;
		transferFunc1[i].z += transferFunc1[i-1].z + transferFunc[i].z * transferFunc[i].w;
		
		transferFunc1[i].w =(transferFunc1[i].w/256.0f);
		transferFunc1[i].x =(transferFunc1[i].x/256.0f);
		transferFunc1[i].y =(transferFunc1[i].y/256.0f);
		transferFunc1[i].z =(transferFunc1[i].z/256.0f);
		//printf("%f %f\n",transferFunc1[i].w/256,transferFunc1[i].x/256);
		//printf("%f,%f,%f,%f\n",tempA[i],OTF_2Da[before],tempG[i],OTF_2Dg[before]);

	}
	//for(int x=0; x<256; x++){
	//	for(int y=0; y<256; y++){

	//		float4 result;
	//		float4 temp={0.0f};

	//		if(y > x){
	//			for(int i=x; i<y; i++){
	//				temp.x = transferFunc[i].x;
	//				temp.y = transferFunc[i].y;
	//				temp.z = transferFunc[i].z;
	//				temp.w = transferFunc[i].w;
	//				
	//				float diff = i-x;

	//				if(diff == 0.0f)
	//					diff = 1.0f;

	//				temp.w = 1.0f-pow(1-temp.w, 1/diff);

	//				result.x += (1-result.w)*temp.x*temp.w;
	//				result.y += (1-result.w)*temp.y*temp.w;
	//				result.z += (1-result.w)*temp.z*temp.w;
	//				result.w += (1-result.w)*temp.w;
	//			}
	//		}
	//		else if(x > y){
	//			for(int i=y; i<x; i++){
	//				temp.x = transferFunc[i].x;
	//				temp.y = transferFunc[i].y;
	//				temp.z = transferFunc[i].z;
	//				temp.w = transferFunc[i].w;

	//				float diff = i-y;

	//				if(diff == 0.0f)
	//					diff = 1.0f;

	//				temp.w = 1.0f-pow(1-temp.w, 1/diff);

	//				result.x += (1-result.w)*temp.x*temp.w;
	//				result.y += (1-result.w)*temp.y*temp.w;
	//				result.z += (1-result.w)*temp.z*temp.w;
	//				result.w += (1-result.w)*temp.w;
	//			}
	//		}
	//		else {
	//			result.x = 1.0f;
	//			result.y = 1.0f;
	//			result.z = 1.0f;
	//			result.w = 0.0f;
	//		}
	//		OTF_2D[256*x + y].sum_R = result.x;
	//		OTF_2D[256*x + y].sum_G = result.y;
	//		OTF_2D[256*x + y].sum_B = result.z;
	//		OTF_2D[256*x + y].sum_a = result.w;
	//	}
	//}
	//struct OTF_2D *p;
	//p=getPre_integration();
	//for(int i=0; i<256; i++)
	//{
	//	printf("%f\n",transferFunc1[i].x);
	//}
	//-------------------------------------------------------------------
	// create transfer function texture
  //  float4 transferFunc[] =
  //  {
  //     /* {  0.0, 0.0, 0.0, 0.0, },
  //      {  1.0, 0.0, 0.0, 1.0, },
  //      {  1.0, 0.5, 0.0, 1.0, },
  //      {  1.0, 1.0, 0.0, 1_.0, },
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

   // create 3D array

	//cudaExtent Size2 = make_cudaExtent(256, 256, 1);
 //   cudaChannelFormatDesc channelDesc3 = cudaCreateChannelDesc<float4>();
 //   checkCudaErrors(cudaMalloc3DArray(&d_transferFuncArray1, &channelDesc3, Size2));

 //   // copy data to 3D array
 //   cudaMemcpy3DParms copyParams3 = {0};
 //   copyParams3.srcPtr   = make_cudaPitchedPtr(OTF_2D, Size2.width*sizeof(float4), Size2.width, Size2.height);
 //   copyParams3.dstArray = d_transferFuncArray1;
 //   copyParams3.extent   = Size2;
 //   copyParams3.kind     = cudaMemcpyHostToDevice;
 //   checkCudaErrors(cudaMemcpy3D(&copyParams3));

 //   // set texture parameters
 //   tex.normalized = true;                      // access with normalized texture coordinates
 //   tex.filterMode = cudaFilterModeLinear;      // linear interpolation
 //   tex.addressMode[0] = cudaAddressModeBorder;  // clamp texture coordinates
 //   tex.addressMode[1] = cudaAddressModeBorder;
 //   // tex.addressMode[2] = cudaAddressModeBorder;
 //   // bind array to 3D texture
 //   checkCudaErrors(cudaBindTextureToArray(transferTex1, d_transferFuncArray1, channelDesc3));
//////////////////////////////////////////////////////////////////////////////////////////////
	cudaChannelFormatDesc channelDesc3 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray1;
    checkCudaErrors(cudaMallocArray(&d_transferFuncArray1, &channelDesc3, sizeof(transferFunc1)/sizeof(float4), 1));
    checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray1, 0, 0, transferFunc1, sizeof(transferFunc1), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(transferTex1, d_transferFuncArray1, channelDesc3));


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

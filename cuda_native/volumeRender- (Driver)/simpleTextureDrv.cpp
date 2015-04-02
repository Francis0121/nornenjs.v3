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

/*
* This sample demonstrates how use texture fetches in CUDA
*
* This sample takes an input PGM image (image_filename) and generates
* an output PGM image (image_filename_out).  This CUDA kernel performs
* a simple 2D transform (rotation) on the texture coordinates (u,v).
* The results between simpleTexture and simpleTextureDrv are identical.
* The main difference is the implementation.  simpleTextureDrv makes calls
* to the CUDA driver API and demonstrates how to use cuModuleLoad to load
* the CUDA ptx (*.ptx) kernel just prior to kernel launch.
*
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include <windows.h>
// includes, project
#include <helper_functions.h>
#include <helper_math.h>
// includes, CUDA
#include <cuda.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>

using namespace std;
typedef unsigned char uchar;
const char *volumeFilename = "Bighead.den";

cudaExtent volumeSize = make_cudaExtent(256, 256, 225);
typedef unsigned char VolumeType;

float c_invViewMatrix[12];
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

static CUresult initCUDA(int argc, char **argv, CUfunction *);

const char *sSDKsample = "simpleTextureDrv (Driver API)";

//define input ptx file for different platforms
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#define PTX_FILE "simpleTexture_kernel64.ptx"
#define CUBIN_FILE "simpleTexture_kernel64.cubin"
#else
#define PTX_FILE "simpleTexture_kernel32.ptx"
#define CUBIN_FILE "simpleTexture_kernel32.cubin"
#endif


LARGE_INTEGER start, end, liFrequency; //시간 측정 변수

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

inline int cudaDeviceInit(int ARGC, char **ARGV)
{
    int cuDevice = 0;
    int deviceCount = 0;
    CUresult err = cuInit(0);

    if (CUDA_SUCCESS == err)
    {
        checkCudaErrors(cuDeviceGetCount(&deviceCount));
    }

    if (deviceCount == 0)
    {
        fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
        exit(EXIT_FAILURE);
    }

    int dev = 0;
    dev = getCmdLineArgumentInt(ARGC, (const char **) ARGV, "device=");

    if (dev < 0)
    {
        dev = 0;
    }

    if (dev > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> cudaDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
        fprintf(stderr, "\n");
        return -dev;
    }

    checkCudaErrors(cuDeviceGet(&cuDevice, dev));
    char name[100];
    cuDeviceGetName(name, 100, cuDevice);

    if (checkCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == false)
    {
        printf("> Using CUDA Device [%d]: %s\n", dev, name);
    }

    return dev;
}

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { -1, 192},   // Default case
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index].Cores);
    return nGpuArchCoresPerSM[index].Cores;
}
// end of GPU Architecture definitions

// This function returns the best GPU based on performance
inline int getMaxGflopsDeviceId()
{
    CUdevice current_device = 0, max_perf_device = 0;
    int device_count     = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, best_SM_arch     = 0;
    int major = 0, minor = 0, multiProcessorCount, clockRate;

    cuInit(0);
    checkCudaErrors(cuDeviceGetCount(&device_count));

    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, current_device));

        if (major > 0 && major < 9999)
        {
            best_SM_arch = MAX(best_SM_arch, major);
        }

        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        checkCudaErrors(cuDeviceGetAttribute(&multiProcessorCount,
                                             CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                             current_device));
        checkCudaErrors(cuDeviceGetAttribute(&clockRate,
                                             CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                             current_device));
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, current_device));

        if (major == 9999 && minor == 9999)
        {
            sm_per_multiproc = 1;
        }
        else
        {
            sm_per_multiproc = _ConvertSMVer2Cores(major, minor);
        }

        int compute_perf  = multiProcessorCount * sm_per_multiproc * clockRate;

        if (compute_perf  > max_compute_perf)
        {
            // If we find GPU with SM major > 2, search only these
            if (best_SM_arch > 2)
            {
                // If our device==dest_SM_arch, choose this, or else pass
                if (major == best_SM_arch)
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            }
            else
            {
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
            }
        }

        ++current_device;
    }

    return max_perf_device;
}

// General initialization call to pick the best CUDA Device
inline CUdevice findCudaDevice(int argc, char **argv, int *p_devID)
{
    CUdevice cuDevice;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = cudaDeviceInit(argc, argv);

        if (devID < 0)
        {
            printf("exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        char name[100];
        devID = getMaxGflopsDeviceId();
        checkCudaErrors(cuDeviceGet(&cuDevice, devID));
        cuDeviceGetName(name, 100, cuDevice);
        printf("> Using CUDA Device [%d]: %s\n", devID, name);
    }

    cuDeviceGet(&cuDevice, devID);

    if (p_devID)
    {
        *p_devID = devID;
    }

    return cuDevice;
}
// end of CUDA Helper Functions

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;

void
showHelp()
{
    printf("\n> [%s] Command line options\n", sSDKsample);
    printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        showHelp();
        return 0;
    }

    runTest(argc, argv);
}
uchar *loadRawFile2(char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    uchar *data = (uchar *)malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    printf("Read '%s', %d bytes\n", filename, read);

    return data;
}
void *loadRawFile(char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    printf("Read '%s', %d bytes\n", filename, read);

    return data;
}
void ScreenCapture( const char *strFilePath ,uint *d_output)
{
    //비트맵 파일 처리를 위한 헤더 구조체
    BITMAPFILEHEADER    BMFH;
    BITMAPINFOHEADER    BMIH;
 
    int nWidth = 0;
    int nHeight = 0;
    unsigned long dwQuadrupleWidth = 0;     //LJH 추가, 가로 사이즈가 4의 배수가 아니라면 4의 배수로 만들어서 저장
 
   // GLbyte *pPixelData = NULL;              //front buffer의 픽셀 값들을 얻어 오기 위한 버퍼의 포인터
 
    nWidth  = 256;     //(나의 경우)리눅스에서의 경우 해상도 고정이므로 그 값을 입력
    nHeight = 256;
 
    //4의 배수인지 아닌지 확인해서 4의 배수가 아니라면 4의 배수로 맞춰준다.
    dwQuadrupleWidth = ( nWidth % 4 ) ? ( ( nWidth ) + ( 4 - ( nWidth % 4 ) ) ) : ( nWidth );
 
    //비트맵 파일 헤더 처리
    BMFH.bfType  = 0x4D42;      //B(42)와 M(4D)에 해당하는 ASCII 값을 넣어준다.
    //바이트 단위로 전체파일 크기
    BMFH.bfSize  = sizeof( BITMAPFILEHEADER ) + sizeof( BITMAPINFOHEADER ) + ( dwQuadrupleWidth * 3 * nHeight );
    //영상 데이터 위치까지의 거리
    BMFH.bfOffBits = sizeof( BITMAPFILEHEADER ) + sizeof( BITMAPINFOHEADER );
 
    //비트맵 인포 헤더 처리
    BMIH.biSize             = sizeof( BITMAPINFOHEADER );       //이 구조체의 크기
    BMIH.biWidth            = nWidth;                           //픽셀 단위로 영상의 폭
    BMIH.biHeight           = nHeight;                          //영상의 높이
    BMIH.biPlanes           = 1;                                //비트 플레인 수(항상 1)
    BMIH.biBitCount         = 24;                               //픽셀당 비트수(컬러, 흑백 구별)
    BMIH.biCompression      = BI_RGB;                           //압축 유무
    BMIH.biSizeImage        = 256 * 3 * 256;					//영상의 크기
    BMIH.biXPelsPerMeter    = 0;                                //가로 해상도
    BMIH.biYPelsPerMeter    = 0;                                //세로 해상도
    BMIH.biClrUsed          = 0;                                //실제 사용 색상수
    BMIH.biClrImportant     = 0;                                //중요한 색상 인덱스
 
    //pPixelData = new GLbyte[ dwQuadrupleWidth * 3 * nHeight ];  //LJH 수정
 
    //프런트 버퍼로 부터 픽셀 정보들을 얻어온다.
    //glReadPixels(
    //    0, 0,                   //캡처할 영역의 좌측상단 좌표
    //    nWidth, nHeight,        //캡처할 영역의 크기
    //    GL_BGR,                 //캡처한 이미지의 픽셀 포맷
    //    GL_UNSIGNED_BYTE,       //캡처한 이미지의 데이터 포맷
    //    pPixelData              //캡처한 이미지의 정보를 담아둘 버퍼 포인터
    //    );
 
    {//저장 부분
        FILE *outFile = fopen( strFilePath, "wb" );
        if( outFile == NULL )
        {
            //에러 처리
            //printf( "에러" );
            //fclose( outFile );
        }
        fwrite( &BMFH, sizeof( unsigned char ), sizeof(BITMAPFILEHEADER), outFile );         //파일 헤더 쓰기
        fwrite( &BMIH, sizeof( unsigned char ), sizeof(BITMAPINFOHEADER), outFile );         //인포 헤더 쓰기
		for(int i=0; i<256; i++){
			for(int j=0; j<256; j++){
				fwrite((d_output+(i*256 +j+0)), sizeof( unsigned char ), 1, outFile );
				fwrite((d_output+(i*256 +j+1)), sizeof( unsigned char ), 1, outFile );
				fwrite((d_output+(i*256 +j+2)), sizeof( unsigned char ), 1, outFile );
			}
		}
        //fwrite( d_output, sizeof( uchar ), BMIH.biSizeImage, outFile );   //c_output파일로 읽은 데이터 쓰기
 
        fclose( outFile );  //파일 닫기
    }
 
    if ( d_output != NULL )
    {
        delete d_output;
    }
}
////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResults = true;

    // initialize CUDA
    CUfunction transform = NULL;

    if (initCUDA(argc, argv, &transform) != CUDA_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
	unsigned int maxtrixBufferSize = 12;
	unsigned int imageWidth=256;
	unsigned int imageHeight=256;
	float density = 0.05;
	float brightness = 1.0;
	float transferOffset = 0.0;
	float transferScale = 1.0;
	
    // load image from disk
    float *h_data = NULL;
   
	char *path = sdkFindFilePath(volumeFilename, argv[0]);
    if (path == 0)
    {
        printf("Error finding file '%s'\n", volumeFilename);
        exit(EXIT_FAILURE);
    }
    size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*sizeof(VolumeType);
    void *h_volume = loadRawFile(path, size);
    
    // allocate device memory for result
    CUdeviceptr d_data = (CUdeviceptr)NULL;
    checkCudaErrors(cuMemAlloc(&d_data, 256*256*4));
	
	c_invViewMatrix[0] = -1.0f;
	c_invViewMatrix[1] = 0.0f;
	c_invViewMatrix[2] = 0.0f;
	c_invViewMatrix[3] = 0.0f;
	c_invViewMatrix[4] = 0.0f;
	c_invViewMatrix[5] = 0.0f;
	c_invViewMatrix[6] = -1.0;
	c_invViewMatrix[7] = -3.0;
	c_invViewMatrix[8] = 0.0f;
	c_invViewMatrix[9] = -1.0;
    c_invViewMatrix[10] = 0.0f;
	c_invViewMatrix[11] = 0.0f;

	CUdeviceptr d_invViewMatrix = (CUdeviceptr)NULL;
    checkCudaErrors(cuMemAlloc(&d_invViewMatrix, 12*4));
    
	checkCudaErrors(cuMemcpyHtoD(d_invViewMatrix, c_invViewMatrix ,12*sizeof(float)));
	
    // allocate array and copy image data
    CUarray cu_array;
    CUDA_ARRAY3D_DESCRIPTOR desc;
    desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    desc.NumChannels = 1;
    desc.Width = 256;
    desc.Height = 256;
    desc.Depth = 225;
    desc.Flags=0;
    checkCudaErrors(cuArray3DCreate(&cu_array, &desc));

    CUDA_MEMCPY3D copyParam;
    memset(&copyParam, 0, sizeof(copyParam));
    copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
    copyParam.srcHost = h_volume;
    copyParam.srcPitch = 256 * sizeof(unsigned char);
    copyParam.srcHeight = 256;
    copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParam.dstArray = cu_array;
    copyParam.dstHeight=256;
    copyParam.WidthInBytes = 256 * sizeof(unsigned char);
    copyParam.Height = 256;
    copyParam.Depth = 225;
    cuMemcpy3D(&copyParam);

    // set texture parameters
    CUtexref cu_texref;
    checkCudaErrors(cuModuleGetTexRef(&cu_texref, cuModule, "tex"));
    checkCudaErrors(cuTexRefSetArray(cu_texref, cu_array, CU_TRSA_OVERRIDE_FORMAT));
    checkCudaErrors(cuTexRefSetAddressMode(cu_texref, 0, CU_TR_ADDRESS_MODE_BORDER));
    checkCudaErrors(cuTexRefSetAddressMode(cu_texref, 1, CU_TR_ADDRESS_MODE_BORDER));
    checkCudaErrors(cuTexRefSetFilterMode(cu_texref, CU_TR_FILTER_MODE_LINEAR));
    checkCudaErrors(cuTexRefSetFlags(cu_texref, CU_TRSF_NORMALIZED_COORDINATES));
    checkCudaErrors(cuTexRefSetFormat(cu_texref, CU_AD_FORMAT_UNSIGNED_INT8, 1));
	
	float4 *input_float_1D = (float4 *)malloc(sizeof(float4)*256);
    for(int i=0; i<=80; i++){    //alpha
		 input_float_1D[i].x = 0.0f;
		 input_float_1D[i].y = 0.0f;
		 input_float_1D[i].z = 0.0f;
		 input_float_1D[i].w = 0.0f;
	}
	for(int i=80+1; i<=120; i++){
		input_float_1D[i].x = (1.0f / (120.0f-80.0f)) * ( i - 80.0f);
		input_float_1D[i].y = (1.0f / (120.0f-80.0f)) * ( i - 80.0f);
		input_float_1D[i].z = (1.0f / (120.0f-80.0f)) * ( i - 80.0f);
		input_float_1D[i].w = (1.0f / (120.0f-80.0f)) * ( i - 80.0f);
		
	}
	for(int i=120+1; i<256; i++){
		input_float_1D[i].x =1.0f;
		input_float_1D[i].y =1.0f;
		input_float_1D[i].z =1.0f;
		input_float_1D[i].w =1.0f;
	}

	CUarray otf_array;
    CUDA_ARRAY_DESCRIPTOR ad;
    ad.Format = CU_AD_FORMAT_FLOAT;
    ad.Width = 256;
    ad.Height = 1;
    ad.NumChannels = 4;
	checkCudaErrors(cuArrayCreate(&otf_array, &ad));

	checkCudaErrors(cuMemcpyHtoA(otf_array,0,input_float_1D,256*sizeof(float4)));

	CUtexref otf_texref;
	
	checkCudaErrors(cuModuleGetTexRef(&otf_texref, cuModule, "texture_float_1D"));
	checkCudaErrors(cuTexRefSetFilterMode(otf_texref, CU_TR_FILTER_MODE_LINEAR ));
	checkCudaErrors(cuTexRefSetAddressMode(otf_texref, 0, CU_TR_ADDRESS_MODE_CLAMP ));
	checkCudaErrors(cuTexRefSetFlags(otf_texref, CU_TRSF_NORMALIZED_COORDINATES));
	checkCudaErrors(cuTexRefSetFormat(otf_texref, CU_AD_FORMAT_FLOAT, 4));
	checkCudaErrors(cuTexRefSetArray(otf_texref, otf_array, CU_TRSA_OVERRIDE_FORMAT));

   // checkCudaErrors(cuParamSetTexRef(transform, CU_PARAM_TR_DEFAULT, cu_texref));

    // There are two ways to launch CUDA kernels via the Driver API.
    // In this CUDA Sample, we illustrate both ways to pass parameters
    // and specify parameters.  By default we use the simpler method.
    //int block_size = 8;
    if (0)
    {
        // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel Launching (simpler method)
        void *args[8] = { &d_data, &d_invViewMatrix,  &imageWidth, &imageHeight, &density, &brightness, &transferOffset, &transferScale };

        checkCudaErrors(cuLaunchKernel(transform, (32), (32), 1,
                                       16     , 16     , 1,
                                       0,
                                       NULL, args, NULL));
        checkCudaErrors(cuCtxSynchronize());
        //sdkCreateTimer(&timer);
        //sdkStartTimer(&timer);

    }
    else
    {
        QueryPerformanceFrequency(&liFrequency);  // 시간 측정 초기화
		QueryPerformanceCounter(&start); 
		// This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel Launching (advanced method)
        int offset = 0;
        char argBuffer[256];

        // pass in launch parameters (not actually de-referencing CUdeviceptr).  CUdeviceptr is
        // storing the value of the parameters
        *((CUdeviceptr *)&argBuffer[offset]) = d_data;
        offset += sizeof(d_data);
		*((CUdeviceptr *)&argBuffer[offset]) = d_invViewMatrix;
        offset += sizeof(d_invViewMatrix);
        *((unsigned int *)&argBuffer[offset]) = imageWidth;
        offset += sizeof(imageWidth);
        *((unsigned int *)&argBuffer[offset]) = imageHeight;
        offset += sizeof(imageHeight);
        *((float *)&argBuffer[offset]) = density;
        offset += sizeof(density);
		*((float *)&argBuffer[offset]) = brightness;
        offset += sizeof(brightness);
		*((float *)&argBuffer[offset]) = transferOffset;
        offset += sizeof(transferOffset);
		*((float *)&argBuffer[offset]) = transferScale;
        offset += sizeof(transferScale);

        void *kernel_launch_config[5] =
        {
            CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
            CU_LAUNCH_PARAM_BUFFER_SIZE,    &offset,
            CU_LAUNCH_PARAM_END
        };

        // new CUDA 4.0 Driver API Kernel launch call (warmup)
        checkCudaErrors(cuLaunchKernel(transform, 32, 32, 1,
                                       16   , 16     , 1,
                                       0,
                                       NULL, NULL, (void **)&kernel_launch_config));

		
	   
        
	}
	
    checkCudaErrors(cuCtxSynchronize());
    HANDLE source_fh,dest_fh;
	int readn, writen;

	

    // allocate mem for the result on host side
    uint *h_odata = (uint *) malloc(256*256*4);
    // copy result from device to host
    cuMemcpyDtoH(h_odata, d_data, 256*256*4);
	QueryPerformanceCounter(&end);
	   printf("%f\n",(double)(end.QuadPart - start.QuadPart) / (double)liFrequency.QuadPart);
		
    checkCudaErrors(cuCtxSynchronize());
    
	const char* szStr = "dldndrb1";
	ScreenCapture(szStr,h_odata);
	
    // cleanup memory
    checkCudaErrors(cuMemFree(d_data));
    checkCudaErrors(cuArrayDestroy(cu_array));

    checkCudaErrors(cuCtxDestroy(cuContext));

    exit(bTestResults ? EXIT_SUCCESS : EXIT_FAILURE);
	
}

bool inline
findModulePath(const char *module_file, string &module_path, char **argv, string &ptx_source)
{
    char *actual_path = sdkFindFilePath(module_file, argv[0]);

    if (actual_path)
    {
        module_path = actual_path;
    }
    else
    {
        printf("> findModulePath file not found: <%s> \n", module_file);
        return false;
    }

    if (module_path.empty())
    {
        printf("> findModulePath file not found: <%s> \n", module_file);
        return false;
    }
    else
    {
        printf("> findModulePath <%s>\n", module_path.c_str());

        if (module_path.rfind(".ptx") != string::npos)
        {
            FILE *fp = fopen(module_path.c_str(), "rb");
            fseek(fp, 0, SEEK_END);
            int file_size = ftell(fp);
            char *buf = new char[file_size+1];
            fseek(fp, 0, SEEK_SET);
            fread(buf, sizeof(char), file_size, fp);
            fclose(fp);
            buf[file_size] = '\0';
            ptx_source = buf;
            delete[] buf;
        }

        return true;
    }
}


////////////////////////////////////////////////////////////////////////////////
//! This initializes CUDA, and loads the *.ptx CUDA module containing the
//! kernel function.  After the module is loaded, cuModuleGetFunction
//! retrieves the CUDA function pointer "cuFunction"
////////////////////////////////////////////////////////////////////////////////
static CUresult
initCUDA(int argc, char **argv, CUfunction *transform)
{
    CUfunction cuFunction = 0;
    CUresult status;
    int major = 0, minor = 0, devID = 0;
    char deviceName[100];
    string module_path, ptx_source;

    cuDevice = findCudaDevice(argc, argv, &devID);

    // get compute capabilities and the devicename
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
    checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    status = cuCtxCreate(&cuContext, 0, cuDevice);

    if (CUDA_SUCCESS != status)
    {
        printf("cuCtxCreate(0) returned %d\n-> %s\n", status, getCudaDrvErrorString(status));
        goto Error;
    }

    // first search for the module_path before we try to load the results
    if (!findModulePath(PTX_FILE, module_path, argv, ptx_source))
    {
        if (!findModulePath(CUBIN_FILE, module_path, argv, ptx_source))
        {
            printf("> findModulePath could not find <simpleTexture_kernel> ptx or cubin\n");
            status = CUDA_ERROR_NOT_FOUND;
            goto Error;
        }
    }
    else
    {
        printf("> initCUDA loading module: <%s>\n", module_path.c_str());
    }

    if (module_path.rfind("ptx") != string::npos)
    {
        // in this branch we use compilation with parameters
        const unsigned int jitNumOptions = 3;
        CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
        void **jitOptVals = new void *[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;

        // set up pointer to set the Maximum # of registers for a particular kernel
        jitOptions[2] = CU_JIT_MAX_REGISTERS;
        int jitRegCount = 32;
        jitOptVals[2] = (void *)(size_t)jitRegCount;

        status = cuModuleLoadDataEx(&cuModule, ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);

        printf("> PTX JIT log:\n%s\n", jitLogBuffer);
    }
    else
    {
        status = cuModuleLoad(&cuModule, module_path.c_str());
    }

    if (CUDA_SUCCESS != status)
    {
        goto Error;
    }

    status = cuModuleGetFunction(&cuFunction, cuModule, "transformKernel");

    if (CUDA_SUCCESS != status)
    {
        goto Error;
    }

    *transform = cuFunction;
    return CUDA_SUCCESS;
Error:
    cuCtxDestroy(cuContext);
    return status;
}

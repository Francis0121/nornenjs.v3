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
    Volume rendering sample

    This sample loads a 3D volume from disk and displays it using
    ray marching and 3D textures.

    Note - this is intended to be an example of using 3D textures
    in CUDA, not an optimized volume renderer.

    Changes
    sgg 22/3/2010
    - updated to use texture for display instead of glDrawPixels.
    - changed to render from front-to-back rather than back-to-front.
*/

// OpenGL Graphics includes
#include <GL/glew.h>

#include <windows.h>
#if defined (__APPLE__) || defined(MACOSX)
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>

#endif

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f


LARGE_INTEGER rendertime_start;
LARGE_INTEGER rendertime_stop;
LARGE_INTEGER proc_freq;

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "volume.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_volume.ppm",
    NULL
};

const char *sSDKsample = "CUDA 3D Volume Render";

const char *volumeFilename = "Bighead.den";
cudaExtent volumeSize = make_cudaExtent(256, 256, 225);
cudaExtent volumeSize_block = make_cudaExtent(256/4, 256/4, 225/4);
typedef unsigned char VolumeType;

//char *volumeFilename = "mrt16_angio.raw";
//cudaExtent volumeSize = make_cudaExtent(416, 512, 112);
//typedef unsigned short VolumeType;

uint width = 768, height = 768;
dim3 blockSize(32, 32);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -3.0f);
float invViewMatrix[12];

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;

int *pArgc;
char **pArgv;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(void *h_volume, cudaExtent volumeSize);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
extern "C" void* make_blockVolume(void* image, cudaExtent volumeSize_block, cudaExtent volumeSize);
extern "C" void initBlockTexture(const void *h_volume_block, int x, int y, int z);

void initPixelBuffer();

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Volume Render: %3.1f fps", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&timer);
    }
}
void ScreenCapture( const char *strFilePath ,uchar *d_output)
{
    //비트맵 파일 처리를 위한 헤더 구조체
    BITMAPFILEHEADER    BMFH;
    BITMAPINFOHEADER    BMIH;
 
    int nWidth = 0;
    int nHeight = 0;
    unsigned long dwQuadrupleWidth = 0;     //LJH 추가, 가로 사이즈가 4의 배수가 아니라면 4의 배수로 만들어서 저장
 
    GLbyte *pPixelData = NULL;              //front buffer의 픽셀 값들을 얻어 오기 위한 버퍼의 포인터
 
    nWidth  = 512;     //(나의 경우)리눅스에서의 경우 해상도 고정이므로 그 값을 입력
    nHeight = 512;
 
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
    BMIH.biSizeImage        = 512 * 3 * 512;					//영상의 크기
    BMIH.biXPelsPerMeter    = 0;                                //가로 해상도
    BMIH.biYPelsPerMeter    = 0;                                //세로 해상도
    BMIH.biClrUsed          = 0;                                //실제 사용 색상수
    BMIH.biClrImportant     = 0;                                //중요한 색상 인덱스
 
    pPixelData = new GLbyte[ dwQuadrupleWidth * 3 * nHeight ];  //LJH 수정
 
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
        fwrite( &BMFH, sizeof( uchar ), sizeof(BITMAPFILEHEADER), outFile );         //파일 헤더 쓰기
        fwrite( &BMIH, sizeof( uchar ), sizeof(BITMAPINFOHEADER), outFile );         //인포 헤더 쓰기
		for(int i=0; i<512; i++){
			for(int j=0; j<512; j++){
				fwrite((d_output+(i*512 +j+0)), sizeof( uchar ), 1, outFile );
				fwrite((d_output+(i*512 +j+0)), sizeof( uchar ), 1, outFile );
				fwrite((d_output+(i*512 +j+0)), sizeof( uchar ), 1, outFile );
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

// render image using CUDA
void render()
{
    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);
	
    // map PBO to get CUDA device pointer
	uint  *d_output;
	uint  *c_output; 
	c_output=(uint *)malloc(width*height*4);
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, width*height*4));
	printf("grid %d %d %d\n",gridSize.x,gridSize.y,gridSize.z);
	printf("block %d %d %d\n",blockSize.x,blockSize.y,blockSize.z);
    // call CUDA kernel, writing results to PBO
    render_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale);
	
	cudaMemcpy(c_output, d_output, width*height*4, cudaMemcpyDeviceToHost);

	//const char* szStr = "dldndrb1";
	/*for(int i=0; i<255*255; i++){
		printf("%d\n",c_output[i]);

	}*/
	//ScreenCapture(szStr,c_output);
    getLastCudaError("kernel failed");
    free(c_output);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}
float i=0.0f;
void display()
{
    sdkStartTimer(&timer);
	///i+=3.0f;
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
	
    glRotatef(viewRotation.x+270, 1.0, 0.0, 0.0);
    glRotatef(viewRotation.y+180, 0.0, -1.0, 0.0);
    glTranslatef(-viewTranslation.x,-viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();
	
	//	int count=0;
	//for(int i=0; i<16; i++){
	//	
	//	if(count%4 ==0){
	//		printf("\n");
	//	}
	//	count++;
	//	printf("%lf ",(float)modelView[i]);
	//}
    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

	static int drawcount = 0;
	static double fps_sum = 0.0;

	double frequency;
	double elapsed, renderfps = 0.0;

	if(!QueryPerformanceFrequency(&proc_freq) ){
		printf("QueryPerformanceFrequency Error!\n");
	}

	frequency = 1.0 / proc_freq.QuadPart;

	QueryPerformanceCounter(&rendertime_stop);

	elapsed = (rendertime_stop.QuadPart - rendertime_start.QuadPart) * frequency;

	
	if(fps_sum < 1.0f){
		fps_sum += elapsed;
		drawcount++;
	}else{
		printf("fps : %d \n", drawcount);

		fps_sum = 0.0f;
		drawcount = 0;
	}



	QueryPerformanceCounter(&rendertime_start);
    render();
	
    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
	
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
    // draw using glDrawPixels (slower)
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // draw using texture

    // copy from pbo to texture
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	
    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();
	
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

#endif
	
    glutSwapBuffers();
    glutReportErrors();
	
    sdkStopTimer(&timer);

    computeFPS();
}

void idle()
{
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
            exit(EXIT_SUCCESS);
            break;

        case 'f':
            linearFiltering = !linearFiltering;
            setTextureFilterMode(linearFiltering);
            break;

        case '+':
            density += 0.01f;
            break;

        case '-':
            density -= 0.01f;
            break;

        case ']':
            brightness += 0.1f;
            break;

        case '[':
            brightness -= 0.1f;
            break;

        case ';':
            transferOffset += 0.01f;
            break;

        case '\'':
            transferOffset -= 0.01f;
            break;

        case '.':
            transferScale += 0.01f;
            break;

        case ',':
            transferScale -= 0.01f;
            break;

        default:
            break;
    }

    printf("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        buttonState  |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // right = zoom
		//printf("%d %d\n",x,y);
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2)
    {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1)
    {
        // left = rotate
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
		//printf("%f %f\n",viewRotation.x,viewRotation.y);
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

int iDivUp(int a, int b)
{
	printf("%d,%d\n",a,b);//debug
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h)
{
    width = w;
    height = h;
    initPixelBuffer();

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
	
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    freeCudaBuffers();

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &tex);
    }
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA volume rendering");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions missing.");
        exit(EXIT_SUCCESS);
    }
}

void initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));
	
    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// Load raw data from disk
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

// General initialization call for CUDA Device
int chooseCudaDevice(int argc, const char **argv, bool bUseOpenGL)
{
    int result = 0;

    if (bUseOpenGL)
    {
        result = findCudaGLDevice(argc, argv);
    }
    else
    {
        result = findCudaDevice(argc, argv);
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    char *ref_file = NULL;

    //start logs
    printf("%s Starting...\n\n", sSDKsample);  //이건 그냥 출력부.

 
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    initGL(&argc, argv);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    chooseCudaDevice(argc, (const char **)argv, true);
    

    // load volume data
    char *path = sdkFindFilePath(volumeFilename, argv[0]);

    if (path == 0)
    {
        printf("Error finding file '%s'\n", volumeFilename);
        exit(EXIT_FAILURE);
    }

    size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*sizeof(VolumeType);
    
	void *h_volume = loadRawFile(path, size);
    void *block_volume = make_blockVolume(h_volume, volumeSize_block, volumeSize); //블락 볼륨 만들기
    
	initCuda(h_volume, volumeSize);
	initBlockTexture(block_volume,volumeSize_block.width,volumeSize_block.height,volumeSize_block.depth);
    
	free(h_volume);
	free(block_volume);
    sdkCreateTimer(&timer);

    printf("Press '+' and '-' to change density (0.01 increments)\n"
           "      ']' and '[' to change brightness\n"
           "      ';' and ''' to modify transfer function offset\n"
           "      '.' and ',' to modify transfer function scale\n\n");

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));


	// This is the normal rendering path for VolumeRender
    glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);

	initPixelBuffer();
	
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        glutMainLoop();
    

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}

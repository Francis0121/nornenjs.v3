#include <jni.h>

#include "app-jnigl.h"

#include <GLES/gl.h>
#include <GLES/glext.h>
#include <stdlib.h>
void drawCube();

float TOUCH_SCALE_FACTOR = 180.0f / 320;
float TRACKBALL_SCALE_FACTOR = 36.0f;
float mPreviousX=0;
float mPreviousY=0;
float mAngleX=0;
float mAngleY=0;
GLuint  g_textureName;

void nativeOnTouchEvent(int e, float x, float y)
{
	switch (e) {
	case 0x2:
		float dx = x - mPreviousX;
		float dy = y - mPreviousY;
		mAngleX += dx * TOUCH_SCALE_FACTOR;
		mAngleY += dy * TOUCH_SCALE_FACTOR;
	}
	mPreviousX = x;
	mPreviousY = y;
}

void nativeOnTrackballEvent(int e, float x, float y)
{
	mAngleX += x * TRACKBALL_SCALE_FACTOR;
	mAngleY += y * TRACKBALL_SCALE_FACTOR;
}

void nativeDrawIteration(float mx, float my)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);


    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(0.0f, 0, -2.0f);

    glTranslatef(0.0f, 0, 0.0f);

    drawCube();

}

void nativeOnCreate()
{


}

void nativeOnDestroy()
{


}

void nativeOnPause()
{

}


void nativeOnResume()
{

}


void nativeInitGL(int w, int h)
{
	/*
	* By default, OpenGL enables features that improve quality
	* but reduce performance. One might want to tweak that
	* especially on software renderer.
	*/
	glDisable(GL_DITHER);

	/*
	* Some one-time OpenGL initialization can be made here
	* probably based on features of this particular context
	*/
	glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_FASTEST);
	
	glClearColor(0,0,0,0);
	glEnable(GL_CULL_FACE);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
}

void initTextureData(int *data, int width, int height)
{
	glGenTextures(1, &g_textureName);

	glBindTexture(GL_TEXTURE_2D, g_textureName);

//    for (int i = 0; i <height*width; i++){
//             data[i] = 0xff000000 | (((data[i] >> 24) & 0xff) << 24)
//                                  | (((data[i] >>  0) & 0xff) << 16)
//                                  | (((data[i] >>  8) & 0xff) <<  8)
//                                  | (((data[i] >> 16) & 0xff) <<  0);
//    }

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,  GL_UNSIGNED_BYTE, (void*)data);
	//free(buf);
}

void setTextureData(int *data, int width, int height)
{
//     for (int i = 0; i <height*width; i++){
//
//            data[i] = 0xff000000 | (((data[i] >> 24) & 0xff) << 24)
//                                 | (((data[i] >>  0) & 0xff) << 16)
//                                 | (((data[i] >>  8) & 0xff) <<  8)
//                                 | (((data[i] >> 16) & 0xff) <<  0);
//    }
	glBindTexture(GL_TEXTURE_2D, g_textureName);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,  GL_UNSIGNED_BYTE, (void*)data);

}

void nativeOnResize(int w, int h)
{
    glViewport(0, 0, w, h);

    /*
     * Set our projection matrix. This doesn't have to be done
     * each time we draw, but usually a new projection needs to
     * be set when the viewport is resized.
     */

    float ratio = (float) w / (float)h;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustumf(-ratio, ratio, -1, 1, 1, 10);
}

void drawCube()
{

	int one = 1;
    static short vertices[] = {
                -2.0f	, 2.0f	, 0.0f, // 0, Left Top
                 2.0f	, 2.0f	, 0.0f,	// 1, Right Top
                 2.0f	, -2.0f	, 0.0f,	// 2, Right Bottom
                -2.0f	, -2.0f	, 0.0f	// 3, Left Bottom
    };

    static float colors[] = {
                  1,    1,    1,  1,
                  1,    1,    1,  1,
                  1,    1,    1,  1,
                  1,    1,    1,  1
          };

    static unsigned short indices[] = {
                  0, 1, 2,
                  0, 2, 3
    };
    static float texture[] = {
    	    		//Mapping coordinates for the vertices
    	    		0.0f, 0.0f,
    	    		1.0f, 0.0f,
    	    		1.0f, 1.0f,
    	    		0.0f, 1.0f,
    	    };

    glFrontFace(GL_CW);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_textureName);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glVertexPointer(3, GL_SHORT, 0, vertices);
    glColorPointer(4, GL_FLOAT, 0, colors);
    glTexCoordPointer(2, GL_FLOAT, 0, texture);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indices);

    glDisableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

}



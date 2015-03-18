#include <jni.h>

#include "app-jnigl.h"

#include <GLES/gl.h>
#include <GLES/glext.h>

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
	//float x = e.getX();
	//float y = e.getY();
	switch (e) {
	case 0x2:
		float dx = x - mPreviousX;
		float dy = y - mPreviousY;
		mAngleX += dx * TOUCH_SCALE_FACTOR;
		mAngleY += dy * TOUCH_SCALE_FACTOR;
		//requestRender();
	}
	mPreviousX = x;
	mPreviousY = y;
}

void nativeOnTrackballEvent(int e, float x, float y)
{
	mAngleX += x * TRACKBALL_SCALE_FACTOR;
	mAngleY += y * TRACKBALL_SCALE_FACTOR;
	//        requestRender();
}

void nativeDrawIteration(float mx, float my)
{
    /*
     * Usually, the first thing one might want to do is to clear
     * the screen. The most efficient way of doing this is to use
     * glClear().
     */

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /*
     * Now we're ready to draw some 3D objects
     */


    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

// approach 1

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(0.0f, 0, -2.0f);

   // glRotatef(mAngleY, 1, 0, 0);
    //glRotatef(mAngleX, 0, 1, 0);


    glTranslatef(0.0f, 0, 0.0f);

    //glScalef(0.5f, 0.5f, 0.5f);

    drawCube();

/*    glLoadIdentity();

    glTranslatef(0.0f, 0, -3.0f);

    glRotatef(mAngleY, 1, 0, 0);
    glRotatef(mAngleX, 0, 1, 0);

    glTranslatef(1.0f, 0, 0.0f);

    glScalef(0.5f, 0.5f, 0.5f);

    drawCube();*/

//         // approach 2
//
//            glMatrixMode(GL_MODELVIEW);
//            glLoadIdentity();
//
//            glTranslatef(0.0f, 0, -3.0f);
//
//            glRotatef(my, 1, 0, 0);
//            glRotatef(mx, 0, 1, 0);
//
//            //draw cube1
//            glPushMatrix();
//	            glTranslatef(-1.0f, 0, 0.0f);
//	            glScalef(0.5f, 0.5f, 0.5f);
//	            mCube.draw(gl);
//	          glPopMatrix();
//
//	      	  //draw cube2
//            glPushMatrix();
//	            glTranslatef(1.0f, 0, 0.0f);
//	            glScalef(0.5f, 0.5f, 0.5f);
//	            mCube.draw(gl);
//	          glPopMatrix();




    // previous code
//            //
//            glLoadIdentity();
//            glTranslatef(1.0f, 0, -3.0f);
//            glScalef(0.5f, 0.5f, 0.5f);
//
//            glRotatef(mx, 0, 1, 0);
//            glRotatef(my, 1, 0, 0);
//            mCube.draw(gl);
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

//void nativeOnAccelerometer(float x,float y,float z);
//void nativeSendEvent(int action, float x, float y);

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
	glHint(GL_PERSPECTIVE_CORRECTION_HINT,
		 GL_FASTEST);


	glClearColor(0,0,0,0);
	glEnable(GL_CULL_FACE);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
}


void setTextureData(int *data, int width, int height)
{
	glGenTextures(1, &g_textureName);

	glBindTexture(GL_TEXTURE_2D, g_textureName);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);
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
    //int one = 0x10000;




	int one = 1;
    static short vertices[] = {
                -1.0f	, 1.0f	, 0.0f, // 0, Left Top
                 1.0f	, 1.0f	, 0.0f,	// 1, Right Top
                 1.0f	, -1.0f	, 0.0f,	// 2, Right Bottom
                -1.0f	, -1.0f	, 0.0f	// 3, Left Bottom
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
    // Buffers to be passed to gl*Pointer() functions
    // must be direct, i.e., they must be placed on the
    // native heap where the garbage collector cannot
    // move them.
    //
    // Buffers with multi-byte datatypes (e.g., short, int, float)
    // must have their byte order set to native order

//    ByteBuffer vbb = ByteBuffer.allocateDirect(vertices.length*4);
//    vbb.order(ByteOrder.nativeOrder());
//    mVertexBuffer = vbb.asIntBuffer();
//    mVertexBuffer.put(vertices);
//    mVertexBuffer.position(0);
//
//    ByteBuffer cbb = ByteBuffer.allocateDirect(colors.length*4);
//    cbb.order(ByteOrder.nativeOrder());
//    mColorBuffer = cbb.asIntBuffer();
//    mColorBuffer.put(colors);
//    mColorBuffer.position(0);
//
//    mIndexBuffer = ByteBuffer.allocateDirect(indices.length);
//    mIndexBuffer.put(indices);
//    mIndexBuffer.position(0);

    /* DataType */
    /*
    #define GL_BYTE                           0x1400
    #define GL_UNSIGNED_BYTE                  0x1401
    #define GL_SHORT                          0x1402
    #define GL_UNSIGNED_SHORT                 0x1403
    #define GL_FLOAT                          0x1406
    #define GL_FIXED                          0x140C
    */

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



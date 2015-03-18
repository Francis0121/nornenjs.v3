package com.semo.jnigl;


import android.os.Bundle;
import android.app.Activity;
import android.util.Log;
import android.view.Menu;
import android.view.MotionEvent;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.content.Context;
import android.opengl.GLSurfaceView;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

import javax.microedition.khronos.opengles.GL10;

public class JniGLActivity extends Activity {

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //setContentView(R.layout.activity_jni_gl);
        
        // Create our Preview view and set it as the content of our
        // Activity
        mGLSurfaceView = new TouchSurfaceView(this);
        setContentView(mGLSurfaceView);
        mGLSurfaceView.requestFocus();
        mGLSurfaceView.setFocusableInTouchMode(true);             
    }

//    @Override
//    public boolean onCreateOptionsMenu(Menu menu) {
//        getMenuInflater().inflate(R.menu.activity_jni_gl, menu);
//        return true;
//    }
    
    @Override
    protected void onResume() {
        // Ideally a game should implement onResume() and onPause()
        // to take appropriate action when the activity looses focus
        super.onResume();
        mGLSurfaceView.onResume();
    }

    @Override
    protected void onPause() {
        // Ideally a game should implement onResume() and onPause()
        // to take appropriate action when the activity looses focus
        super.onPause();
        mGLSurfaceView.onPause();
    }

    private GLSurfaceView mGLSurfaceView;    
    
    /** load irrlicht.so */
    static {
    	Log.i("jnigl", "try to load libjnigl.so");
        System.loadLibrary("jnigl");
    }

    public native void nativeOnCreate();
    public native void nativeOnPause();
    public native void nativeOnResume();
    public native void nativeOnDestroy();
    public native void nativeOnTrackballEvent(int e, float x, float y);
    public native void nativeOnTouchEvent(int e, float x, float y);
    
    public native void nativeInitGL();
    public native void nativeResize(int w, int h);

//    public native void nativeGetStatus(IrrlichtStatus status);
//    public native void nativeSendEvent(IrrlichtEvent event);
//    public native void nativeEnvJ2C(String sdcardPath);
//	public native void nativeOnAccelerometer(float x, float y, float z);

    public static native void nativeDrawIteration(float mx, float my);        
}

class TouchSurfaceView extends GLSurfaceView {
    private JniGLActivity mActivity;
    
    public TouchSurfaceView(Context context) {
        super(context);
        
        //setEGLConfigChooser(8, 8, 8, 8, 16, 0);
		//getHolder().setFormat(PixelFormat.RGBA_8888);
        
        mActivity = (JniGLActivity) context;
        
        mRenderer = new CubeRenderer(mActivity);
        setRenderer(mRenderer);
        //setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
    }

    @Override public boolean onTrackballEvent(MotionEvent e) {
    	mActivity.nativeOnTrackballEvent(e.getAction(), e.getX(), e.getY());
    	requestRender();
        return true;
    }

    @Override public boolean onTouchEvent(MotionEvent e) {	
		mActivity.nativeOnTouchEvent(e.getAction(), e.getX(), e.getY());
		requestRender();
    	
        return true;
    }
    
    /**
     * Render a cube.
     */
    private class CubeRenderer implements GLSurfaceView.Renderer {
    	
        private JniGLActivity mActivity;
        
        public CubeRenderer(JniGLActivity activity) {
        	mActivity = activity;
        }

        public void onDrawFrame(GL10 gl) {
        	mActivity.nativeDrawIteration(0, 0);
        }

        public void onSurfaceChanged(GL10 gl, int width, int height) {
        	mActivity.nativeResize(width, height);
        }

        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        	mActivity.nativeInitGL();
        }

    }

    private CubeRenderer mRenderer;

}

    


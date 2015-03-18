package com.semo.jnigl;


import android.os.Bundle;
import android.app.Activity;
import android.util.Log;
import android.view.Menu;
import android.view.MotionEvent;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.content.Context;
import android.opengl.GLSurfaceView;

import java.net.URISyntaxException;
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

        Log.d("bmp", "onCreate");
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

    public static native void nativeDrawIteration(float mx, float my);
    public static native void nativeSetTextureData(int[] pixels, int width, int height);
}

class TouchSurfaceView extends GLSurfaceView {

    private CubeRenderer mRenderer;
    private JniGLActivity mActivity;
    private Context mContext;

    public TouchSurfaceView(Context context) {
        super(context);
        mContext = context;
        mActivity = (JniGLActivity) context;
        mRenderer = new CubeRenderer(mActivity);
        setRenderer(mRenderer);
    }

    @Override 
    public boolean onTrackballEvent(MotionEvent e) {
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
            Log.d("bmp", "onDrawFrame");
            mActivity.nativeDrawIteration(0, 0);
        }

        public void onSurfaceChanged(GL10 gl, int width, int height) {
            mActivity.nativeResize(width, height);
        }

        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            Log.d("bmp", "onSurfaceCreated");

            Bitmap bmp = BitmapFactory.decodeResource(mContext.getResources(), R.drawable.ic_launcher);
            int[] pixels = new int[bmp.getWidth()*bmp.getHeight()];
            bmp.getPixels(pixels, 0, bmp.getWidth(), 0, 0, bmp.getWidth(), bmp.getHeight());
            mActivity.nativeSetTextureData(pixels, bmp.getWidth(), bmp.getHeight());
            mActivity.nativeInitGL();
        }

    }

}

    

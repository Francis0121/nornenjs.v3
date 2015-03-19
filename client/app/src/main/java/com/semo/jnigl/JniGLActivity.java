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
import android.widget.ImageView;
import com.github.nkzawa.emitter.Emitter;
import com.github.nkzawa.socketio.client.IO;
import com.github.nkzawa.socketio.client.Socket;

import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

import javax.microedition.khronos.opengles.GL10;

public class JniGLActivity extends Activity {

    private Socket socket;
    
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_jni_gl);
        mGLSurfaceView = new TouchSurfaceView(this);
        setContentView(mGLSurfaceView);
        mGLSurfaceView.requestFocus();
        mGLSurfaceView.setFocusableInTouchMode(true);

        Log.d("bmp", "onCreate");
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.d("bmp", "onResume");
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.d("bmp", "onPause");
    }

    private GLSurfaceView mGLSurfaceView;

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

    private class CubeRenderer implements GLSurfaceView.Renderer {

        private JniGLActivity mActivity;
        private byte[] byteArray;
        private Socket socket;

        public CubeRenderer(JniGLActivity activity) {
            mActivity = activity;
            Log.d("socket", "connectin");
            try {

                socket = IO.socket("http://112.108.40.166:5000");
            } catch (URISyntaxException e) {
                Log.d("socket", e.getMessage());
                e.printStackTrace();
                Log.d("socket", "connectin2");
            }
            Log.d("socket", "connectin2");

            socket.on(Socket.EVENT_CONNECT, new Emitter.Listener() {
                @Override
                public void call(Object... args) {
                    Log.d("socket", "connect");
                    socket.emit("stream"); // 112.108.40.166
                }
            });


            socket.on("jpeg", new Emitter.Listener() { //112.108.40.166
                @Override
                public void call(Object... args) {
                    byteArray = (byte[]) args[0];
                    Log.d("socket", "jpeg");
                }
            });

            socket.connect();
        }

        public void onDrawFrame(GL10 gl) {
            
            if(byteArray!=null) {
                Log.d("bmp", "onDrawFrame");
                Bitmap imgPanda = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);

                int[] pixels = new int[imgPanda.getWidth()*imgPanda.getHeight()];
                imgPanda.getPixels(pixels, 0, imgPanda.getWidth(), 0, 0, imgPanda.getWidth(), imgPanda.getHeight());
                mActivity.nativeSetTextureData(pixels, imgPanda.getWidth(), imgPanda.getHeight());
            }
            mActivity.nativeDrawIteration(0, 0);
        }

        public void onSurfaceChanged(GL10 gl, int width, int height) {
            mActivity.nativeResize(width, height);
        }

        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            Log.d("bmp", "onSurfaceCreated");
            mActivity.nativeInitGL();
        }

    }

}

    

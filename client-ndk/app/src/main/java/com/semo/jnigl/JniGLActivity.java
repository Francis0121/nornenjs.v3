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
import android.view.SurfaceHolder;
import android.widget.ImageView;
import com.github.nkzawa.emitter.Emitter;
import com.github.nkzawa.socketio.client.IO;
import com.github.nkzawa.socketio.client.Socket;
import org.json.JSONException;
import org.json.JSONObject;

import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

import javax.microedition.khronos.opengles.GL10;

public class JniGLActivity extends Activity {

    public boolean isOn = false;
    public float beforeX = 0.0f, beforeY = 0.0f;
    public float rotationX = 0.0f, rotationY = 0.0f;

    private MyEventListener myEventListener;

    public void setMyEventListener(MyEventListener myEventListener) {
        this.myEventListener = myEventListener;
    }

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

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d("bmp", "onPause");
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        Log.v("opengl", "onTouchEvent");

        switch(event.getAction()) {

            case MotionEvent.ACTION_DOWN :
                Log.d("opengl", "onTouchEvent : ACTION_DOWN");
                isOn = true;
                beforeX = event.getX();
                beforeY = event.getY();

                break;
            case MotionEvent.ACTION_MOVE :
                Log.d("opengl", "onTouchEvent : ACTION_MOVE");
                if(isOn) {
                    //calc
                    rotationX += (event.getX() - beforeX) / 10.0;
                    rotationY += (event.getY() - beforeY) / 10.0;

                    beforeX = event.getX();
                    beforeY = event.getY();
                    
                    myEventListener.onMyevent(rotationX, rotationY);
                }
                break;
            case MotionEvent.ACTION_UP :
                Log.d("opengl", "onTouchEvent : ACTION_UP");
                isOn = false;
                break;
        }
        return super.onTouchEvent(event);
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
    public static native void nativeInitTextureData(int[] pixels, int width, int height);
    public static native void nativeSetTextureData(int[] pixels, int width, int height);
}

class TouchSurfaceView extends GLSurfaceView {

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        super.surfaceDestroyed(holder);
        Log.d("bmp", "surfaceDestoryed");
   }

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

    /**
     * Render a cube.
     */
    private class CubeRenderer implements GLSurfaceView.Renderer, MyEventListener {

        private JniGLActivity mActivity;
        private byte[] byteArray;
        //private int[] intArray;
        private Socket socket;

        public CubeRenderer(JniGLActivity activity) {
            mActivity = activity;
            Log.d("socket", "connectin");
            // ~ socket connection
            try {
                socket = IO.socket("http://112.108.40.166:5000");
            } catch (URISyntaxException e) {
                e.printStackTrace();
            }

            socket.emit("connectMessage");

            socket.on("connectMessage", new Emitter.Listener() {

                @Override
                public void call(Object... args) {
                    Log.d("socket", "on connectMessage");
                    JSONObject message = (JSONObject) args[0];

                    try {
                        if( ! ((Boolean) message.get("success")) ){
                            return;
                        }
                        socket.emit("init");
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }

                }

            });

            socket.on("stream", new Emitter.Listener() { //112.108.40.166
                @Override
                public void call(Object... args) {
                    byteArray = (byte[]) args[0];
                }
            });

            socket.connect();
            
            mActivity.setMyEventListener(this);
        }
        
        Bitmap imgPanda;
        int[] pixels = new int[512*512];
        
        public void onDrawFrame(GL10 gl) {
            
            if(byteArray!=null) {
                imgPanda = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
                imgPanda.getPixels(pixels, 0, imgPanda.getWidth(), 0, 0, imgPanda.getWidth(), imgPanda.getHeight());
                mActivity.nativeSetTextureData(pixels, 512, 512);
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


        public int[] convert(byte buf[]) {
            int intArr[] = new int[buf.length / 4];
            int offset = 0;
            for(int i = 0; i < intArr.length; i++) {
                intArr[i] = (buf[3 + offset] & 0xFF) | ((buf[2 + offset] & 0xFF) << 8) |
                        ((buf[1 + offset] & 0xFF) << 16) | ((buf[0 + offset] & 0xFF) << 24);
                offset += 4;
            }
            return intArr;
        }


        @Override
        public void onMyevent(float rotationX, float rotationY) {
            //받아서 서버에 보내기
            Log.d("opengl", "!!!!!!!!!!!");
            
            JSONObject jsonObject = new JSONObject();
            try {
                jsonObject.put("rotationX", rotationX);
                jsonObject.put("rotationY", rotationY);
            } catch (JSONException e) {
                e.printStackTrace();
                Log.e("error", "Make json object");
            }
            socket.emit("event", jsonObject);
        }
    }
    
    
}

    

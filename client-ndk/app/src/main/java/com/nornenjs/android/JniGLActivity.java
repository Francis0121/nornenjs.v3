package com.nornenjs.android;


import android.os.Bundle;
import android.app.Activity;
import android.util.FloatMath;
import android.util.Log;
import android.view.MotionEvent;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.content.Context;
import android.opengl.GLSurfaceView;
import android.view.SurfaceHolder;
import com.github.nkzawa.emitter.Emitter;
import com.github.nkzawa.socketio.client.IO;
import com.github.nkzawa.socketio.client.Socket;
import org.json.JSONException;
import org.json.JSONObject;

import java.net.URISyntaxException;

public class JniGLActivity extends Activity {

    static final int NONE = 0;
    static final int DRAG = 1;
    static final int ZOOM = 2;
    static final int MULTI_TOUCH = 3;
    int mode = NONE;

    // 핀치시 두좌표간의 거리 저장
    float oldDist = 1f;
    float newDist = 1f;

    public float beforeX = 0.0f, beforeY = 0.0f;
    public float rotationX = 0.0f, rotationY = 0.0f;
    public float div=3.0f;

    public float oldVectorX1 =0.0f, oldVectorY1 =0.0f;
    public float oldVectorX2 =0.0f, oldVectorY2 =0.0f;

    public float newVectorX1 =0.0f, newVectorY1 =0.0f;
    public float newVectorX2 =0.0f, newVectorY2 =0.0f;

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
       // Log.v("opengl", "onTouchEvent");
        int act = event.getAction();
        switch(act & MotionEvent.ACTION_MASK) {

            case MotionEvent.ACTION_DOWN :
                if(event.getPointerCount()==1){
                    beforeX = event.getX();  //posX1
                    beforeY = event.getY();  //posY1


                    mode = DRAG;
                }
                break;

            case MotionEvent.ACTION_MOVE :

                if(mode == DRAG && event.getPointerCount() == 1 ) {

                    rotationX += (event.getX() - beforeX) / 10.0;
                    rotationY += (event.getY() - beforeY) / 10.0;

                    beforeX = event.getX();
                    beforeY = event.getY();

                    myEventListener.onMyevent(rotationX, rotationY);
                }
                else if(event.getPointerCount() == 2) {

                    newVectorX1 = event.getX(0);
                    newVectorX2 = event.getX(1);
                    newVectorY1 = event.getY(0);
                    newVectorY2 = event.getY(1);

                   // Log.d("opengl newx1, newy1", "" + newVectorX1 + "  "+ newVectorY1 + "  " + (VecotrDirection(oldVectorX1,newVectorX1)));
                   // Log.d("opengl newx2, newy2", "" + newVectorX2 + "  "+ newVectorY2 + "  " + (VecotrDirection(oldVectorX2,newVectorX2)));
                  //  Log.d("opengl oldx1, oldy1", "" + oldVectorX1 + "  "+ oldVectorY1 + "  " + (VecotrDirection(oldVectorY1,newVectorY2)));
                  //  Log.d("opengl oldx2, oldy2", "" + oldVectorX2 + "  "+ oldVectorY2 + "  " + (VecotrDirection(oldVectorY2,newVectorY2)));
                    if((VecotrDirection(oldVectorX1,newVectorX1) == (VecotrDirection(oldVectorX2,newVectorX2)) &&
                            (VecotrDirection(oldVectorY1,newVectorY1) == (VecotrDirection(oldVectorY2,newVectorY2))))){

                         newDist = spacing(event);
                        if(Math.abs(newDist - oldDist) < 10 && newDist < 150) { // 이동

                            Log.d("opengl two finger translateion","dddddddddd");

                        }




                    }
                    else{
                        newDist = spacing(event);

                        if (newDist - oldDist > 50) { // zoom in

                            oldDist = newDist;
                            div += (((newDist / oldDist) / 50) * 10);

                            if (div >= 10.0f) {
                                div = 10.0f;
                            }

                            Log.d("opengl zoom in", "" + div);

                        } else if (oldDist - newDist > 50) { // zoom out

                            oldDist = newDist;
                            div -= (((newDist / oldDist) / 50) * 10);

                            if (div < 0.0f) {
                                div = 0.0f;
                            }

                            Log.d("opengl zoom out", "" + div);
                        }

                    }
                }
                break;

            case MotionEvent.ACTION_UP :
                mode = NONE;
                break;
            case MotionEvent.ACTION_POINTER_UP:
                mode = NONE;
                break;
            case MotionEvent.ACTION_POINTER_DOWN:	// 하나 클릭한 상태에서 추가 클릭.
                // newDist = spacing(event);
                oldDist = spacing(event);

                oldVectorX1 = event.getX(0);
                oldVectorX2 = event.getX(1);

                oldVectorY1 = event.getY(0);
                oldVectorY2 = event.getY(1);
                break;

            case MotionEvent.ACTION_CANCEL:

            default:
                    break;
        }
        return super.onTouchEvent(event);
    }
    private float spacing(MotionEvent event) {
        float x = event.getX(0) - event.getX(1);
        float y = event.getY(0) - event.getY(1);

        return FloatMath.sqrt(x * x + y * y);
    }
    private boolean VecotrDirection(float vector1, float vector2) { //음수면 0 양수면 1

        if(vector2 - vector1<0) {

            return false;
        }else {

            return true;
        }

    }

    private GLSurfaceView mGLSurfaceView;

    /** load irrlicht.so */
    static {
        Log.i("opengles", "try to load opengles.so");
        System.loadLibrary("opengles");
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
        private Socket socket;

        public CubeRenderer(JniGLActivity activity) {
            mActivity = activity;
            Log.d("socket", "connectin");
            // ~ socket connection
            try {
                socket = IO.socket("http://112.108.40.19:5000");
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
                        if (!((Boolean) message.get("success"))) {
                            return;
                        }
                        socket.emit("init");
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }

                }

            });

            socket.on("loadCudaMemory", new Emitter.Listener() { //112.108.40.166
                @Override
                public void call(Object... args) {
                    socket.emit("androidPng");
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
            //Log.d("opengl", "!!!!!!!!!!!");
            
            JSONObject jsonObject = new JSONObject();
            try {
                jsonObject.put("rotationX", rotationX);
                jsonObject.put("rotationY", rotationY);
            } catch (JSONException e) {
                e.printStackTrace();
                Log.e("error", "Make json object");
            }

            socket.emit("touch", jsonObject);
        }
    }
    
    
}

    

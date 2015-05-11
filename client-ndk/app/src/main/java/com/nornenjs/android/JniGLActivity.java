package com.nornenjs.android;


import android.content.Intent;
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
import java.util.Map;


public class JniGLActivity extends Activity {

    int mode = NONE;
    static final int NONE = 0;
    static final int DRAG = 1;

    float oldDist = 1.0f;
    float newDist = 1.0f;

    public float beforeX = 0.0f, beforeY = 0.0f;
    public float rotationX = 0.0f, rotationY = 0.0f;
    public float translationX =0.0f, translationY =0.0f;

    public float div=3.0f;

    public float oldVectorX1 =0.0f, oldVectorY1 =0.0f;
    public float oldVectorX2 =0.0f, oldVectorY2 =0.0f;

    public float newVectorX1 =0.0f, newVectorY1 =0.0f;
    public float newVectorX2 =0.0f, newVectorY2 =0.0f;

    public float oldMidVectorX=0.0f, oldMidVectorY=0.0f;
    public float newMidVectorX=0.0f, newMidVectorY=0.0f;

    public Integer count = 0;
    public Integer draw = 0;
    public Integer pinch = 0;
    public Integer rotation = 0;
    public Integer move = 0;


    private MyEventListener myEventListener;

    public int volumeWidth, volumeHeight, volumeDepth;
    public String volumeSavePath;

    public void setMyEventListener(MyEventListener myEventListener) {
        this.myEventListener = myEventListener;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Intent intent = getIntent();
        volumeWidth=  intent.getIntExtra("width",512);
        volumeHeight = intent.getIntExtra("height",512);
        volumeDepth = intent.getIntExtra("depth",200);
        volumeSavePath = intent.getStringExtra("savePath");

        String host = getString(R.string.host);
        setContentView(R.layout.activity_jni_gl);
        mGLSurfaceView = new TouchSurfaceView(this, host);
        setContentView(mGLSurfaceView);
        mGLSurfaceView.requestFocus();
        mGLSurfaceView.setFocusableInTouchMode(true);

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


                    myEventListener.RotationEvent(rotationX, rotationY);
                    rotation++;

                }
                else if(event.getPointerCount() == 2) { //multi touch

                    newVectorX1 = event.getX(0); newVectorX2 = event.getX(1);
                    newVectorY1 = event.getY(0); newVectorY2 = event.getY(1);

                    if((VecotrDirection(oldVectorX1,newVectorX1) == (VecotrDirection(oldVectorX2,newVectorX2)) &&  //multi touch translation
                            (VecotrDirection(oldVectorY1,newVectorY1) == (VecotrDirection(oldVectorY2,newVectorY2))))){

                            newDist = spacing(event);

                            newMidVectorX= midPoint(newVectorX1, newVectorX2);
                            newMidVectorY= midPoint(newVectorY1,newVectorY2);

                            translationX += (newMidVectorX - oldMidVectorX) / 250.0;
                            translationY -= (newMidVectorY - oldMidVectorY) / 250.0;

                            oldMidVectorX = newMidVectorX;
                            oldMidVectorY = newMidVectorY;

                            //translationPng = false;

                            myEventListener.TranslationEvent(translationX, translationY);
                            move++;

                    }
                    else{ // multi touch pinch zoom
                        newDist = spacing(event);

                        if (newDist - oldDist > 15) { // zoom in

                            oldDist = newDist;
                            div -= (((newDist / oldDist) / 50) * 10);

                            if (div <= 0.2f) {
                                div = 0.2f;
                            }

                            myEventListener.PinchZoomEvent(div);
                            pinch++;

                        } else if (oldDist - newDist > 15) { // zoom out

                            oldDist = newDist;
                            div += (((newDist / oldDist) / 50) * 10);

                            if (div >= 10.0f) {
                                div = 10.0f;
                            }

                            myEventListener.PinchZoomEvent(div);
                            pinch++;
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
            case MotionEvent.ACTION_POINTER_DOWN:

                oldDist = spacing(event);

                oldVectorX1 = event.getX(0);oldVectorX2 = event.getX(1);
                oldVectorY1 = event.getY(0);oldVectorY2 = event.getY(1);

                oldMidVectorX= midPoint(oldVectorX1,oldVectorX2);
                oldMidVectorY= midPoint(oldVectorY1,oldVectorY2);

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
    private float midPoint(float vector1, float vector2) {

        float midVector = vector1 + vector2;
        return midVector/2;

    }
    private boolean VecotrDirection(float vector1, float vector2) { //음수면 0 양수면 1

        if(vector2 - vector1<0){
            return false;
        }else{
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
    private String host;

    public TouchSurfaceView(Context context, String host) {
        super(context);
        this.host = host;
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
        private Integer width;
        private Integer height;
        private Socket relay;
        private Socket socket;



        public void bindSocket(String ipAddress, String port, String deviceNumber){
            try {
                socket = IO.socket("http://"+ipAddress+":"+port);

                JSONObject json = new JSONObject();

                json.put("savePath", mActivity.volumeSavePath);
                json.put("width", mActivity.volumeWidth);
                json.put("height", mActivity.volumeHeight);
                json.put("depth", mActivity.volumeDepth);

                socket.emit("join", deviceNumber);
                socket.emit("init", json);

                socket.on("loadCudaMemory", new Emitter.Listener() { //112.108.40.166
                    @Override
                    public void call(Object... args) {
                        socket.emit("androidPng");
                    }
                });

                socket.on("stream", new Emitter.Listener() { //112.108.40.166
                    @Override
                    public void call(Object... args) {

                        JSONObject info = (JSONObject) args[0];

                        //Log.d("ByteBuffer", info.toString());

                        try {
                            byteArray = (byte[]) info.get("data");
                            width = (Integer) info.get("width");
                            height = (Integer) info.get("height");
                            //Log.d("ByteBuffer", ""+width+" "+ height+ " " + byteArray.length);
                        } catch (JSONException e) {
                            e.printStackTrace();
                            Log.e("ByteBuffer", e.getMessage(),e);
                        }

                        mActivity.count++;
                    }
                });

                socket.connect();

            } catch (Exception e) {
                Log.e("socket", e.getMessage(), e);

            }
        }

        public CubeRenderer(JniGLActivity activity) {

            mActivity = activity;
            Log.d("socket", "connection");
            // ~ socket connection
            try {
                relay = IO.socket(host);
            } catch (URISyntaxException e) {
                e.printStackTrace();
            }

            relay.emit("getInfo", 0);

            /**
             * emit connect - response message
             */
            relay.on("getInfoClient", new Emitter.Listener() {

                @Override
                public void call(Object... args) {

                    JSONObject info = (JSONObject) args[0];

                    try {
                        Log.d("socket", "Connection");
                        if (!info.getBoolean("conn")) {
                            Log.d("socket", "Connection User is full");
                            return;
                        } else {
                            relay.disconnect();
                            String ipAddress = info.getString("ipAddress");
                            String port = info.getString("port");
                            String deviceNumber = info.getString("deviceNumber");

                            bindSocket(ipAddress, port, deviceNumber);
                        }
                    }catch (Exception e){
                        Log.e("Socket", e.getMessage(), e);
                    }
                }
            });

            relay.connect();

            mActivity.setMyEventListener(this);
        }

        Bitmap imgPanda;
        int[] pixels = new int[256*256];
        //int[] pixels2 = new int[512*512];

        public void onDrawFrame(GL10 gl) {

            if(byteArray!=null) {


                imgPanda = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
                imgPanda.getPixels(pixels, 0, width.intValue(), 0, 0, width.intValue(), height.intValue());

                mActivity.nativeSetTextureData(pixels, width.intValue(), height.intValue());
                mActivity.draw++;
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
        public void RotationEvent(float rotationX, float rotationY) {
            JSONObject jsonObject = new JSONObject();
            try {
                jsonObject.put("rotationX", rotationX);
                jsonObject.put("rotationY", rotationY);

            } catch (JSONException e) {
                e.printStackTrace();
                Log.e("error", "Make json object");
            }

            socket.emit("rotation", jsonObject);
        }
        @Override
        public void TranslationEvent(float translationX, float translationY) {
            JSONObject jsonObject = new JSONObject();
            try {
                jsonObject.put("positionX", translationX);
                jsonObject.put("positionY", translationY);

            } catch (JSONException e) {
                e.printStackTrace();
                Log.e("error", "Make json object");
            }

            socket.emit("translation", jsonObject);
        }
        @Override
        public void PinchZoomEvent(float div) {
            JSONObject jsonObject = new JSONObject();
            try {
                jsonObject.put("positionZ", div);

            } catch (JSONException e) {
                e.printStackTrace();
                Log.e("error", "Make json object");
            }

            socket.emit("pinchZoom", jsonObject);
        }
    }
    
    
}

    

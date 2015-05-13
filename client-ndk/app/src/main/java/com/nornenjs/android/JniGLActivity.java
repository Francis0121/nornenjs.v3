package com.nornenjs.android;


import android.content.Intent;
import android.os.Bundle;
import android.app.Activity;
import android.os.Handler;
import android.os.Message;
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


public class JniGLActivity extends Activity implements MyViewListener{


    static final String TAG = "JniGLActivity";
    int mode = NONE;
    static final int NONE = 0;
    static final int DRAG = 1;
    static final int VOLUME = 0;
    static final int MPRX = 1;
    static final int MPRY = 2;
    static final int MPRZ = 3;


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

    public boolean rotationPng = false;
    public boolean translationPng = false;
    public boolean pinchzoomPng = false;

    private MyEventListener myEventListener;
    GLSurfaceView mGLSurfaceView;

    public int volumeWidth, volumeHeight, volumeDepth;
    public String volumeSavePath = "/storage/data/eabd1bf4-83e2-429d-a35d-b20025f84de8";//일단 상수 박아줌
    public int datatype;

    public void setMyEventListener(MyEventListener myEventListener) {
        this.myEventListener = myEventListener;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Intent intent = getIntent();
        volumeWidth=  intent.getIntExtra("width",256);
        volumeHeight = intent.getIntExtra("height", 256);
        volumeDepth = intent.getIntExtra("depth", 255);
        volumeSavePath = intent.getStringExtra("savepath");
        datatype = intent.getIntExtra("datatype", 0);//0이 기본값

        Log.d("emitTag", "emit JNIActivity : " + datatype);

        String host = getString(R.string.host);
        setContentView(R.layout.loding);
        Log.d(TAG, "before create TouchSurfaceView");
        mGLSurfaceView = new TouchSurfaceView(this, host);
        //if(mGLSurfaceView.getRenderMode())
        //setContentView(mGLSurfaceView);
        Log.d(TAG, "setcontentView mGLSurfaceView");
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
        Log.d("bmp", "onDestroy");
        //조건부
        if(mGLSurfaceView != null)
            myEventListener.BackToPreview();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        //모든 이벤트는 이 액티비티가 받고 있음.

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

                        translationPng = false;
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
                            pinchzoomPng = false;
                            myEventListener.PinchZoomEvent(div);
                            pinch++;

                        } else if (oldDist - newDist > 15) { // zoom out

                            oldDist = newDist;
                            div += (((newDist / oldDist) / 50) * 10);

                            if (div >= 10.0f) {
                                div = 10.0f;
                            }
                            pinchzoomPng = false;
                            myEventListener.PinchZoomEvent(div);
                            pinch++;
                        }
                    }
                }
                break;

            case MotionEvent.ACTION_UP :
                mode = NONE;
                Log.d("emitTag", "Event ended");
                myEventListener.GetPng();
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
                break;


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
//    void setRenderingView()
//    {
//        setContentView(mGLSurfaceView);
//
//    }


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

    @Override
    public void setView() {
        //setContentView(mGLSurfaceView);
        if(!mGLSurfaceView.isShown())
        {
            Log.d(TAG, "setView() called");
            new Thread()

            {

                public void run()

                {

                    Message msg = handler.obtainMessage();

                    handler.sendMessage(msg);

                }

            }.start();
        }

    }

    final Handler handler = new Handler()
    {

        public void handleMessage(Message msg)

        {
            Log.d(TAG, "handleMessage() called");

            setContentView(mGLSurfaceView);

        }

    };
}

class TouchSurfaceView extends GLSurfaceView {

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        super.surfaceDestroyed(holder);
        Log.d("bmp", "surfaceDestoryed");
    }

    private CudaRenderer mRenderer;
    private JniGLActivity mActivity;
    private Context mContext;
    private String host;


    public TouchSurfaceView(Context context, String host) {
        super(context);
        this.host = host;
        this.mContext = context;
        this.mActivity = (JniGLActivity) context;
        this.mRenderer = new CudaRenderer(this.mActivity, host);
        Log.d("emitTag","make CudaRenderer");
        setRenderer(this.mRenderer);
    }

    @Override
    public boolean onTrackballEvent(MotionEvent e) {
        this.mActivity.nativeOnTrackballEvent(e.getAction(), e.getX(), e.getY());
        requestRender();
        return true;
    }

    /**
     * Render a cuda.
     */


}

class CudaRenderer implements GLSurfaceView.Renderer, MyEventListener {

    private JniGLActivity mActivity;
    private byte[] byteArray;
    private Integer width;
    private Integer height;
    private Socket relay;
    private Socket socket;

    Bitmap imgPanda;


    private String mprarr[] = {"none","transferScaleX","transferScaleY","transferScaleZ"};

    public void bindSocket(String ipAddress, String port, String deviceNumber){
        try {
            Log.d("emitTag", toString());
            socket = IO.socket("http://"+ipAddress+":"+port);

            JSONObject json = new JSONObject();
            Log.d("SurfaceView", "from intent " + mActivity.volumeWidth + ", " + mActivity.volumeHeight + ", " + mActivity.volumeDepth + ", " + mActivity.volumeSavePath);
            json.put("savePath", mActivity.volumeSavePath);
            json.put("width", mActivity.volumeWidth);
            json.put("height", mActivity.volumeHeight);
            json.put("depth", mActivity.volumeDepth);


            socket.emit("join", deviceNumber);
            socket.emit("init", json);


            socket.on("loadCudaMemory", new Emitter.Listener() { //112.108.40.166
                @Override
                public void call(Object... args) {


                    if(mActivity.datatype == mActivity.VOLUME) {
                        socket.emit("androidPng");
                        Log.d("emitTag","VOLUME emit");
                    }
                    else
                    {

                        try
                        {
                            JSONObject json2 = new JSONObject();
                            json2.put("mprType", mActivity.datatype);
                            json2.put(mprarr[mActivity.datatype], 0.5);//처음에는 50을 줌
                            Log.d("emitTag",mprarr[mActivity.datatype] + " : " + 0.5);
                            for(int i = 1; i < 4; i++)
                            {
                                if(i != mActivity.datatype)
                                {
                                    json2.put(mprarr[i], 0);//처음에는 50을 줌
                                    Log.d("emitTag", mprarr[i] + " : " + 0);
                                }
                            }
                            socket.emit("mpr", json2);

                        }catch(JSONException e)
                        {

                        }

                        Log.d("emitTag", "mpr emit..type : " + mActivity.datatype);
                    }



                }
            });

            socket.on("stream", new Emitter.Listener() { //112.108.40.166
                @Override
                public void call(Object... args) {

                    JSONObject info = (JSONObject) args[0];

                    Log.d("ByteBuffer", info.toString());

                    try {
                        byteArray = (byte[]) info.get("data");
                        width = (Integer) info.get("width");
                        height = (Integer) info.get("height");
                        mActivity.setView();
                        imgPanda = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);

                        Log.d("pixels", "getWidth()1 : " + imgPanda.getWidth() + ", getHeight() : " + imgPanda.getHeight());
                        Log.d("pixels", "width.intValue()1 : " + width + ", height.intValue() : " + height);

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

    public CudaRenderer(JniGLActivity activity, String host) {

        mActivity = activity;
        Log.d("emitTag", "connection");
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
                    Log.d("emitTag", "Connection");
                    if (!info.getBoolean("conn")) {
                        Log.d("emitTag", "Connection User is full");
                        return;
                    } else {
                        relay.disconnect();
                        relay.off("getInfoClient");

                        String ipAddress = info.getString("ipAddress");
                        String port = info.getString("port");
                        String deviceNumber = info.getString("deviceNumber");

                        Log.d("emitTag", "bindSocket() call");
                        bindSocket(ipAddress, port, deviceNumber);
                    }
                } catch (Exception e) {
                    Log.e("Socket", e.getMessage(), e);
                }
            }
        });

        relay.connect();

        mActivity.setMyEventListener(this);
    }


    //int[] pixels2 = new int[512*512];

    public void onDrawFrame(GL10 gl) {

        if(byteArray!=null) {

//            if(imgPanda != null)
//            {
//                imgPanda.recycle();
//                imgPanda = null;
//            }

            //imgPanda = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);

            int size;
//            if(width.intValue() == 256){
//                size = 256;
//            }
//            else
//            {
//                size = 512;
//            }

            int[] pixels = new int[width.intValue()*height.intValue()];
            //Log.d("pixels","size : " + size);
//
//            imgPanda.getPixels(pixels, 0, size, 0, 0, size, size);
            Log.d("pixels", "getWidth()2 : " + imgPanda.getWidth() + ", getHeight() : " + imgPanda.getHeight());
            Log.d("pixels", "width.intValue()2 : " + width.intValue() + ", height.intValue() : " + height.intValue());
            //imgPanda.getPixels(pixels, 0, width.intValue(), 0, 0, width.intValue(), height.intValue());
            if(imgPanda.getWidth() == width.intValue())
            {
                imgPanda.getPixels(pixels, 0, width.intValue(), 0, 0, width.intValue(), height.intValue());
                mActivity.nativeSetTextureData(pixels, width.intValue(), height.intValue());
                mActivity.draw++;
            }


            //mActivity.nativeSetTextureData(pixels, size, size);
            //mActivity.draw++;
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

    @Override
    public void GetPng() {
        JSONObject jsonObject = new JSONObject();
        socket.emit("androidPng", jsonObject);
        Log.d("GETPng","GETPng");

    }

    @Override
    public void BackToPreview() {
        Log.e("emitTag", "Back to PreViewActivity..");
        socket.disconnect();
        socket.off("loadCudaMemory");
        socket.off("stream");
        Log.e("emitTag", "socket.disconnect()");
    }
}

    

package com.nornenjs.android;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PixelFormat;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.FloatMath;
import android.util.Log;
import android.view.*;
import android.widget.*;
import com.github.nkzawa.emitter.Emitter;
import com.github.nkzawa.socketio.client.IO;
import com.github.nkzawa.socketio.client.Socket;
import org.json.JSONException;
import org.json.JSONObject;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;
import java.net.URISyntaxException;



//import android.content.Intent;
//import android.os.Bundle;
//import android.app.Activity;
//import android.os.Handler;
//import android.os.Message;
//import android.util.FloatMath;
//import android.util.Log;
//import android.view.MotionEvent;
//
//import javax.microedition.khronos.egl.EGLConfig;
//import javax.microedition.khronos.opengles.GL10;
//import android.graphics.Bitmap;
//import android.graphics.BitmapFactory;
//import android.content.Context;
//import android.opengl.GLSurfaceView;
//import android.view.SurfaceHolder;
//import com.github.nkzawa.emitter.Emitter;
//import com.github.nkzawa.socketio.client.IO;
//import com.github.nkzawa.socketio.client.Socket;
//import org.json.JSONException;
//import org.json.JSONObject;
//
//import java.net.URISyntaxException;
//import java.util.Map;


public class MprActivity extends Activity {

//setContentView(R.layout.activity_mpr);
    static final String TAG = "MprActivity";

    static final int VOLUME = 0;
    static final int MPRX = 1;
    static final int MPRY = 2;
    static final int MPRZ = 3;


    private MyEventListener myEventListener;
    GLSurfaceView mGLSurfaceView;
    MPRRenderer mRenderer;

    public int volumeWidth, volumeHeight, volumeDepth;
    public String volumeSavePath = "/storage/data/eabd1bf4-83e2-429d-a35d-b20025f84de8";//일단 상수 박아줌
    public int datatype;

    public Integer count = 0;
    public Integer draw = 0;

    SeekBar sb;

    public void setMyEventListener(MyEventListener myEventListener) {
        this.myEventListener = myEventListener;
    }

    public ChangeView changeView;


    String host;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.loding);

        Intent intent = getIntent();
        volumeWidth=  intent.getIntExtra("width", 256);
        volumeHeight = intent.getIntExtra("height", 256);
        volumeDepth = intent.getIntExtra("depth", 255);
        volumeSavePath = intent.getStringExtra("savepath");
        datatype = intent.getIntExtra("datatype", 0);//0이 기본값이었는데 바꿨음..mpr이니까

        Log.d("emitTag", "emit MprActivity : " + datatype);

        host = getString(R.string.host);


        Log.d(TAG, "before create TouchSurfaceView");
        mGLSurfaceView = new TouchSurfaceView(this, host);
        Log.d(TAG, "setcontentView mGLSurfaceView");

        mRenderer = new MPRRenderer(this, host);
//        if(intent.getStringExtra("step").equals("preview"))
//            mRenderer = new MPRRenderer(this, host);
//        else
//            mRenderer = new MPRRenderer(this);
//        Log.d("emitTag","make CudaRenderer");

        mGLSurfaceView.setRenderer(this.mRenderer);

        mGLSurfaceView.requestFocus();
        mGLSurfaceView.setFocusableInTouchMode(true);

        changeView = new ChangeView(MprActivity.this);
        //GLSurfaceView.getHolder().setFormat(PixelFormat.TRANSLUCENT);

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

    //@Override



    class TouchSurfaceView extends GLSurfaceView {

        @Override
        public void surfaceDestroyed(SurfaceHolder holder) {
            super.surfaceDestroyed(holder);
            Log.d("bmp", "surfaceDestoryed");
        }

        //public MPRRenderer mRenderer;
        private MprActivity mActivity;
        private Context mContext;
        private String host;


        public TouchSurfaceView(Context context, String host) {
            super(context);
            this.host = host;
            this.mContext = context;
            this.mActivity = (MprActivity) context;
//            this.mRenderer = new MPRRenderer(this.mActivity, host);
//            Log.d("emitTag","make CudaRenderer");
//            setRenderer(this.mRenderer);
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

    class MPRRenderer implements GLSurfaceView.Renderer, MyEventListener, SeekBar.OnSeekBarChangeListener {

        private MprActivity mActivity;
        private byte[] byteArray;
        private Integer width;
        private Integer height;
        private Socket relay;
        private Socket socket;

        public double transferScale = 0.5;
        Bitmap imgPanda;


        private String mprarr[] = {"none","transferScaleX","transferScaleY","transferScaleZ"};

        public void bindSocket(String ipAddress, String port, String deviceNumber){
            try {
                Log.d("emitTag", toString());
                socket = IO.socket("http://"+ipAddress+":"+port);

                JSONObject json = new JSONObject();
                json.put("savePath", mActivity.volumeSavePath);
                json.put("width", mActivity.volumeWidth);
                json.put("height", mActivity.volumeHeight);
                json.put("depth", mActivity.volumeDepth);


                socket.emit("join", deviceNumber);
                socket.emit("init", json);

                Log.d(TAG, "in Renerer");

                socket.on("loadCudaMemory", new Emitter.Listener() { //112.108.40.166
                    @Override
                    public void call(Object... args) {


                        if (mActivity.datatype == mActivity.VOLUME) {
                            socket.emit("androidPng");
                            Log.d("emitTag", "VOLUME emit");
                        } else {

                            try {

                                JSONObject json2 = new JSONObject();
                                json2.put("mprType", mActivity.datatype);
                                json2.put(mprarr[mActivity.datatype], transferScale);//처음에는 50을 줌
                                json2.put("positionZ", 4.5);
                                Log.d("emitTag", mprarr[mActivity.datatype] + " : " + 0.5);
                                for (int i = 1; i < 4; i++) {
                                    if (i != mActivity.datatype) {
                                        json2.put(mprarr[i], 0);//처음에는 50을 줌
                                        Log.d("emitTag", mprarr[i] + " : " + 0);
                                    }
                                }
                                if (!mGLSurfaceView.isShown()) {
                                    json2.put("png", "ok");//뭐지?
                                }
                                //옵션으로 1을 줄지 말지
                                socket.emit("mpr", json2);


                            } catch (JSONException e) {

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
                            changeView.setMprView();
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

        public MPRRenderer(Context activity)
        {
            mActivity = (MprActivity) activity;
            mActivity.setMyEventListener(this);

        }


        //public MPRRenderer(MprActivity activity, String host) {
        public MPRRenderer(Context activity, String host) {

            mActivity = (MprActivity) activity;
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



        int[] pixels2 = new int[512*512];
        int[] pixels = new int[256*256];

        public void onDrawFrame(GL10 gl) {

            if(imgPanda!=null) {

                if(width.intValue() == 512)
                {
                    imgPanda.getPixels(pixels2, 0, width.intValue(), 0, 0, width.intValue(), height.intValue());
                    mActivity.nativeSetTextureData(pixels2, width.intValue(), height.intValue());
                }
                else
                {
                    imgPanda.getPixels(pixels, 0, width.intValue(), 0, 0, width.intValue(), height.intValue());
                    mActivity.nativeSetTextureData(pixels, width.intValue(), height.intValue());
                }
                
            }
            else
                Log.d("Jni", "byteArray is null");

            mActivity.nativeDrawIteration(0, 0);

        }

        public void onSurfaceChanged(GL10 gl, int width, int height) {
            mActivity.nativeResize(width, height);
        }

        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            Log.d("bmp", "onSurfaceCreated");
            mActivity.nativeInitGL();
        }



        @Override
        public void RotationEvent(float rotationX, float rotationY) {}

        @Override
        public void TranslationEvent(float translationX, float translationY) {}

        @Override
        public void PinchZoomEvent(float div) {}

        @Override
        public void GetPng() {}

        @Override
        public void OtfEvent(int start, int middle1, int middle2, int end, int flag) {}

        @Override
        public void BrightnessEvent(float brightness) {

        }

        @Override
        public void BackToPreview() {
            Log.e("emitTag", "Back to PreViewActivity..");
            if(socket != null && socket.connected())
            {
                socket.disconnect();
                socket.off("loadCudaMemory");
                socket.off("stream");
                Log.e("emitTag", "socket.disconnect()");
            }
        }

        @Override
        public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
            progressEmit(progress, 0);
        }

        @Override
        public void onStartTrackingTouch(SeekBar seekBar) {}

        @Override
        public void onStopTrackingTouch(SeekBar seekBar) {
            progressEmit(-1, 1);
        }

        public void progressEmit(int progress, int pngOption)
        {
            if(pngOption == 0)
            {
                if(progress == 0)
                    transferScale = 0;
                else
                    transferScale = progress/100.0;
            }

            JSONObject json2 = new JSONObject();
            try
            {
                json2.put("mprType", mActivity.datatype);
                json2.put(mprarr[mActivity.datatype], transferScale);//처음에는 50을 줌
                json2.put("positionZ", 4.5);

                for (int i = 1; i < 4; i++) {
                    if (i != mActivity.datatype) {
                        json2.put(mprarr[i], 0);//처음에는 50을 줌
                        Log.d("emitTag", mprarr[i] + " : " + 0);
                    }
                }

            }catch(JSONException e)
            {
                e.printStackTrace();
            }
            Log.d("emitTag", mprarr[mActivity.datatype] + " : " + transferScale);

            //옵션으로 1을 줄지 말지
            if(pngOption == 1)
            {
                try
                {
                    json2.put("png", "ok");
                }catch (JSONException e)
                {
                    e.printStackTrace();
                }
            }

            socket.emit("mpr", json2);
        }

    }
}





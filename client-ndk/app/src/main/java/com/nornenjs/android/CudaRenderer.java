package com.nornenjs.android;

/**
 * Created by hyok on 15. 5. 4.
 */

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLSurfaceView;
import android.os.SystemClock;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.SeekBar;
import android.widget.TextView;
import com.github.nkzawa.emitter.Emitter;
import com.github.nkzawa.socketio.client.IO;
import com.github.nkzawa.socketio.client.Socket;
import com.nineoldandroids.view.ViewPropertyAnimator;
import org.json.JSONException;
import org.json.JSONObject;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;
import java.net.URISyntaxException;
import java.util.List;


class CudaRenderer implements GLSurfaceView.Renderer, MyEventListener, View.OnClickListener, SeekBar.OnSeekBarChangeListener{

    private JniGLActivity mActivity;
    private byte[] byteArray;
    private Integer width;
    private Integer height;
    private Socket relay;
    private Socket socket;

    private final int VOLUME = 1;
    private final int MIP = 2;
    private final int OPTION = 3;
    private final int MRPX = 4;
    private final int MRPY = 5;
    private final int MRPZ = 6;

    Bitmap imgPanda;
    Bitmap imgPanda2;

    public boolean mip = false;

    private String mprarr[] = {"none","transferScaleX","transferScaleY","transferScaleZ"};
    public double transferScale = 0.5;

    public void bindSocket(String ipAddress, String port, String deviceNumber){
        try {
            //Log.d("emitTag", toString());
            socket = IO.socket("http://" + ipAddress + ":" + port);

            JSONObject json = new JSONObject();
            Log.d("SurfaceView", "from intent " + mActivity.volumeWidth + ", " + mActivity.volumeHeight + ", " + mActivity.volumeDepth + ", " + mActivity.volumeSavePath);
            json.put("savePath", mActivity.volumeSavePath);
            json.put("width", mActivity.volumeWidth);
            json.put("height", mActivity.volumeHeight);
            json.put("depth", mActivity.volumeDepth);

            if(mip)
                json.put("mip", "mip");

            Log.d("emitTag", "emit join");
            socket.emit("join", deviceNumber);
            Log.d("emitTag", "emit init");
            socket.emit("init", json);


            socket.on("loadCudaMemory", new Emitter.Listener() { //112.108.40.166
                @Override
                public void call(Object... args) {

                    JSONObject jsonObject = new JSONObject();
                    socket.emit("androidPng", jsonObject);

                    Log.d("emitTag", "VOLUME emit");


                }
            });

            socket.on("stream", new Emitter.Listener() { //112.108.40.166
                @Override
                public void call(Object... args) {

                    JSONObject info = (JSONObject) args[0];

                    Log.d("ByteBuffer", info.toString());

                    try {
                        byteArray = (byte[]) info.get("data");
                        imgPanda2 = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
                        imgPanda = imgPanda2;
                        width = (Integer) info.get("width");
                        height = (Integer) info.get("height");
                        mActivity.mGLSurfaceView.requestRender();
                        mActivity.changeView.setView();
                        //Log.d("pixels", "getWidth()1 : " + imgPanda.getWidth() + ", getHeight() : " + imgPanda.getHeight());
                        //Log.d("pixels", "width.intValue()1 : " + width + ", height.intValue() : " + height);

                    } catch (JSONException e) {
                        e.printStackTrace();
                        Log.e("ByteBuffer", e.getMessage(), e);
                    }

                    mActivity.count++;
                }
            });

            Log.d("emitTag", "socket connect");
            socket.connect();

            if(socket.connected())
            {
                Log.d("emitTag", "socket connected");
            }
            else
            {
                Log.d("emitTag", "socket unconnected");
            }

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
        Log.d("emitTag", "relay emit");
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

        Log.d("emitTag", "relay connect");
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

//            imgPanda.getPixels(pixels2, 0, 256, 0, 0, 256, 256);
//            mActivity.nativeSetTextureData(pixels2, 256, 256);
//            mActivity.draw++;
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

            if(mip)
                jsonObject.put("mip", "mip");

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

            if(mip)
                jsonObject.put("mip", "mip");

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

            if(mip)
                jsonObject.put("mip", "mip");

        } catch (JSONException e) {
            e.printStackTrace();
            Log.e("error", "Make json object");
        }

        socket.emit("pinchZoom", jsonObject);
    }

    @Override
    public void GetPng() {
        JSONObject jsonObject = new JSONObject();
        try
        {
            if(mip)
                jsonObject.put("mip", "mip");

        }catch(JSONException e)
        {
            e.printStackTrace();
        }

        socket.emit("androidPng", jsonObject);
        Log.d("GETPng", "GETPng");

    }

    @Override
    public void OtfEvent(int start, int middle1, int middle2, int end, int flag) {
        Log.d("OTF", "EventConfirm");
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("start", start);
            jsonObject.put("middle1", middle1);
            jsonObject.put("middle2", middle2);
            jsonObject.put("end", end);
            jsonObject.put("flag", flag);


        } catch (JSONException e) {
            e.printStackTrace();
            Log.e("error", "Make json object");
        }

        socket.emit("OTF", jsonObject);
    }


    @Override
    public void BackToPreview() {
        Log.e("emitTag", "Back to PreViewActivity..");

        try{

            Log.d("", "loop beak???");
            socket.disconnect();

            socket.off("loadCudaMemory");
            socket.off("stream");
            socket = null;
            Log.e("emitTag", "socket try catch");
        }catch (Exception e){
            e.printStackTrace();;
        }

        try{

//            while(!socket.connected()) {
//                //Log.e("000000000", "00000000000000");
//            }
            //SystemClock.sleep(1000);
            relay.disconnect();
            relay.off("getInfoClient");
            Log.e("emitTag", "socket try catch");
        }catch (Exception e){
            e.printStackTrace();;
        }

    }

    @Override
    public void BrightnessEvent(float brightness) {
        Log.d("BrightnessEvent", "BrightnessEvent : " + brightness + "calc : " + (brightness / 100.0));
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("brightness", (brightness / 100.0));

        } catch (JSONException e) {
            e.printStackTrace();
            Log.e("error", "Make json object");
        }

        socket.emit("Brightness", jsonObject);
    }



    void getMPR(int position)
    {

        Log.d("getmprEvent", "position : " + position);

        try {

            JSONObject json2 = new JSONObject();
            json2.put("mprType", position);
            json2.put(mprarr[position], transferScale);//처음에는 50을 줌
            json2.put("positionZ", 4.5);
            Log.d("emitTag", mprarr[position] + " : " + 0.5);

            for (int i = 1; i < 4; i++) {
                if (i != position) {
                    json2.put(mprarr[i], 0);//처음에는 50을 줌
                    Log.d("emitTag", mprarr[i] + " : " + 0);
                }
            }

            json2.put("png", "ok");
            socket.emit("mpr", json2);


        } catch (JSONException e) {

        }

    }

    @Override
    public void onClick(View v) {
        Log.d("click", "btn click : ");
        switch (v.getId())
        {
            case R.id.toggleVol :

                if(mActivity.datatype != 0){
                    mip = false;

                    mActivity.datatype = 0;
                    GetPng();

                    setBackgroundResource(VOLUME);

                    mActivity.mrpSb.setVisibility(View.INVISIBLE);
                }

                break;

            case R.id.toggleMip :

                mip = true;
                mActivity.datatype = MIP;
                checkOtfFlag(mActivity.menuFlag);

                GetPng();

                setBackgroundResource(MIP);
                mActivity.mrpSb.setVisibility(View.INVISIBLE);

                break;

            case R.id.toggleMenu :

                if(!mActivity.mRenderer.mip && !mActivity.mrpSb.isShown())
                {
                    if(!mActivity.menuFlag) {
                        ViewPropertyAnimator.animate(mActivity.otf_table).translationY(0).setDuration(600);
                        mActivity.toggleMenu.setBackgroundResource(R.drawable.option_on);
                    }
                    else {
                        ViewPropertyAnimator.animate(mActivity.otf_table).translationY(mActivity.otf_table.getHeight()).setDuration(600);
                        mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);
                    }
                    mActivity.menuFlag = !mActivity.menuFlag;

                }

                break;


            case R.id.positionX :

                getMPR(mActivity.MPRX);
                mActivity.datatype = mActivity.MPRX;

                checkOtfFlag(mActivity.menuFlag);

                mActivity.mrpSb.setVisibility(View.VISIBLE);
                setBackgroundResource(MRPX);

                break;
            case R.id.positionY :

                getMPR(mActivity.MPRY);
                mActivity.datatype = mActivity.MPRY;

                checkOtfFlag(mActivity.menuFlag);

                mActivity.mrpSb.setVisibility(View.VISIBLE);
                setBackgroundResource(MRPY);

                break;
            case R.id.positionZ :

                getMPR(mActivity.MPRZ);
                mActivity.datatype = mActivity.MPRZ;

                checkOtfFlag(mActivity.menuFlag);

                mActivity.mrpSb.setVisibility(View.VISIBLE);
                setBackgroundResource(MRPZ);
                break;
        }
    }

    public void setBackgroundResource(int onFlag) {
        switch(onFlag)
        {

            case VOLUME :

                mActivity.togglebtn.setBackgroundResource(R.drawable.volume_on);
                mActivity.toggleMip.setBackgroundResource(R.drawable.mip_off);
                mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);
                mActivity.positionX.setBackgroundResource(R.drawable.mprx_off);
                mActivity.positionY.setBackgroundResource(R.drawable.mpry_off);
                mActivity.positionZ.setBackgroundResource(R.drawable.mprz_off);
                break;

            case  MIP :

                mActivity.togglebtn.setBackgroundResource(R.drawable.volume_off);
                mActivity.toggleMip.setBackgroundResource(R.drawable.mip_on);
                mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);
                mActivity.positionX.setBackgroundResource(R.drawable.mprx_off);
                mActivity.positionY.setBackgroundResource(R.drawable.mpry_off);
                mActivity.positionZ.setBackgroundResource(R.drawable.mprz_off);
                break;

            case MRPX :

                mActivity.togglebtn.setBackgroundResource(R.drawable.volume_off);
                mActivity.positionX.setBackgroundResource(R.drawable.mprx_on);
                mActivity.positionY.setBackgroundResource(R.drawable.mpry_off);
                mActivity.positionZ.setBackgroundResource(R.drawable.mprz_off);
                mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);
                mActivity.toggleMip.setBackgroundResource(R.drawable.mip_off);
                break;

            case MRPY :

                mActivity.togglebtn.setBackgroundResource(R.drawable.volume_off);
                mActivity.positionX.setBackgroundResource(R.drawable.mprx_off);
                mActivity.positionY.setBackgroundResource(R.drawable.mpry_on);
                mActivity.positionZ.setBackgroundResource(R.drawable.mprz_off);
                mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);
                mActivity.toggleMip.setBackgroundResource(R.drawable.mip_off);
                break;

            case MRPZ :

                mActivity.togglebtn.setBackgroundResource(R.drawable.volume_off);
                mActivity.positionX.setBackgroundResource(R.drawable.mprx_off);
                mActivity.positionY.setBackgroundResource(R.drawable.mpry_off);
                mActivity.positionZ.setBackgroundResource(R.drawable.mprz_on);
                mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);
                mActivity.toggleMip.setBackgroundResource(R.drawable.mip_off);
                break;
        }

    }
    public void checkOtfFlag(boolean menuflag)
    {
        if (menuflag) {
            ViewPropertyAnimator.animate(mActivity.otf_table).translationY(mActivity.otf_table.getHeight()).setDuration(600);
            mActivity.menuFlag= !menuflag;//false
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
        Log.d("progressEmit", "progress : " + progress);

        if(pngOption == 0)
        {
            if(progress <= 0)
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




package com.nornenjs.android;

/**
 * Created by hyok on 15. 5. 4.
 */

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLSurfaceView;
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

            socket.emit("join", deviceNumber);
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
        if(socket != null && socket.connected())
        {
            socket.disconnect();
            socket.off("loadCudaMemory");
            socket.off("stream");
            socket = null;
            Log.e("emitTag", "socket.disconnect()");
        }

        if(relay != null && relay.connected())
        {
            Log.e("emitTag", "relay.disconnect()");
            relay.disconnect();
            relay.off("getInfoClient");
        }
        else
        {
            Log.e("emitTag", "relay is null or unconnect");
        }
    }

    @Override
    public void BrightnessEvent(float brightness) {
        Log.d("BrightnessEvent", "BrightnessEvent : " + brightness + "calc : " + (brightness / 100.0));
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("brightness", (brightness/100.0));

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
//            if (!mActivity.mGLSurfaceView.isShown()) {
//                json2.put("png", "ok");//뭐지?
//            }
            json2.put("png", "ok");
            //옵션으로 1을 줄지 말지
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


                mip = false;
                v.setBackgroundResource(R.drawable.volume_on);
                mActivity.toggleMip.setBackgroundResource(R.drawable.mip_off);
                mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);

                mActivity.datatype = 0;
                GetPng();

                mActivity.positionX.setBackgroundResource(R.drawable.mprx_off);
                mActivity.positionY.setBackgroundResource(R.drawable.mpry_off);
                mActivity.positionZ.setBackgroundResource(R.drawable.mprz_off);

                mActivity.mrpSb.setVisibility(View.INVISIBLE);

//                if(mip)
//                {
//                    mip = false;
//                    mActivity.toggleMip.setBackgroundResource(R.drawable.mip_off);
//                }
//
//                if(mActivity.mrpSb.isShown())
//                {
//                    mActivity.positionX.setBackgroundResource(R.drawable.mprx_off);
//                    mActivity.positionY.setBackgroundResource(R.drawable.mpry_off);
//                    mActivity.positionZ.setBackgroundResource(R.drawable.mprz_off);
//
//                    mActivity.mrpSb.setVisibility(View.INVISIBLE);
//                }
//
//                v.setBackgroundResource(R.drawable.volume_on);
//                mActivity.datatype = 0;
//                GetPng();

                break;

            case R.id.toggleMip :
                mip = true;

                mActivity.togglebtn.setBackgroundResource(R.drawable.volume_off);
                v.setBackgroundResource(R.drawable.mip_on);
                mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);

                if(mActivity.menuFlag) {
                    ViewPropertyAnimator.animate(mActivity.otf_table).translationY(mActivity.otf_table.getHeight()).setDuration(600);
                    mActivity.menuFlag = !mActivity.menuFlag;
                }
                GetPng();

                mActivity.positionX.setBackgroundResource(R.drawable.mprx_off);
                mActivity.positionY.setBackgroundResource(R.drawable.mpry_off);
                mActivity.positionZ.setBackgroundResource(R.drawable.mprz_off);

                mActivity.mrpSb.setVisibility(View.INVISIBLE);
                break;

            case R.id.toggleMenu :
                Log.d("click", "mip click : " + mActivity.mRenderer.mip);

                if(!mActivity.mRenderer.mip && !mActivity.mrpSb.isShown())
                {
                    Log.d("click", "mip click mActivity.menuFlag : " + mActivity.menuFlag);
                    if(!mActivity.menuFlag) {
                        Log.d("Onclick", "otf_table.getHeight()3 : " + mActivity.otf_table.getHeight());
                        Log.d("Onclick", "otf_table.getY()3 : " + mActivity.otf_table.getY());
                        ViewPropertyAnimator.animate(mActivity.otf_table).translationY(0).setDuration(600);
                        mActivity.toggleMenu.setBackgroundResource(R.drawable.option_on);
                    }
                    else {
                        Log.d("Onclick", "otf_table.getHeight()4 : " + mActivity.otf_table.getHeight());
                        Log.d("Onclick", "otf_table.getY()4 : " + mActivity.otf_table.getY());
                        ViewPropertyAnimator.animate(mActivity.otf_table).translationY(mActivity.otf_table.getHeight()).setDuration(600);
                        mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);
                    }
                    mActivity.menuFlag = !mActivity.menuFlag;

                }
                else
                    Log.d("Onclick","mActivity.mRenderer.mip : " + mActivity.mRenderer.mip);

                break;


            case R.id.positionX :

                getMPR(1);

                mActivity.datatype = 1;
                if(mActivity.menuFlag) {
                    ViewPropertyAnimator.animate(mActivity.otf_table).translationY(mActivity.otf_table.getHeight()).setDuration(600);
                    mActivity.menuFlag = false;
                }
                mActivity.mrpSb.setVisibility(View.VISIBLE);
                mActivity.togglebtn.setBackgroundResource(R.drawable.volume_off);

                mActivity.positionX.setBackgroundResource(R.drawable.mprx_on);
                mActivity.positionY.setBackgroundResource(R.drawable.mpry_off);
                mActivity.positionZ.setBackgroundResource(R.drawable.mprz_off);
                mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);
                mActivity.toggleMip.setBackgroundResource(R.drawable.mip_off);

                break;
            case R.id.positionY :

                getMPR(2);

                mActivity.datatype = 2;
                if(mActivity.menuFlag) {
                    ViewPropertyAnimator.animate(mActivity.otf_table).translationY(mActivity.otf_table.getHeight()).setDuration(600);
                    mActivity.menuFlag = false;
                }
                mActivity.mrpSb.setVisibility(View.VISIBLE);
                mActivity.togglebtn.setBackgroundResource(R.drawable.volume_off);

                mActivity.positionX.setBackgroundResource(R.drawable.mprx_off);
                mActivity.positionY.setBackgroundResource(R.drawable.mpry_on);
                mActivity.positionZ.setBackgroundResource(R.drawable.mprz_off);
                mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);
                mActivity.toggleMip.setBackgroundResource(R.drawable.mip_off);

                break;
            case R.id.positionZ :

                getMPR(3);

                mActivity.datatype = 3;
                if(mActivity.menuFlag) {
                    ViewPropertyAnimator.animate(mActivity.otf_table).translationY(mActivity.otf_table.getHeight()).setDuration(600);
                    mActivity.menuFlag = false;
                }
                mActivity.mrpSb.setVisibility(View.VISIBLE);
                mActivity.togglebtn.setBackgroundResource(R.drawable.volume_off);

                mActivity.positionX.setBackgroundResource(R.drawable.mprx_off);
                mActivity.positionY.setBackgroundResource(R.drawable.mpry_off);
                mActivity.positionZ.setBackgroundResource(R.drawable.mprz_on);
                mActivity.toggleMenu.setBackgroundResource(R.drawable.option_off);
                mActivity.toggleMip.setBackgroundResource(R.drawable.mip_off);

                break;
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




package com.nornenjs.android;

import android.content.Intent;
import android.os.Bundle;
import android.app.Activity;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.util.FloatMath;
import android.util.Log;
import android.view.*;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import com.nornenjs.android.ChangeView;


public class JniGLActivity extends Activity{


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

    public float div=2.0f;//3

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

    MyEventListener myEventListener;
    public GLSurfaceView mGLSurfaceView;
    CudaRenderer mRenderer;

    RelativeLayout otf_table;

    boolean menuFlag = true;

    public ChangeView changeView;

    SeekBar mrpSb;

    public int volumeWidth, volumeHeight, volumeDepth;
    public String volumeSavePath = "/storage/data/eabd1bf4-83e2-429d-a35d-b20025f84de8";//일단 상수 박아줌
    public int datatype;//4는 MIP

    Button togglebtn, toggleMip, toggleMenu, positionX, positionY, positionZ;

    public void setMyEventListener(MyEventListener myEventListener) {
        this.myEventListener = myEventListener;
    }


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Intent intent = getIntent();
        volumeWidth=  intent.getIntExtra("width", 256);
        volumeHeight = intent.getIntExtra("height", 256);
        volumeDepth = intent.getIntExtra("depth", 255);
        volumeSavePath = intent.getStringExtra("savepath");
        datatype = intent.getIntExtra("datatype", 0);//0이 기본값

        Log.d("emitTag", "emit JNIActivity : " + datatype);

        String host = getString(R.string.host);
        setContentView(R.layout.loding);

        mGLSurfaceView = new TouchSurfaceView(this, host);
        mRenderer = new CudaRenderer(this, host);
        mGLSurfaceView.setRenderer(mRenderer);
        mGLSurfaceView.requestFocus();
        mGLSurfaceView.setFocusableInTouchMode(true);

        mGLSurfaceView.setRenderMode(mGLSurfaceView.RENDERMODE_WHEN_DIRTY);

        changeView = new ChangeView(JniGLActivity.this);

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
        Log.d("bmp", "onDestroy");
        //조건부

//        if(mGLSurfaceView != null)
//            myEventListener.BackToPreview();
        super.onDestroy();
    }

    @Override
    public void onBackPressed() {

        if(mGLSurfaceView != null)
            myEventListener.BackToPreview();
        super.onBackPressed();
    }

    int touchCount;
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        //모든 이벤트는 이 액티비티가 받고 있음.

        if(mGLSurfaceView != null && mGLSurfaceView.isShown() && !mrpSb.isShown()) {


            int act = event.getAction();
            switch (act & MotionEvent.ACTION_MASK) {

                case MotionEvent.ACTION_DOWN:
                    if (event.getPointerCount() == 1) {

                        beforeX = event.getX();  //posX1
                        beforeY = event.getY();  //posY1
                        mode = DRAG;
                        touchCount = 0;
                    }
                    break;

                case MotionEvent.ACTION_MOVE:

                    if (mode == DRAG && event.getPointerCount() == 1) {

                        rotationX += (event.getX() - beforeX) / 10.0;
                        rotationY += (event.getY() - beforeY) / 10.0;

                        beforeX = event.getX();
                        beforeY = event.getY();

                        if((++touchCount)%3 == 0)
                            myEventListener.RotationEvent(rotationX, rotationY);
                        rotation++;

                    } else if (event.getPointerCount() == 2) { //multi touch

                        newVectorX1 = event.getX(0);
                        newVectorX2 = event.getX(1);
                        newVectorY1 = event.getY(0);
                        newVectorY2 = event.getY(1);

                        if ((VecotrDirection(oldVectorX1, newVectorX1) == (VecotrDirection(oldVectorX2, newVectorX2)) &&  //multi touch translation
                                (VecotrDirection(oldVectorY1, newVectorY1) == (VecotrDirection(oldVectorY2, newVectorY2))))) {

                            newDist = spacing(event);

                            newMidVectorX = midPoint(newVectorX1, newVectorX2);
                            newMidVectorY = midPoint(newVectorY1, newVectorY2);

                            translationX += (newMidVectorX - oldMidVectorX) / 250.0;
                            translationY -= (newMidVectorY - oldMidVectorY) / 250.0;

                            oldMidVectorX = newMidVectorX;
                            oldMidVectorY = newMidVectorY;

                            translationPng = false;
                            if((++touchCount)%3 == 0)
                                myEventListener.TranslationEvent(translationX, translationY);
                            move++;

                        } else { // multi touch pinch zoom
                            newDist = spacing(event);

                            if (newDist - oldDist > 15) { // zoom in

                                oldDist = newDist;
                                div -= (((newDist / oldDist) / 50) * 10);

                                if (div <= 0.2f) {
                                    div = 0.2f;
                                }
                                pinchzoomPng = false;
                                if((++touchCount)%3 == 0)
                                    myEventListener.PinchZoomEvent(div);
                                pinch++;

                            } else if (oldDist - newDist > 15) { // zoom out

                                oldDist = newDist;
                                div += (((newDist / oldDist) / 50) * 10);

                                if (div >= 10.0f) {
                                    div = 10.0f;
                                }
                                pinchzoomPng = false;
                                if((++touchCount)%3 == 0)
                                    myEventListener.PinchZoomEvent(div);
                                pinch++;
                            }
                        }
                    }
                    break;

                case MotionEvent.ACTION_UP:
                    mode = NONE;
                    Log.d("emitTag", "Event ended");
                    myEventListener.GetPng();
                    touchCount = 0;
                    break;

                case MotionEvent.ACTION_POINTER_UP:
                    mode = NONE;
                    break;

                case MotionEvent.ACTION_POINTER_DOWN:

                    oldDist = spacing(event);

                    oldVectorX1 = event.getX(0);
                    oldVectorY1 = event.getY(0);
                    oldVectorX2 = event.getX(1);
                    oldVectorY2 = event.getY(1);

                    oldMidVectorX = midPoint(oldVectorX1, oldVectorX2);
                    oldMidVectorY = midPoint(oldVectorY1, oldVectorY2);

                    break;

                case MotionEvent.ACTION_CANCEL:
                    break;


                default:
                    break;
            }
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

//    //@Override
//    public void setView() {


    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        if(otf_table != null)
            Log.d("Onclick", "otf_table.getHeight()1 : " + otf_table.getHeight());
        Log.d("Onclick", "onWindowFocusChanged ");

        Log.d("Onclick", "onWindowFocusChanged ");
        super.onWindowFocusChanged(hasFocus);
    }
}

class TouchSurfaceView extends GLSurfaceView {

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        super.surfaceDestroyed(holder);
        Log.d("bmp", "surfaceDestoryed");
    }


    //public CudaRenderer mRenderer;
    private JniGLActivity mActivity;
    private Context mContext;
    private String host;


    public TouchSurfaceView(Context context, String host) {
        super(context);
        this.host = host;
        this.mContext = context;
//        this.mActivity = (JniGLActivity) context;
//        mRenderer = new CudaRenderer(mActivity, host);
//        setRenderer(mRenderer);

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

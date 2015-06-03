package com.nornenjs.android;

import android.app.Activity;
import android.content.res.Resources;
import android.os.Handler;
import android.os.Message;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.SeekBar;

/**
 * Created by hyok on 15. 6. 2.
 */
public class ChangeView {

    private ConvertDisplay display;
    private JniGLActivity jniActivity;
    private MprActivity mprActivity;

    public ChangeView(JniGLActivity activity) {
        jniActivity = activity;
        display = new ConvertDisplay(activity);
    }

    public ChangeView(MprActivity activity) {
        mprActivity = activity;
        display = new ConvertDisplay(activity);
    }



    public void setView() {
        if(!jniActivity.mGLSurfaceView.isShown())
        {
            Log.d("", "setView() called");
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

    final Handler handler;

    {
        handler = new Handler() {

            public void handleMessage(Message msg)

            {

                Log.d("", "handleMessage() called");

                final RelativeLayout newContainer = new RelativeLayout(jniActivity);//FrameLayout

                RelativeLayout.LayoutParams layoutParams = new RelativeLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT);

                newContainer.setLayoutParams(layoutParams);

                jniActivity.setContentView(R.layout.toggle);
                jniActivity.togglebtn = (Button) jniActivity.findViewById(R.id.toggleVol);
                jniActivity.togglebtn.setWidth(40);//(int) display.GetDipsFromPixel(40)
                jniActivity.togglebtn.invalidate();
                jniActivity.togglebtn.setHeight(40);//(int) display.GetDipsFromPixel(40)
                jniActivity.togglebtn.invalidate();


                jniActivity.toggleMip = (Button) jniActivity.findViewById(R.id.toggleMip);
                jniActivity.toggleMenu = (Button) jniActivity.findViewById(R.id.toggleMenu);
                jniActivity.positionX = (Button) jniActivity.findViewById(R.id.positionX);
                jniActivity.positionY = (Button) jniActivity.findViewById(R.id.positionY);
                jniActivity.positionZ = (Button) jniActivity.findViewById(R.id.positionZ);

                RelativeLayout.LayoutParams layoutParams1 = new RelativeLayout.LayoutParams((int) display.GetDipsFromPixel(40), (int) display.GetDipsFromPixel(40));
                RelativeLayout.LayoutParams layoutParams2 = new RelativeLayout.LayoutParams((int) display.GetDipsFromPixel(40), (int) display.GetDipsFromPixel(40));
                RelativeLayout.LayoutParams layoutParams3 = new RelativeLayout.LayoutParams((int) display.GetDipsFromPixel(40), (int) display.GetDipsFromPixel(40));
                RelativeLayout.LayoutParams layoutParams4 = new RelativeLayout.LayoutParams((int) display.GetDipsFromPixel(40), (int) display.GetDipsFromPixel(40));
                RelativeLayout.LayoutParams layoutParams5 = new RelativeLayout.LayoutParams((int) display.GetDipsFromPixel(40), (int) display.GetDipsFromPixel(40));

                jniActivity.togglebtn.setLayoutParams (new RelativeLayout.LayoutParams((int) display.GetDipsFromPixel(40), (int) display.GetDipsFromPixel(40)));

                layoutParams1.addRule(RelativeLayout.RIGHT_OF, R.id.toggleVol);
                jniActivity.toggleMip.setLayoutParams(layoutParams1);

                layoutParams2.addRule(RelativeLayout.RIGHT_OF, R.id.toggleMip);
                jniActivity.toggleMenu.setLayoutParams(layoutParams2);

                layoutParams3.addRule(RelativeLayout.RIGHT_OF, R.id.toggleMenu);
                jniActivity.positionX.setLayoutParams(layoutParams3);

                layoutParams4.addRule(RelativeLayout.RIGHT_OF, R.id.positionX);
                jniActivity.positionY.setLayoutParams(layoutParams4);

                layoutParams5.addRule(RelativeLayout.RIGHT_OF, R.id.positionY);
                jniActivity.positionZ.setLayoutParams(layoutParams5);


                jniActivity.togglebtn.setOnClickListener(jniActivity.mRenderer);
                jniActivity.toggleMip.setOnClickListener(jniActivity.mRenderer);
                jniActivity.toggleMenu.setOnClickListener(jniActivity.mRenderer);
                jniActivity.positionX.setOnClickListener(jniActivity.mRenderer);
                jniActivity.positionY.setOnClickListener(jniActivity.mRenderer);
                jniActivity.positionZ.setOnClickListener(jniActivity.mRenderer);

                ViewParent parent = jniActivity.togglebtn.getParent();
                ViewGroup group = (ViewGroup) parent;
                try {
                    group.addView(jniActivity.mGLSurfaceView);
                }catch(IllegalStateException e)
                {
                    e.printStackTrace();
                }

                jniActivity.otf_table = (RelativeLayout) jniActivity.findViewById(R.id.otf_table);
                //여기여기
                RelativeLayout.LayoutParams layoutparams = new RelativeLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, (int) display.GetDipsFromPixel(160));
                layoutparams.addRule(RelativeLayout.ALIGN_PARENT_BOTTOM);
                jniActivity.otf_table.setLayoutParams(layoutparams);

                DrawActivity drawView;
                drawView = (DrawActivity) jniActivity.findViewById(R.id.canvas);
                drawView.otf_width = drawView.getWidth();
                drawView.otf_height = drawView.getHeight();
                drawView.jniGLActivity = jniActivity;

                SeekBar sb = (SeekBar) jniActivity.findViewById(R.id.brightseek);
                sb.setProgress(200);
                sb.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                    @Override
                    public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                        jniActivity.myEventListener.BrightnessEvent(progress);
                    }

                    @Override
                    public void onStartTrackingTouch(SeekBar seekBar) {

                    }

                    @Override
                    public void onStopTrackingTouch(SeekBar seekBar) {
                        jniActivity.myEventListener.GetPng();

                    }
                });


                jniActivity.mrpSb = (SeekBar) jniActivity.findViewById(R.id.player_seek_horizontal);
                jniActivity.mrpSb.setProgress(50);
                jniActivity.mrpSb.setOnSeekBarChangeListener(jniActivity.mRenderer);
                jniActivity.mrpSb.setVisibility(View.INVISIBLE);
                jniActivity.mrpSb.bringToFront();
                jniActivity.mrpSb.invalidate();

                jniActivity.togglebtn.bringToFront();
                jniActivity.togglebtn.invalidate();
                jniActivity.toggleMip.bringToFront();
                jniActivity.toggleMip.invalidate();
                jniActivity.toggleMenu.bringToFront();
                jniActivity.toggleMenu.invalidate();
                jniActivity.positionX.bringToFront();
                jniActivity.positionX.invalidate();
                jniActivity.positionY.bringToFront();
                jniActivity.positionY.invalidate();
                jniActivity.positionZ.bringToFront();
                jniActivity.positionZ.invalidate();

                jniActivity.otf_table.bringToFront();
                jniActivity.otf_table.invalidate();

                jniActivity.setContentView(group);

//            Log.d("Onclick", "otf_table.getHeight()1 : " + otf_table.getHeight());
//
//            otf_table.setTranslationY(otf_table.getHeight());
//            Log.d("Onclick", "otf_table.getHeight()2 : " + otf_table.getHeight());

            }

        };
    }

    public void setMprView() {
        if(!mprActivity.mGLSurfaceView.isShown())
        {
            Log.d("", "mprActivity.() called");
            new Thread()

            {

                public void run()

                {

                    Message msg = mprhandler.obtainMessage();

                    mprhandler.sendMessage(msg);

                }

            }.start();
        }

    }

    private float convertPixelsToDp(float px){
        //해상도별로 해도 잘 안됨. 폰마다 굵기가 다르게 나옴.
        Resources resources = jniActivity.getResources();
        DisplayMetrics metrics = resources.getDisplayMetrics();
        Log.d("", "metrics.densityDpi : "+ metrics.densityDpi);//내꺼 480일때, 성근 320
        float dp = px * (metrics.densityDpi / 160f);
        return dp;
    }
    final Handler mprhandler = new Handler()
    {

        public void handleMessage(Message msg)

        {
            Log.d("", "handleMessage() called");



            mprActivity.setContentView(R.layout.activity_mpr);

            mprActivity.sb = (SeekBar) mprActivity.findViewById(R.id.player_seek_horizontal);
            mprActivity.sb.setProgress(50);
            mprActivity.sb.setOnSeekBarChangeListener(mprActivity.mRenderer);

            final RelativeLayout newContainer = new RelativeLayout(mprActivity);//FrameLayout

            RelativeLayout.LayoutParams layoutParams = new RelativeLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT );

            newContainer.setLayoutParams(layoutParams);

            ViewParent parent = mprActivity.sb.getParent();
            ViewGroup group = (ViewGroup)parent;
            group.addView(mprActivity.mGLSurfaceView);

            mprActivity.sb.bringToFront();
            mprActivity.sb.invalidate();


        }

    };
}

package com.nornenjs.android;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceHolder;

/**
 * Created by hyok on 15. 6. 4.
 */
public class TouchSurfaceView extends GLSurfaceView {

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
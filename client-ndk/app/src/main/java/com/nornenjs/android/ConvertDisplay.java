package com.nornenjs.android;

import android.app.Activity;
import android.content.Context;
import android.os.Build;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;

import java.lang.reflect.Method;

/**
 * Created by hyok on 15. 6. 2.
 */
public class ConvertDisplay {

    private Activity activity;
    private static final double RATIO = (170.0 / 1920.0);


    public ConvertDisplay(Activity activity) {
        this.activity = activity;
    }

    public double GetDipsFromPixel(float pixels)
    {

        Display display = activity.getWindowManager().getDefaultDisplay();//context.getWindowManager().getDefaultDisplay();
        int realWidth;
        int realHeight;

        if (Build.VERSION.SDK_INT >= 17){
            DisplayMetrics realMetrics = new DisplayMetrics();
            display.getRealMetrics(realMetrics);
            realWidth = realMetrics.widthPixels;
            realHeight = realMetrics.heightPixels;

        } else if (Build.VERSION.SDK_INT >= 14) {
            try {
                Method mGetRawH = Display.class.getMethod("getRawHeight");
                Method mGetRawW = Display.class.getMethod("getRawWidth");
                realWidth = (Integer) mGetRawW.invoke(display);
                realHeight = (Integer) mGetRawH.invoke(display);

            } catch (Exception e) {
                realWidth = display.getWidth();
                realHeight = display.getHeight();
                Log.e("Display Info", "Couldn't use reflection to get the real display metrics.");
            }

        } else {

            realWidth = display.getWidth();
            realHeight = display.getHeight();
        }
        double ratio = pixels / 1920.0;

        Log.d("ConvertDisplay class", "ratio : " + ratio +", calced : " + 3 * realHeight * ratio);
        return 3 * realHeight * ratio;//(int)(pixels * (realHeight / 0.0885));
    }
}

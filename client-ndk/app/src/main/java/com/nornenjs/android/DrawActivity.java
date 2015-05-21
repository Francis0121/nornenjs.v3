package com.nornenjs.android;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.*;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import com.fasterxml.jackson.databind.deser.std.DateDeserializers;

import java.util.jar.Attributes;


public class DrawActivity extends View{

    private final String TAG="DrawActivity";
    private final Path circle = new Path();

    float xx = 50, yy = 50;
    Point p1 ,p2, p3, p4;

    private final Paint cPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint cPaintEraser = new Paint(Paint.ANTI_ALIAS_FLAG);

    public DrawActivity(Context context, AttributeSet att) {
        super(context, att);
        circle.addCircle(xx, yy, 50, Path.Direction.CW);//위치와 크기

        cPaint.setStyle(Paint.Style.FILL);
        cPaint.setColor(Color.DKGRAY);
        cPaint.setStrokeWidth(3);
        cPaintEraser.setStyle(Paint.Style.FILL);
        cPaintEraser.setColor(Color.WHITE);
        cPaintEraser.setStrokeWidth(3);


        Point p1 = new Point(50,50);
        Point p2 = new Point(150,50);
        Point p3 = new Point(50,150);
        Point p4 = new Point(150,150);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);


        //canvas.drawColor(Color.WHITE, PorterDuff.Mode.CLEAR);

        circle.reset();
        circle.addCircle(p1.x, p1.y, p1.radius, Path.Direction.CW);

        canvas.drawPath(circle, cPaint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {

        Log.d(TAG, "onTouchEvent on DrawActivity");

        if(event.getAction() == MotionEvent.ACTION_DOWN)
            return true;
        else if(event.getAction() == MotionEvent.ACTION_MOVE) {
            invalidate();
            xx = event.getX();
            yy = event.getY();
            //circle.addCircle(event.getX(), event.getY(), 50, Path.Direction.CW);//위치와 크기
            invalidate();
        }

//        circle.moveTo(event.getX(), event.getY());
//        invalidate();
        //if(event.getY() == circle.moveTo(event.getX(), event.getY());)
        return super.onTouchEvent(event);

    }

    class Line
    {
        Point start;
        Point end;

    }

    class Point
    {
        float x,y;
        int radius;

        public Point(float x, float y) {
            this.x = x;
            this.y = y;
            radius = 10;
        }

        public float getX() {
            return x;
        }

        public float getY() {
            return y;
        }

        public int getRadius() {
            return radius;
        }

        public void setX(float x) {
            this.x = x;
        }

        public void setY(float y) {
            this.y = y;
        }

    }

}

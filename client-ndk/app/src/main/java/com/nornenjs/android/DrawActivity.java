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
    private final Path figure = new Path();
    private final Path bg = new Path();
    private final Path line = new Path();

    public float tr_x = 350, tr_y = 100, tl_x = 700, tl_y =100, br_x = 800, br_y, bl_x = 250, bl_y;
    //tr_y, tl_y 고정

    public float otf_width, otf_height;

    Point bottomLeft ,bottomRight, topRight, topLeft;
    boolean b_Left ,b_Right, t_Right, t_Left;

    private final Paint cPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint bg_Paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint bg_LinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    DashPathEffect dashPath = new DashPathEffect(new float[]{5,5}, 2);

    public DrawActivity(Context context, AttributeSet att) {
        super(context, att);

        cPaint.setStyle(Paint.Style.FILL);
        cPaint.setColor(Color.DKGRAY);
        cPaint.setStrokeWidth(3);

        bg_Paint.setStyle(Paint.Style.STROKE);
        bg_Paint.setPathEffect(dashPath);
        bg_Paint.setStrokeWidth(3);

        bg_LinePaint.setStyle(Paint.Style.STROKE);
        bg_LinePaint.setColor(Color.BLACK);
        bg_LinePaint.setStrokeWidth(30);

        topLeft = new Point(tr_x, tr_y);//기본값...이 기본값은 유지, 재사용이되야함
        topRight = new Point(tl_x,tl_y);
        bottomLeft = new Point(bl_x, bl_y);
        bottomRight = new Point(br_x, br_y);

        figure.addCircle(topLeft.x, topLeft.y, topLeft.radius, Path.Direction.CW);
        figure.addCircle(topRight.x, topRight.y, topRight.radius, Path.Direction.CW);
        figure.addCircle(bottomLeft.x, bottomLeft.y, bottomLeft.radius, Path.Direction.CW);
        figure.addCircle(bottomRight.x, bottomRight.y, bottomRight.radius, Path.Direction.CW);//점 4개




    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        Log.d("onDraw", "canvas onDraw called");
        drawBackground();

        //canvas.drawColor(Color.WHITE, PorterDuff.Mode.CLEAR);
        figure.reset();
        figure.addCircle(topLeft.x, topLeft.y, topLeft.radius, Path.Direction.CW);
        figure.addCircle(topRight.x, topRight.y, topRight.radius, Path.Direction.CW);
        figure.addCircle(bottomLeft.x, bottomLeft.y, bottomLeft.radius, Path.Direction.CW);
        figure.addCircle(bottomRight.x, bottomRight.y, bottomRight.radius, Path.Direction.CW);
        figure.close();
        canvas.drawPath(figure, cPaint);//원



        canvas.drawPath(bg, bg_Paint);
        canvas.drawPath(line, bg_LinePaint);
        line.moveTo(50, otf_height - 100);//아래 선
        line.lineTo(otf_width - 70, otf_height - 100);//margin 70
        bg_LinePaint.setColor(Color.BLACK);
        canvas.drawPath(line, bg_LinePaint);

        drawTrapezoid(topLeft, topRight, bottomLeft, bottomRight);

        bg_LinePaint.setColor(Color.LTGRAY);
//        cPaint.setAntiAlias(true) ;
        canvas.drawPath(figure, bg_LinePaint);//원
    }

    public void drawTrapezoid(Point topLeft, Point topRight, Point bottomLeft, Point bottomRight)
    {
//        //위의 두가지로 사각형을 그리고, 삼각형 두개를 그려 채운다.
//        figure.addRect(topLeft.x, topLeft.y, topRight.x, bottomRight.y, Path.Direction.CW);
//
//        figure.moveTo(topLeft.x, topLeft.y);
//        figure.lineTo(topLeft.x, bottomLeft.y);
//        figure.lineTo(bottomLeft.x, bottomLeft.y);//왼쪽 삼각형
//
//        figure.moveTo(topRight.x, topRight.y);
//        figure.lineTo(topRight.x, bottomRight.y);
//        figure.lineTo(bottomRight.x, bottomRight.y);//오른쪽 삼각형

        //채우지 말구 선으로만...
        //위의 두가지로 사각형을 그리고, 삼각형 두개를 그려 채운다.
        //figure.addRect(topLeft.x, topLeft.y, topRight.x, bottomRight.y, Path.Direction.CW);

        figure.moveTo(topLeft.x, topLeft.y);
        figure.lineTo(bottomLeft.x, bottomLeft.y);//왼쪽 선
        figure.moveTo(topLeft.x, topLeft.y);
        figure.lineTo(topRight.x, topRight.y);//상단 선

        figure.moveTo(topRight.x, topRight.y);
        figure.lineTo(bottomRight.x, bottomRight.y);//오른쪽 삼각형


    }


    public void drawBackground()
    {
        View layoutMainView = (View)this.findViewById(R.id.canvas);

        otf_width = layoutMainView.getWidth();
        otf_height = layoutMainView.getHeight();

        bg.moveTo(70, 100);//점선
        bg.lineTo(otf_width - 70, 100);//vertical margin 100, horizontal margin 70

        line.moveTo(100, 30);
        line.lineTo(100, otf_height - 30);//vertical margin 70, horizontal margin 100

        bottomLeft.setX(250);//동그라미 밑에 두개의 위치를 지정
        bottomLeft.setY(otf_height - 100);
        bottomRight.setX(800);
        bottomRight.setY(otf_height - 100);

    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {

        Log.d(TAG, "onTouchEvent on DrawActivity");

        if(event.getAction() == MotionEvent.ACTION_DOWN) {

            if(topLeft.checkPoint(event.getX(), event.getY()))
            {

                t_Left = true;
            }
            else if(topRight.checkPoint(event.getX(), event.getY()))
            {

                t_Right = true;
            }
            else if(bottomLeft.checkPoint(event.getX(), event.getY()))
            {

                b_Left = true;
            }
            else if(bottomRight.checkPoint(event.getX(), event.getY()))
            {

                b_Right = true;
            }
            return true;
        }else if(event.getAction() == MotionEvent.ACTION_MOVE) {


            if(t_Left)
            {
                topLeft.setX(event.getX());
                //topLeft.setY(event.getY());
            }
            else if(t_Right)
            {
                topRight.setX(event.getX());
                //topRight.setY(event.getY());
            }
            else if(b_Left)
            {
                //조건문
                //if(event.getX() <= topLeft.x)
                    bottomLeft.setX(event.getX());
                //bottomLeft.setY(event.getY());
            }
            else if(b_Right)
            {
                //조건문
                //if(event.getX() >= topRight.x)
                    bottomRight.setX(event.getX());
                //bottomRight.setY(event.getY());
            }

            invalidate();
        }
        else if(event.getAction() == MotionEvent.ACTION_UP)
        {
            if(t_Left)
            {
                t_Left = false;
            }
            else if(t_Right)
            {
                t_Right = false;
            }
            else if(b_Left)
            {
                b_Left = false;
            }
            else if(b_Right)
            {
                b_Right = false;
            }
        }
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
            radius = 25;
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

        public boolean checkPoint(float x, float y){
            if((x - (this.x + this.radius)) * (x - (this.x + this.radius)) + (y - (this.y + this.radius)) * (y - (this.y + this.radius)) <= this.radius * this.radius)
            {
                return true;
            }

            return false;
        }

    }

}

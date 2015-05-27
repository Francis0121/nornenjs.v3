package com.nornenjs.android;


import android.content.Context;
import android.content.res.Resources;
import android.graphics.*;
import android.util.AttributeSet;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import com.github.nkzawa.socketio.client.Socket;

public class DrawActivity extends View{

    private static final Integer TRANSFER_MAX_VALUE = 255;

    private static final Integer TRANSFER_START = 65;
    private static final Integer TRANSFER_MID1 = 80;
    private static final Integer TRANSFER_MID2 = 100;
    private static final Integer TRANSFER_END = 120;

    private static final Integer TOP_MARGIN_VALUE = 30;
    private static final Integer LEFT_MARGIN_VALUE = 50;
    private static final Integer WIDTH_DIFF = 80;

    private final String TAG="DrawActivity";
    private final Path figure = new Path();
    private final Path bg = new Path();
    private final Path line = new Path();

    private Socket socket;

    public float tr_x, tr_y, tl_x, tl_y, br_x, br_y, bl_x, bl_y;

    public Integer otf_width, otf_height;
    public Integer otf_start = WIDTH_DIFF, otf_end;
    public Integer otfHeightStart, otfHeightEnd;

    Point bottomLeft ,bottomRight, topRight, topLeft;
    boolean b_Left ,b_Right, t_Right, t_Left;

    Line left, top, right;
    boolean left_line, top_line, right_line;

    private final Paint cPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint bg_Paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint bg_LinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    DashPathEffect dashPath = new DashPathEffect(new float[]{5,5}, 2);

    JniGLActivity jniGLActivity;
    private Context mContext;

    public DrawActivity(Context context) {
        super(context);
        this.mContext = context;
    }

    public DrawActivity(Context context, AttributeSet att) {
        super(context, att);
        this.mContext = context;
    }

    private void setInitRelativePosition(Integer otfStart, Integer otfEnd){

        cPaint.setStyle(Paint.Style.FILL);
        cPaint.setColor(Color.DKGRAY);
        cPaint.setStrokeWidth(convertPixelsToDp(3));

        bg_Paint.setStyle(Paint.Style.STROKE);
        bg_Paint.setPathEffect(dashPath);
        bg_Paint.setStrokeWidth(convertPixelsToDp(1));//3

        bg_LinePaint.setStyle(Paint.Style.STROKE);
        bg_LinePaint.setColor(Color.BLACK);
        bg_LinePaint.setStrokeWidth(convertPixelsToDp(10));//30


        // ~ TODO 해당 수식이 맞는지 확인 필요
        Double diffDistance = new Double(otfEnd - otfStart);
        Double onePoint = diffDistance / new Double(TRANSFER_MAX_VALUE);

        tl_x = Double.valueOf(onePoint * TRANSFER_MID1).intValue();
        tr_x = Double.valueOf(onePoint * TRANSFER_MID2).intValue();
        bl_x = Double.valueOf(onePoint * TRANSFER_START).intValue();
        br_x = Double.valueOf(onePoint * TRANSFER_END).intValue();

        topLeft = new Point(tl_x, tl_y);
        topRight = new Point(tr_x, tr_y);
        bottomLeft = new Point(bl_x, bl_y);
        bottomRight = new Point(br_x, br_y);

        left = new Line(topLeft, bottomLeft);
        top = new Line(topLeft, topRight);
        right = new Line(topRight, bottomRight);

        figure.addCircle(topLeft.getX(), topLeft.getY(), convertPixelsToDp(topLeft.getRadius()), Path.Direction.CW);
        figure.addCircle(topRight.getX(), topRight.getY(), convertPixelsToDp(topRight.getRadius()), Path.Direction.CW);
        figure.addCircle(bottomLeft.getX(), bottomLeft.getY(), convertPixelsToDp(bottomLeft.getRadius()), Path.Direction.CW);
        figure.addCircle(bottomRight.getX(), bottomRight.getY(), convertPixelsToDp(bottomRight.getRadius()), Path.Direction.CW);//점 4개
    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if(otf_width == 0 || otf_width == 0.0f)
        {
            Log.d("onDraw", "otf_width is 0");
            View layoutMainView = (View)this.findViewById(R.id.canvas);

            otf_width = layoutMainView.getWidth();
            otf_height = layoutMainView.getHeight();
            otf_end = otf_width - WIDTH_DIFF;

            setInitRelativePosition(otf_start, otf_end);

            otfHeightStart = TOP_MARGIN_VALUE + TOP_MARGIN_VALUE;
            otfHeightEnd = otf_height - TOP_MARGIN_VALUE - TOP_MARGIN_VALUE ;

            topLeft.setY(otfHeightStart);
            topRight.setY(otfHeightStart);
            bottomLeft.setY(otfHeightEnd);
            bottomRight.setY(otfHeightEnd);
        }

        drawBackground(canvas);
    }

    private float convertPixelsToDp(float px){
        //해상도별로 해도 잘 안됨. 폰마다 굵기가 다르게 나옴.
        Resources resources = mContext.getResources();
        DisplayMetrics metrics = resources.getDisplayMetrics();
        Log.d(TAG, "metrics.densityDpi : "+ metrics.densityDpi);//내꺼 480일때, 성근 320
        float dp = px * (metrics.densityDpi / 160f);
        return dp;
    }

    public void drawBackground(Canvas canvas) {
        //점선
        canvas.drawLine(otf_start-LEFT_MARGIN_VALUE+20, otfHeightStart, otf_end+LEFT_MARGIN_VALUE, otfHeightStart, bg_Paint);

        //기준선 2개
        bg_LinePaint.setColor(Color.BLACK);
        bg_LinePaint.setStrokeWidth(convertPixelsToDp(5));//15
        canvas.drawLine(otf_start, otfHeightStart - TOP_MARGIN_VALUE, otf_start, otfHeightEnd + TOP_MARGIN_VALUE, bg_LinePaint); //세로
        canvas.drawLine(otf_start-LEFT_MARGIN_VALUE+20, otfHeightEnd, otf_end+LEFT_MARGIN_VALUE, otfHeightEnd, bg_LinePaint);//가로

        //사다리꼴 3개 라인
        bg_LinePaint.setColor(Color.LTGRAY);
        bg_LinePaint.setStrokeWidth(convertPixelsToDp(6));//30
        canvas.drawLine(topLeft.x, topLeft.y, bottomLeft.x, bottomLeft.y, bg_LinePaint);
        canvas.drawLine(topLeft.x, topLeft.y, topRight.x, topRight.y, bg_LinePaint);
        canvas.drawLine(topRight.x, topRight.y, bottomRight.x, bottomRight.y, bg_LinePaint);

        //꼭지점 4개

        canvas.drawCircle(topLeft.getX(), topLeft.getY(), convertPixelsToDp(topLeft.getRadius()), cPaint);
        canvas.drawCircle(topRight.getX(), topRight.getY(), convertPixelsToDp(topRight.getRadius()), cPaint);
        canvas.drawCircle(bottomLeft.getX(), bottomLeft.getY(), convertPixelsToDp(bottomLeft.getRadius()), cPaint);
        canvas.drawCircle(bottomRight.getX(), bottomRight.getY(), convertPixelsToDp(bottomRight.getRadius()), cPaint);

        Log.d(TAG, "otf TopLeft : " + calc(topLeft.x));
        Log.d(TAG, "otf TopRight : " + calc(topRight.x));
        Log.d(TAG, "otf bottomLeft : " + calc(bottomLeft.x));
        Log.d(TAG, "otf bottomRight " + calc(bottomRight.x));
    }

    float beforeX;
    boolean flag1 = false, flag2 = false;
    int eventCount;
    @Override
    public boolean onTouchEvent(MotionEvent event) {

        if(event.getAction() == MotionEvent.ACTION_DOWN) {
            eventCount = 0;

            Log.d(TAG,"topLeft.x : " + topLeft.x + "event.getX() : " + event.getX());
            if(topLeft.checkPoint(event.getX(), event.getY())) {
                Log.d(TAG,"topLeft true");
                t_Left = true;
            } else if(topRight.checkPoint(event.getX(), event.getY())) {
                t_Right = true;
            } else if(bottomLeft.checkPoint(event.getX(), event.getY())) {
                b_Left = true;
            } else if(bottomRight.checkPoint(event.getX(), event.getY())) {
                b_Right = true;
            } else if(left.IsOnLine(event.getX(), event.getY())) {
                beforeX = event.getX();
                left_line = true;
                Log.d("IsOnLine","left line clicked");
            } else if(right.IsOnLine(event.getX(), event.getY())) {
                beforeX = event.getX();
                right_line = true;
                Log.d("IsOnLine","right line clicked");
            } else if(top.IsOnLine(event.getX(), event.getY())) {
                beforeX = event.getX();
                top_line = true;
                Log.d("IsOnLine","top line clicked");
            }
            return true;
        }else if(event.getAction() == MotionEvent.ACTION_MOVE) {

            if(t_Left) {
                if(event.getX() >= bottomLeft.x && event.getX() >= otf_start && event.getX() <= topRight.x)
                    topLeft.setX(event.getX());
            } else if(t_Right) {
                if(event.getX() <= bottomRight.x && event.getX() <= otf_end && event.getX() >= topLeft.x)
                    topRight.setX(event.getX());
            } else if(b_Left) {
                if(event.getX() <= topLeft.x && event.getX() >= otf_start)
                    bottomLeft.setX(event.getX());
            } else if(b_Right) {
                Log.d(TAG, "otf_end : " + otf_end);
                if(event.getX() >= topRight.x && event.getX() <= otf_end)
                    bottomRight.setX(event.getX());
            } else if(left_line) {

                if(bottomLeft.x >= otf_start) {

                    if(bottomLeft.getX() + (-1)*(beforeX - event.getX()) < otf_start + 5) {
                        bottomLeft.setX(otf_start + 5);
                        topLeft.setX(otf_start + 5 + topLeft.getX() - bottomLeft.getX());
                    } else {
                        if(topLeft.getX() + (-1)*(beforeX - event.getX()) <= topRight.x)
                        {
                            bottomLeft.setX(bottomLeft.getX() + (-1)*(beforeX - event.getX()));
                            topLeft.setX(topLeft.getX() + (-1)*(beforeX - event.getX()));
                        }
                    }
                    beforeX = event.getX();
                }

            } else if(top_line) {
                Log.d(TAG, "bottomLeft.x : " + bottomLeft.x + ", otf_start : " + otf_start);
                if(bottomLeft.x >= otf_start && bottomRight.x <= otf_end) {
                    if(!flag2) {
                        if(bottomLeft.getX() + (-1)*(beforeX - event.getX()) < otf_start + 5) {
                            flag1 = true;
                            bottomLeft.setX(otf_start + 5);
                            topLeft.setX(otf_start + 5 + topLeft.getX() - bottomLeft.getX());
                        } else {
                            flag1 = false;
                            bottomLeft.setX(bottomLeft.getX() + (-1)*(beforeX - event.getX()));
                            topLeft.setX(topLeft.getX() + (-1)*(beforeX - event.getX()));
                        }
                    }

                    if(!flag1) {
                        if(bottomRight.getX() + (-1)*(beforeX - event.getX()) > otf_end - 5) {
                            flag2 = true;
                            bottomRight.setX(otf_end - 5);
                            topRight.setX(otf_end - 5 + topRight.getX() - bottomRight.getX());
                        } else {
                            flag2 = false;
                            bottomRight.setX(bottomRight.getX() + (-1)*(beforeX - event.getX()));
                            topRight.setX(topRight.getX() + (-1)*(beforeX - event.getX()));
                        }
                    }

                    beforeX = event.getX();
                }
            } else if(right_line) {
                if(bottomRight.x <= otf_end) {

                    if(bottomRight.getX() + (-1)*(beforeX - event.getX()) > otf_end - 5) {
                        bottomRight.setX(otf_end - 5);
                        topRight.setX(otf_end - 5 + topRight.getX() - bottomRight.getX());//padding 5
                    } else {
                        if(topRight.getX() + (-1)*(beforeX - event.getX()) >= topLeft.x)
                        {
                            bottomRight.setX(bottomRight.getX() + (-1)*(beforeX - event.getX()));
                            topRight.setX(topRight.getX() + (-1)*(beforeX - event.getX()));
                        }
                    }
                    beforeX = event.getX();
                }
            }
            //여기서 네개의 좌표 보내기.
            try{
                if(++eventCount%3 == 0)
                    jniGLActivity.myEventListener.OtfEvent(calc(bottomLeft.x), calc(topLeft.x), calc(topRight.x), calc(bottomRight.x), 1);
            }
            catch(NullPointerException e)
            {
                e.printStackTrace();
            }

            Log.d(TAG, "otf TopLeft : " + calc(topLeft.x));
            Log.d(TAG, "otf TopRight : " + calc(topRight.x));
            Log.d(TAG, "otf bottomLeft : " + calc(bottomLeft.x));
            Log.d(TAG, "otf bottomRight " + calc(bottomRight.x));
            invalidate();
        } else if(event.getAction() == MotionEvent.ACTION_UP) {
            if(t_Left) {
                t_Left = false;
            } else if(t_Right) {
                t_Right = false;
            } else if(b_Left) {
                b_Left = false;
            } else if(b_Right) {
                b_Right = false;
            } else if(left_line) {
                left_line = false;
                beforeX = 0;
            } else if(top_line) {
                top_line = false;
                beforeX = 0;
            } else if(right_line) {
                right_line = false;
                beforeX = 0;
            }
            jniGLActivity.myEventListener.OtfEvent(calc(bottomLeft.x), calc(topLeft.x), calc(topRight.x), calc(bottomRight.x), 2);
        }
        return super.onTouchEvent(event);

    }

    private int calc(float temp) {
        int value = (int)(((temp - (otf_start+5)) * 255)/((otf_end-5)-(otf_start+5)));
        return value < 0 ? 0 : value;
    }

    class Line
    {
        Point start;
        Point end;

        public Line(Point start, Point end) {
            this.start = start;
            this.end = end;
        }

        public boolean IsOnLine(float checkPointX, float checkPointY)
        {
            float bet_startcheck = (float)Math.sqrt(Math.pow(Math.abs(start.x - checkPointX), 2) + Math.pow(Math.abs(start.y - checkPointY), 2));//distance a
            float bet_startend = (float)Math.sqrt(Math.pow(Math.abs(start.x - end.x), 2) + Math.pow(Math.abs(start.y - end.y), 2));// distance c
            float bet_endcheck = (float)Math.sqrt(Math.pow(Math.abs(end.x - checkPointX), 2) + Math.pow(Math.abs(end.y - checkPointY), 2));//distance b

            double calc = (Math.pow(bet_endcheck, 2) + Math.pow(bet_startend,2) - Math.pow(bet_startcheck,2))/(2*bet_startend*bet_endcheck);
            if(Math.acos(calc) < 0.15)
                    return true;

            return false;
        }

    }

    class Point
    {
        private float x;
        private float y;
        private int radius;

        public Point(float x, float y) {
            this.x = x;
            this.y = y;
            this.radius = 8;//25
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
            if((x - (this.x + this.radius)) * (x - (this.x + this.radius)) + (y - (this.y + this.radius)) * (y - (this.y + this.radius)) <= (this.radius+10) * (this.radius+10))
            {
                return true;
            }

            return false;
        }

    }

}

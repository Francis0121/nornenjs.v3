package com.nornenjs.android;

/**
 * Created by hyok on 15. 5. 4.
 */

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Outline;
import android.os.Build;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.*;
import android.widget.*;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.nornenjs.android.ConvertDisplay;

public class ImageAdapter extends BaseAdapter implements View.OnClickListener{

    private static final String TAG = "ImageAdapter";

    //private static final double RATIO = (170.0 / 1920.0);

    private List<String> titles1;
    private List<Bitmap> thumbnails1;
    private List<Integer> pns1;
    private List<String> date1;
    private List<String> metadata1;
    private Map<Integer, Integer> pnMap;

    private Activity activity;
    private VolumeList volumelist_Page;

    //private Context mContext;

//    public ImageAdapter(Context c) {
//        mContext = c;
//    }
    private ConvertDisplay display;

    public ImageAdapter(List<String> titles1, List<Bitmap> thumbnails1, List<Integer> pns1, List<String> date1, List<String> metadata1, Activity activity) {

        Log.d(TAG, "ImageAdapter called");

        Log.d("ImageAdapter", "thumbnails1.size : " + thumbnails1.size());
        this.titles1 = titles1;
        this.thumbnails1 = thumbnails1;
        this.pns1 = pns1;
        this.date1 = date1;
        this.metadata1 = metadata1;
        this.activity = activity;
        volumelist_Page = new VolumeList();

        display = new ConvertDisplay(activity);
    }


    public int getCount() {
        Log.d("getCount", "thumbnails1.size : " + thumbnails1.size());

        return thumbnails1.size();
    }

    public Object getItem(int position) {
        return null;
    }

    public long getItemId(int position) {
        return 0;
    }

    public static class ViewHolder// extends ViewOutlineProvider
    {
        public ImageView imgViewFlag;
        public TextView title;
        public TextView date;
        public TextView metadata;

    }

    public ViewHolder view;

    //@Override
    public View getView(int position, View convertView, ViewGroup parent) {
        // TODO Auto-generated method stub
        LayoutInflater inflator = activity.getLayoutInflater();
        int pos = position;// * 2;
        if(convertView==null)//list의 끝이면....
        {
            Log.d(TAG + " getView","null");
            view = new ViewHolder();
            convertView = inflator.inflate(R.layout.activity_grid_row, null);

            view.title = (TextView) convertView.findViewById(R.id.textView1);
            view.date = (TextView) convertView.findViewById(R.id.textView2);
            view.metadata = (TextView) convertView.findViewById(R.id.textView3);
            view.imgViewFlag = (ImageView) convertView.findViewById(R.id.imageView1);
            view.metadata.setOnClickListener(this);
            view.imgViewFlag.setOnClickListener(this);

            convertView.setTag(view);
        }
        else
        {
            Log.d(TAG, " not null");
            view = (ViewHolder) convertView.getTag();
        }


        Log.d(TAG + " getView","position : " + position);

        //여기서 문자열 처
        String titleText;
        if(titles1.get(pos).length() > 10)
            titleText = titles1.get(pos).substring(0, 10) + "...";
        else
            titleText = titles1.get(pos);


        view.imgViewFlag.setTag(pns1.get(pos));

        final Bitmap bitmap = volumelist_Page.getBitmapFromMemCache(""+pns1.get(pos));//+pns1.get(position)
        if(bitmap != null){
            view.imgViewFlag.setImageBitmap(bitmap);
            //Log.d(TAG, "cached image use");
        }
        else
            view.imgViewFlag.setImageBitmap(thumbnails1.get(pos));

        view.imgViewFlag.setScaleType(ImageView.ScaleType.CENTER_CROP);

        view.imgViewFlag.setLayoutParams(new RelativeLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, (int) display.GetDipsFromPixel(170)));

        Log.d(TAG, "width : " + view.imgViewFlag.getWidth());
        Log.d(TAG, "height : " + view.imgViewFlag.getHeight());

        view.title.setText(titleText);
        view.date.setText(date1.get(pos));
        view.metadata.setText(metadata1.get(pos));


        convertView.setOnTouchListener(new View.OnTouchListener() {
            public boolean onTouch(View v, MotionEvent event) {
                return true;
            }
        });

        return convertView;
    }

    public void onClick(View v) {

        if(v.getClass().getName().equalsIgnoreCase("android.widget.ImageView")){
            Log.d("onClick", "click image id : " + v.getTag());
            Intent intent = new Intent(activity, PreviewActivity.class);
            intent.putExtra("pns", Integer.parseInt(""+v.getTag()));
            activity.startActivity(intent);
        }
    }

}
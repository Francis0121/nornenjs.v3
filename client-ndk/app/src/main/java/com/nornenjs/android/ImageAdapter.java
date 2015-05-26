package com.nornenjs.android;

/**
 * Created by hyok on 15. 5. 4.
 */

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ImageAdapter extends BaseAdapter implements View.OnClickListener{

    private static final String TAG = "ImageAdapter";


    private List<String> titles1;
    private List<String> titles2;
    private List<Bitmap> thumbnails1;
    private List<Bitmap> thumbnails2;
    private List<Integer> pns1;
    private List<Integer> pns2;
    private List<String> date1;
    private List<String> date2;
    private List<String> metadata1;
    private List<String> metadata2;
    private Map<Integer, Integer> pnMap;

    private Activity activity;
    private VolumeList volumelist_Page;

//    // Constructor
//    public ImageAdapter(Context c) {
//        mContext = c;
//    }

    public ImageAdapter(List<String> titles1, List<String> titles2, List<Bitmap> thumbnails1, List<Bitmap> thumbnails2, List<Integer> pns1, List<Integer> pns2, List<String> date1, List<String> date2, List<String> metadata1, List<String> metadata2, Activity activity) {

        Log.d(TAG, "ImageAdapter called");
        this.titles1 = titles1;
        this.titles2 = titles2;
        this.thumbnails1 = thumbnails1;
        this.thumbnails2 = thumbnails2;
        this.pns1 = pns1;
        this.pns2 = pns2;
        this.date1 = date1;
        this.date2 = date2;
        this.metadata1 = metadata1;
        this.metadata2 = metadata2;
        this.activity = activity;
        volumelist_Page = new VolumeList();
    }


//    public ImageAdapter(List<String> titles, List<String> titles2, List<Bitmap> thumbnails, List<Bitmap> thumbnails2, List<Integer> pns, List<Integer> pns2, Activity activity) {
//        this.titles1 = titles;
//        this.titles2 = titles2;
//        this.thumbnails1 = thumbnails;
//        this.thumbnails2 = thumbnails2;
//        this.pns1 = pns;
//        this.pns2 = pns2;
//        this.activity = activity;
//        this.pnMap = new HashMap<Integer, Integer>();
//    }

    public int getCount() {
        return thumbnails1.size();//더 짧은 오른쪽을 기준
        //return titles.size();
    }

    public Object getItem(int position) {
        return null;
        //return thumbnails1.get(position);
    }

    public long getItemId(int position) {
        return 0;
    }

    public static class ViewHolder
    {
        public ImageView imgViewFlag;
        public TextView title;
        public TextView date;
        public TextView metadata;
        public ImageView imgViewFlag2;
        public TextView title2;
        public TextView date2;
        public TextView metadata2;
    }


    //@Override
    public View getView(int position, View convertView, ViewGroup parent) {
        // TODO Auto-generated method stub
        ViewHolder view;
        LayoutInflater inflator = activity.getLayoutInflater();
        if(convertView==null)
        {
            view = new ViewHolder();
            convertView = inflator.inflate(R.layout.activity_grid_row, null);

            view.title = (TextView) convertView.findViewById(R.id.textView1);
            view.date = (TextView) convertView.findViewById(R.id.textView2);
            view.metadata = (TextView) convertView.findViewById(R.id.textView3);
            view.imgViewFlag = (ImageView) convertView.findViewById(R.id.imageView1);
            view.metadata.setOnClickListener(this);
            view.imgViewFlag.setOnClickListener(this);


            //Log.d(TAG + " getView", "thumbnails2.size() : " + thumbnails2.size() + ", position : " + position);
            view.title2 = (TextView) convertView.findViewById(R.id.textView2_1);
            view.date2 = (TextView) convertView.findViewById(R.id.textView2_2);
            view.metadata2 = (TextView) convertView.findViewById(R.id.textView2_3);
            view.imgViewFlag2 = (ImageView) convertView.findViewById(R.id.imageView2);
            view.metadata2.setOnClickListener(this);
            view.imgViewFlag2.setOnClickListener(this);
            convertView.setTag(view);
        }
        else
        {
            view = (ViewHolder) convertView.getTag();
        }

        Log.d(TAG + " getView","position : " + position);

        //여기서 문자열 처
        String titleText;
        if(titles1.get(position).length() > 10)
            titleText = titles1.get(position).substring(0, 10) + "...";
        else
            titleText = titles1.get(position);

        view.title.setText(titleText);
        view.date.setText(date1.get(position));
        view.metadata.setText(metadata1.get(position));

        view.imgViewFlag.setTag(pns1.get(position));

        if(thumbnails1.size() > position)
        {
            final Bitmap bitmap = volumelist_Page.getBitmapFromMemCache(""+pns1.get(position));//+pns1.get(position)
            if(bitmap != null){
                view.imgViewFlag.setImageBitmap(bitmap);
                //Log.d(TAG, "cached image use");
            }
            else
                view.imgViewFlag.setImageBitmap(thumbnails1.get(position));
        }
//try{~~~~
//        if(titles2.get(position).length() > 10)
//            titleText = titles2.get(position).substring(0, 10) + "...";
//        else
//            titleText = titles2.get(position);
//
//        view.title2.setText(titleText);
//        view.date2.setText(date2.get(position));
//        view.metadata2.setText(metadata2.get(position));
//
//        view.imgViewFlag2.setTag(pns2.get(position));

        //Log.d(TAG + " set", "thumbnails2.size() : " + thumbnails2.size() + ", position : " + position);
        if(thumbnails2.size() > position)
        {

            if(titles2.get(position).length() > 10)
                titleText = titles2.get(position).substring(0, 10) + "...";
            else
                titleText = titles2.get(position);


            view.title2.setText(titleText);
            view.date2.setText(date2.get(position));
            view.metadata2.setText(metadata2.get(position));

            view.imgViewFlag2.setTag(pns2.get(position));

            final Bitmap bitmap = volumelist_Page.getBitmapFromMemCache(""+pns2.get(position));//+pns1.get(position)
            if(bitmap != null){
                view.imgViewFlag2.setImageBitmap(bitmap);
                //Log.d(TAG, "cached image use");
            }
            else
                view.imgViewFlag2.setImageBitmap(thumbnails2.get(position));

        }
        else
        {
//            view.title2.setVisibility(View.GONE);
//            view.date2.setVisibility(View.GONE);
//            view.metadata2.setVisibility(View.GONE);
//            view.imgViewFlag2.setVisibility(View.GONE);
            Log.d(TAG, "thumbnails2.size() <= position");
        }
        convertView.setOnTouchListener(new View.OnTouchListener() {
            public boolean onTouch(View v, MotionEvent event) {
                return true;
            }
        });

        return convertView;
    }



    public void onClick(View v) {

        Log.d("onClick", "click image id : " + v.getTag());
 //       Log.d("ImageAdapter", "GET PN ID "+pnMap.get(v.getId()));
        Intent intent = new Intent(activity, PreviewActivity.class);
        intent.putExtra("pns", Integer.parseInt(""+v.getTag()));
        activity.startActivity(intent);
    }

}
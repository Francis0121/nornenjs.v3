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
    private Map<Integer, Integer> pnMap;

    private Activity activity;
    private VolumeList volumelist_Page;

//    // Constructor
//    public ImageAdapter(Context c) {
//        mContext = c;
//    }


    public ImageAdapter(List<String> titles, List<String> titles2, List<Bitmap> thumbnails, List<Bitmap> thumbnails2, List<Integer> pns, List<Integer> pns2, Activity activity) {
        this.titles1 = titles;
        this.titles2 = titles2;
        this.thumbnails1 = thumbnails;
        this.thumbnails2 = thumbnails2;
        this.pns1 = pns;
        this.pns2 = pns2;
        this.activity = activity;
        this.pnMap = new HashMap<Integer, Integer>();
    }

    public int getCount() {
        return thumbnails1.size();//더 짧은 오른쪽을 기준
        //return titles.size();
    }

    public Object getItem(int position) {
        return thumbnails1.get(position);
        //return titles.get(position);
    }

    public long getItemId(int position) {
        return 0;
    }

    public static class ViewHolder
    {
        public ImageView imgViewFlag;
        public TextView metadata;
        public TextView date;
        public ImageView imgViewFlag2;
        public TextView metadata2;
        public TextView date2;
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

            view.metadata = (TextView) convertView.findViewById(R.id.textView1);
            view.imgViewFlag = (ImageView) convertView.findViewById(R.id.imageView1);
            view.metadata2 = (TextView) convertView.findViewById(R.id.textView2_1);
            view.imgViewFlag2 = (ImageView) convertView.findViewById(R.id.imageView2);

            view.metadata.setOnClickListener(this);
            view.imgViewFlag.setOnClickListener(this);
            view.metadata2.setOnClickListener(this);
            view.imgViewFlag2.setOnClickListener(this);

            convertView.setTag(view);
        }
        else
        {
            view = (ViewHolder) convertView.getTag();
        }

        view.metadata.setText(titles1.get(position));
        view.imgViewFlag.setTag(pns1.get(position));

        if(thumbnails1.size() > position)
        {
            view.imgViewFlag.setImageBitmap(thumbnails1.get(position));
        }

        view.metadata2.setText(titles2.get(position));
        view.imgViewFlag2.setTag(pns2.get(position));

        if(thumbnails2.size() > position) {
            view.imgViewFlag2.setImageBitmap(thumbnails2.get(position));
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
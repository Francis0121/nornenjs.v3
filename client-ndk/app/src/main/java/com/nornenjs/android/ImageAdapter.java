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
    private List<Bitmap> thumbnails1;
    private List<Integer> pns1;
    private List<String> date1;
    private List<String> metadata1;
    private Map<Integer, Integer> pnMap;

    private Activity activity;
    private VolumeList volumelist_Page;

//    // Constructor
//    public ImageAdapter(Context c) {
//        mContext = c;
//    }

    public ImageAdapter(List<String> titles1, List<Bitmap> thumbnails1, List<Integer> pns1, List<String> date1, List<String> metadata1, Activity activity) {

        Log.d(TAG, "ImageAdapter called");
        this.titles1 = titles1;
        this.thumbnails1 = thumbnails1;
        this.pns1 = pns1;
        this.date1 = date1;
        this.metadata1 = metadata1;
        this.activity = activity;
        volumelist_Page = new VolumeList();
    }


    public int getCount() {
        if(thumbnails1.size() % 2 == 0)
            return thumbnails1.size() / 2;
        else
            return (thumbnails1.size() / 2)+1;
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


        int pos = position * 2;
        //여기서 문자열 처
        String titleText;
        if(titles1.get(pos).length() > 10)
            titleText = titles1.get(pos).substring(0, 10) + "...";
        else
            titleText = titles1.get(pos);

        view.imgViewFlag.setTag(pns1.get(pos));

        if(thumbnails1.size() > pos)
        {
            final Bitmap bitmap = volumelist_Page.getBitmapFromMemCache(""+pns1.get(pos));//+pns1.get(position)
            if(bitmap != null){
                view.imgViewFlag.setImageBitmap(bitmap);
                //Log.d(TAG, "cached image use");
            }
            else
                view.imgViewFlag.setImageBitmap(thumbnails1.get(pos));
        }

        view.title.setText(titleText);
        view.date.setText(date1.get(pos));
        view.metadata.setText(metadata1.get(pos));

        //------------------------left side-------------------

        try
        {

            if(thumbnails1.size() > pos+1)
            {
                if(titles1.get(pos+1).length() > 10)
                    titleText = titles1.get(pos+1).substring(0, 10) + "...";
                else
                    titleText = titles1.get(pos+1);

                view.imgViewFlag2.setTag(pns1.get(pos + 1));

                final Bitmap bitmap = volumelist_Page.getBitmapFromMemCache(""+pns1.get(pos+1));//+pns1.get(position)
                if(bitmap != null){
                    view.imgViewFlag2.setImageBitmap(bitmap);
                    //Log.d(TAG, "cached image use");
                }
                else
                    view.imgViewFlag2.setImageBitmap(thumbnails1.get(pos+1));

                view.title2.setText(titleText);
                view.date2.setText(date1.get(pos+1));
                view.metadata2.setText(metadata1.get(pos+1));
            }
            else
            {
                Log.d(TAG + " getView()","pos + 1 : " + (pos + 1) + ", thumbnails1.size() : " + thumbnails1.size());
//                view.title2.setVisibility(View.GONE);
//                view.date2.setVisibility(View.GONE);
//                view.metadata2.setVisibility(View.GONE);
//                view.imgViewFlag2.setVisibility(View.GONE);
            }

        }catch(IndexOutOfBoundsException e)
        {
            e.printStackTrace();
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
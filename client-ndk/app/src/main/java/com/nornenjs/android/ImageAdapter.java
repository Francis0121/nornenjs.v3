package com.nornenjs.android;

/**
 * Created by hyok on 15. 5. 4.
 */

import android.app.Activity;
import android.content.Intent;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.ArrayList;

public class ImageAdapter extends BaseAdapter implements View.OnClickListener{
//    private Context mContext;

    private ArrayList<String> listImage;
    private ArrayList<Integer> listFlag;
    private Activity activity;

    private VolumeList volumelist_Page;
//    // Constructor
//    public ImageAdapter(Context c) {
//        mContext = c;
//    }
    public ImageAdapter(Activity activity,ArrayList<String> listImage, ArrayList<Integer> listFlag) {
        super();
        this.listImage = listImage;
        this.listFlag = listFlag;
        this.activity = activity;
    }

    public int getCount() {
        return listImage.size();
//        return mThumbIds.length;
    }

    public Object getItem(int position) {
        return listImage.get(position);
        //return null;
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
            //view.txtViewTitle.setBackground(R.drawable.bold);
            convertView = inflator.inflate(R.layout.activity_grid_row, null);

            //convertView.setLayoutParams(new GridView.LayoutParams(convertView.getLayoutParams().width,convertView.getLayoutParams().height));
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

        view.metadata.setText(listImage.get(position));
        view.imgViewFlag.setImageResource(listFlag.get(position));

        view.metadata2.setText(listImage.get(position));
        view.imgViewFlag2.setImageResource(listFlag.get(position));

        convertView.setOnTouchListener(new View.OnTouchListener() {
            public boolean onTouch(View v, MotionEvent event) {
                return true;
            }
        });

        return convertView;
    }



    public void onClick(View v) {
        Intent intent = new Intent(activity, PreviewActivity.class);
        activity.startActivity(intent);
    }
}
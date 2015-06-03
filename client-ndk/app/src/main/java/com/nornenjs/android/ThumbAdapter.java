package com.nornenjs.android;

/**
 * Created by hyok on 15. 5. 4.
 */

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.util.AttributeSet;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ThumbAdapter extends BaseAdapter{

    private static final String TAG = "ThumbAdapter";

    List<Bitmap> thumbnails;
    String text;
    private Activity activity;

    public ThumbAdapter(List<Bitmap> thumbnails, Activity activity) {
        this.thumbnails = thumbnails;
        this.activity = activity;
        Log.d(TAG, "ThumbAdapter created");
    }

    @Override
    public int getCount() {
        return thumbnails.size()+1;//+1했음
    }

    @Override
    public Object getItem(int position) {

        //return thumbnails.get(position);
        return null;
    }

    @Override
    public long getItemId(int position) {
        return 0;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {

        ViewHolder view;

        view = new ViewHolder();
        LayoutInflater inflator = activity.getLayoutInflater();

        convertView = inflator.inflate(R.layout.activity_thumb_row, null);


        if(position == 0)
        {
            view.thumbnailText = (TextView) convertView.findViewById(R.id.thumbText);
            view.thumbnailText.setVisibility(View.VISIBLE);
            view.thumbnailText.setText(text);
            view.thumbnailImage = (SquareImageView) convertView.findViewById(R.id.thumb);
            view.thumbnailImage.setVisibility(View.GONE);
        }
        else
        {
            view.thumbnailImage = (SquareImageView) convertView.findViewById(R.id.thumb);
            view.thumbnailImage.setImageBitmap(thumbnails.get(position-1));
            view.thumbnailImage.getLayoutParams().height = view.thumbnailImage.getLayoutParams().width;
            convertView.setTag(view);

        }
        convertView.setTag(view);
        return convertView;
    }

    public static class ViewHolder
    {
        public TextView thumbnailText;
        public SquareImageView thumbnailImage;

    }

}


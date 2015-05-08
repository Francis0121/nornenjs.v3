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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ImageAdapter extends BaseAdapter implements View.OnClickListener{

    private static final String TAG = "ImageAdapter";

//    private Context mContext;

    private List<String> titles;
    private List<Integer> thumbnails;
    private List<Integer> pns;
    private Map<Integer, Integer> pnMap;

    private Activity activity;
    private VolumeList volumelist_Page;

//    // Constructor
//    public ImageAdapter(Context c) {
//        mContext = c;
//    }

    public ImageAdapter(Activity activity, List<String> titles, List<Integer> thumbnails, List<Integer> pns) {
        super();
        this.titles = titles;
        this.thumbnails = thumbnails;
        this.pns = pns;
        this.activity = activity;
        this.pnMap = new HashMap<Integer, Integer>();
    }

    public int getCount() {
        return titles.size();
//        return mThumbIds.length;
    }

    public Object getItem(int position) {
        return titles.get(position);
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

        view.metadata.setText(titles.get(position));
        view.imgViewFlag.setImageResource(thumbnails.get(position));
        view.imgViewFlag.setTag(pns.get(position));

        view.metadata2.setText(titles.get(position));
        view.imgViewFlag2.setImageResource(thumbnails.get(position));
        view.imgViewFlag2.setTag(pns.get(position));

        Log.d(TAG, "" + view.imgViewFlag);
        Log.d(TAG, "" + view.imgViewFlag.getId());
        //pnMap.put(view.imgViewFlag.getId(), pns.get(position));

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
        activity.finish();
    }
}
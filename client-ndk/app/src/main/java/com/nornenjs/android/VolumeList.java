package com.nornenjs.android;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.AbsListView;
import android.widget.ListView;

import java.util.ArrayList;


public class VolumeList extends Activity {

    private ImageAdapter mAdapter;
    private ArrayList<String> listCountry;
    private ArrayList<Integer> listFlag;

    private ListView imagelist;

    private PoppyViewHelper mPoppyViewHelper;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_volume_list);

        prepareList();

        mAdapter = new ImageAdapter(this,listCountry, listFlag);

        // Set custom adapter to gridview
        imagelist = (ListView) findViewById(R.id.imagelist);
        imagelist.setAdapter(mAdapter);


        mPoppyViewHelper = new PoppyViewHelper(this);
        View poppyView = mPoppyViewHelper.createPoppyViewOnListView(R.id.searchbar, R.id.imagelist, R.layout.poppyview, new AbsListView.OnScrollListener() {
                    public void onScrollStateChanged(AbsListView view, int scrollState) {

                    }

                    public void onScroll(AbsListView view, int firstVisibleItem, int visibleItemCount, int totalItemCount) {

                    }
                });

//        imagelist.setOnItemClickListener(new AdapterView.OnItemClickListener() {
//            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
//
//                imagelist.setVisibility(View.INVISIBLE);
//                FragmentManager fm = getFragmentManager();
//                FragmentTransaction fragmentTransaction = fm.beginTransaction();
//                fragmentTransaction.replace(R.id.previewFragment, preview);
//                fragmentTransaction.commit();
//            }
//        });
    }


    public void openPreview()
    {
        Log.d("preview","before create intent");
        //Intent intent = new Intent(VolumeList.this, PreView.class);
        Log.d("preview","before startActivity");
        //startActivity(intent);
    }

    public void prepareList()
    {
        // 톨신으로 받으면 동적으로 넣어주면 됨.
        listCountry = new ArrayList<String>();

        listCountry.add("환자1");
        listCountry.add("환자2");
        listCountry.add("환자3");
        listCountry.add("환자4");
        listCountry.add("환자5");
        listCountry.add("환자6");
        listCountry.add("환자7");
        listCountry.add("환자8");
        listCountry.add("환자9");
        listCountry.add("환자10");
        listCountry.add("환자11");
        listCountry.add("환자12");
        listCountry.add("환자13");
        listCountry.add("환자14");
        listCountry.add("환자15");
        listCountry.add("환자16");
        listCountry.add("환자17");
        listCountry.add("환자18");
        listCountry.add("환자19");

        listFlag = new ArrayList<Integer>();
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
        listFlag.add(R.drawable.head);
    }

//    @Override
//    public void onBackPressed() {
//        //super.onBackPressed();
//        //if(gridView.is)
//        imagelist.setVisibility(View.VISIBLE);
//        FragmentManager fm = getFragmentManager();
//        FragmentTransaction fragmentTransaction = fm.beginTransaction();
//        fragmentTransaction.remove(preview);
//        fragmentTransaction.commit();
//    }
}

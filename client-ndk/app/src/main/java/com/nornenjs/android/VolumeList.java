package com.nornenjs.android;

import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.util.LruCache;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.*;
import cn.pedant.SweetAlert.SweetAlertDialog;
import com.nornenjs.android.dto.ResponseVolume;
import com.nornenjs.android.dto.Volume;
import com.nornenjs.android.dto.VolumeFilter;
import com.nornenjs.android.dynamicview.PoppyViewHelper;
import com.nornenjs.android.util.Pagination;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;
import org.w3c.dom.Text;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.cocosw.bottomsheet.BottomSheet;

public class VolumeList extends Activity {

    private static String TAG="VolumeList";

    static LruCache<String, Bitmap> mMemoryCache;
    private VolumeFilter volumeFilter;

    private ImageAdapter mAdapter;
    private ImageAdapter searchAdapter;

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

    private List<String> backuptitles1;
    private List<String> backuptitles2;
    private List<Bitmap> backupthumbnails1;
    private List<Bitmap> backupthumbnails2;
    private List<Integer> backuppns1;
    private List<Integer> backuppns2;
    private List<String> backupdate1;
    private List<String> backupdate2;
    private List<String> backupmetadata1;
    private List<String> backupmetadata2;

    private ListView imagelist;

    private RelativeLayout progressBar;
    private TextView alert;

    private PoppyViewHelper mPoppyViewHelper;

    public static int CurrentPage = 1;
    public static int totalPage;

    private BottomSheet sheet;
    public boolean bottom = false;

    //View v;
    final int maxMemory = (int) (Runtime.getRuntime().maxMemory() / 1024);
    final int cacheSize = maxMemory / 8;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_volume_list);

        SharedPreferences pref = getSharedPreferences("userInfo", 0);
        volumeFilter = new VolumeFilter(pref.getString("username",""), "");

//        final int maxMemory = (int) (Runtime.getRuntime().maxMemory() / 1024);
//        final int cacheSize = maxMemory / 8;

        mMemoryCache = new LruCache<String, Bitmap>(cacheSize) {
            @Override
            protected int sizeOf(String key, Bitmap bitmap) {
                return bitmap.getByteCount() / 1024;
            }
        };
        Log.d(TAG,"create func");

//        if(mMemoryCache == null)
//            Log.d(TAG, "mMemoryCacheis null");


        titles1 = new ArrayList<String>();
        titles2 = new ArrayList<String>();
        thumbnails1 = new ArrayList<Bitmap>();
        thumbnails2 = new ArrayList<Bitmap>();
        pns1 = new ArrayList<Integer>();
        pns2 = new ArrayList<Integer>();

        backuptitles1 = new ArrayList<String>();
        backuptitles2 = new ArrayList<String>();
        backupthumbnails1 = new ArrayList<Bitmap>();
        backupthumbnails2 = new ArrayList<Bitmap>();
        backuppns1 = new ArrayList<Integer>();
        backuppns2 = new ArrayList<Integer>();


        date1 = new ArrayList<String>();
        date2 = new ArrayList<String>();
        metadata1 = new ArrayList<String>();
        metadata2 = new ArrayList<String>();

        backupdate1 = new ArrayList<String>();
        backupdate2 = new ArrayList<String>();
        backupmetadata1 = new ArrayList<String>();
        backupmetadata2 = new ArrayList<String>();

        mAdapter = new ImageAdapter(titles1, titles2, thumbnails1, thumbnails2, pns1, pns2, date1, date2, metadata1, metadata2, VolumeList.this);
        searchAdapter = new ImageAdapter(backuptitles1, backuptitles2, backupthumbnails1, backupthumbnails2, backuppns1, backuppns2, backupdate1, backupdate2, backupmetadata1, backupmetadata2, VolumeList.this);


        imagelist = (ListView) findViewById(R.id.imagelist);
        imagelist.setAdapter(mAdapter);


        progressBar = (RelativeLayout) findViewById(R.id.progress_layout);

        alert = (TextView) findViewById(R.id.alert);

        mPoppyViewHelper = new PoppyViewHelper(VolumeList.this);
        View poppyView = mPoppyViewHelper.createPoppyViewOnListView(R.id.searchbar, R.id.imagelist, R.layout.poppyview, new AbsListView.OnScrollListener() {
            public void onScrollStateChanged(AbsListView view, int scrollState) {

            }

            public void onScroll(AbsListView view, int firstVisibleItem, int visibleItemCount, int totalItemCount) {

            }
        });

        new PostVolumeTask().execute("none");

    }

    public void setView() {
        Log.d(TAG, "setView() called");
        imagelist.setVisibility(View.VISIBLE);
        progressBar.setVisibility(View.GONE);
    }

    public void searchRequest(String keyword) {
        Log.d(TAG, "searchRequest() called");
        imagelist.setVisibility(View.INVISIBLE);
        progressBar.setVisibility(View.VISIBLE);
        //alert.setVisibility(View.GONE);//??이거 뭐였더라? loding view가 안보이지 없어도되듯.
        new PostVolumeTask().execute("search", keyword);
        imagelist.setAdapter(searchAdapter);
    }

    public void getPage()
    {
        Log.d(TAG, "request next page.. CurrentPage : " + CurrentPage + "totalPage : " + totalPage);
        if(CurrentPage < totalPage)
        {
            new PostVolumeTask().execute("page");
        }

    }

    //memory caching--
    public void addBitmapToMemoryCache(String key, Bitmap bitmap) {
        if (mMemoryCache.get(key) == null) {
            Log.d(TAG, "func addBitmapToMemoryCache put");
            mMemoryCache.put(key, bitmap);
            //Log.d(TAG, "getBitmapFromMemCache get : " + key + ", mMemoryCache.get(key) : " + mMemoryCache.get(key));//여기선 출력 잘됨
        }
    }

    public Bitmap getBitmapFromMemCache(String key) {
        return mMemoryCache.get(key);
    }
    //---


    private class PostVolumeTask extends AsyncTask<String, Void, ResponseVolume> {


        @Override
        protected void onPreExecute() {
            super.onPreExecute();

        }
        String request;
        @Override
        protected ResponseVolume doInBackground(String... params) {

            request = params[0];

            Log.d(TAG, "request : " + request);
            try {
                // The URL for making the POST request
                String url = getString(R.string.tomcat) + "/mobile/volume/"+volumeFilter.getUsername()+"/list";


                // Create a new RestTemplate instance
                RestTemplate restTemplate = new RestTemplate();

                // Make the network request, posting the message, expecting a String in response from the server
                ResponseEntity<ResponseVolume> response;
                if("none".equals(params[0]))
                {//첫 요청
                    url+="/1";
                    response = restTemplate.postForEntity(url, volumeFilter, ResponseVolume.class);
                }
                else if("search".equals(params[0]))
                {//검색할때 요청
                    url+="/1";
                    volumeFilter.setTitle(params[1]);
                    response = restTemplate.postForEntity(url, volumeFilter, ResponseVolume.class);
                }else
                {//page 요청이면
                    Log.d(TAG,"page request");
                    url+="/" + (CurrentPage + 1);
                    //volumeFilter.setPage(2);//+1을 했는데도 증가하지 않고 계속 1값임.
                    response = restTemplate.postForEntity(url, volumeFilter, ResponseVolume.class);
                }

                // Return the response body to display to the user
                return response.getBody();

            } catch (Exception e) {
                Log.e(TAG, e.getMessage(), e);
            }

            return null;
        }

        @Override
        protected void onPostExecute(ResponseVolume responseVolume) {
            super.onPostExecute(responseVolume);
            List<Volume> volumes;
            if (responseVolume == null) {
                volumes = null;
                Log.d(TAG, "volumes is null");
            }else {
                volumes = responseVolume.getVolumes();

                Log.d(TAG, "volumes.toString()" + volumes.toString());
            }

            if("none".equals(request))
            {
                try
                {
                    Map<String, Object> volumeFilterMap = responseVolume.getVolumeFilter();
                    Log.d(TAG, "volumeFilterMap.toString()" + volumeFilterMap.toString());
                    Map<String, Integer> num = (Map<String, Integer>)volumeFilterMap.get("pagination");
                    totalPage = num.get("numPages");
                    Log.d(TAG, "numPages" + num.get("numPages"));
                    CurrentPage = num.get("requestedPage");
                }catch(NullPointerException e)
                {
                    //alert
                    SweetAlertDialog pDialog = new SweetAlertDialog(VolumeList.this, SweetAlertDialog.WARNING_TYPE);
                    pDialog.getProgressHelper().setBarColor(Color.parseColor("#A5DC86"));//A5DC86
                    pDialog.getProgressHelper().setRimColor(Color.parseColor("#33485c"));
                    pDialog.setTitleText("서버에 접속할 수 없습니다.");
                    pDialog.setCancelable(false);
                    pDialog.show();

                    View view = findViewById(R.id.blank);
                    progressBar.setVisibility(View.GONE);

                    pDialog.setOnDismissListener(new DialogInterface.OnDismissListener() {
                        @Override
                        public void onDismiss(DialogInterface dialog) {
                            finish();
                        }
                    });
                    e.printStackTrace();
                }
            }
            else if("page".equals(request))
            {
                Map<String, Object> volumeFilterMap = responseVolume.getVolumeFilter();
                Map<String, Integer> num = (Map<String, Integer>)volumeFilterMap.get("pagination");
                CurrentPage = num.get("requestedPage");
                Log.d(TAG, "num.get() : " + num.get("requestedPage"));
                Log.d(TAG, "currentPage : " + CurrentPage);
            }

            if(volumes == null)
            {//통신이 안된 경우
                //Toast.makeText(톧ㅅ녀ㅣ시ㅏ!나!ㅅ니시나사니시나사니시나사니시나ㅏthis, );
                Log.d(TAG, "통신이 안된 경우");
            }
            else
            {
                for(Volume volume : volumes)
                {
                    Log.d(TAG, volume.toString());

                    if("none".equals(request) || "page".equals(request))
                    {
                        if(titles1.size() == titles2.size())
                        {
                            titles1.add(volume.getTitle());
                            date1.add(volume.getInputDate().substring(0, 10));
                            metadata1.add(volume.getWidth().toString() +"x"+ volume.getHeight().toString() +"x"+ volume.getDepth().toString());
                            pns1.add(volume.getPn());
                        }
                        else if(titles1.size() > titles2.size())
                        {
                            titles2.add(volume.getTitle());
                            date2.add(volume.getInputDate().substring(0,10));
                            metadata2.add(volume.getWidth().toString() +"x"+ volume.getHeight().toString() +"x"+ volume.getDepth().toString());
                            pns2.add(volume.getPn());
                        }
                        new GetThumbnail().execute("" + volume.getThumbnailPnList().get(0),"none");
                    }
                    else if("search".equals(request))
                    {
                        if(backuptitles1.size() == backuptitles2.size())
                        {
                            backuptitles1.add(volume.getTitle());
                            backupdate1.add(volume.getInputDate().substring(0, 10));
                            backupmetadata1.add(volume.getWidth().toString() +"x"+ volume.getHeight().toString() +"x"+ volume.getDepth().toString());
                            backuppns1.add(volume.getPn());
                        }
                        else if(backuptitles1.size() > backuptitles2.size())
                        {
                            backuptitles2.add(volume.getTitle());
                            backupdate2.add(volume.getInputDate().substring(0,10));
                            backupmetadata2.add(volume.getWidth().toString() +"x"+ volume.getHeight().toString() + "x" + volume.getDepth().toString());
                            backuppns2.add(volume.getPn());
                        }
                        new GetThumbnail().execute("" + volume.getThumbnailPnList().get(0),"search");
                    }

                    Log.d(TAG, "volume.getThumbnailPnList().get(0) : " + volume.getThumbnailPnList().get(0));


                }

            }
        }


    }
    int count = 0;
    private class GetThumbnail extends AsyncTask<String, Void, Bitmap>{
        String request;

        @Override
        protected Bitmap doInBackground(String... params) {

            request = params[1];
            Log.d("one Image", "In doInBackground params0 : " + params[0]);
            Bitmap data = downloadImage(getString(R.string.tomcat) + "/data/thumbnail/" + params[0]);
            //pns
            //addBitmapToMemoryCache(String.valueOf(params[0]), data);
            Log.d("one Image", "after downloadImage(): ");

            getBitmapFromMemCache("call from VolumeActivity");
            return data;

        }

        @Override
        protected void onPostExecute(Bitmap bytes) {
            if(bytes != null)
            {
            //thumbnails.get(index).setImageBitmap(bytes);//image1.setImageBitmap(bytes);
                if("none".equals(request))
                {
                    Log.d(TAG, "get none thumbnails");
                    if(thumbnails1.size() == thumbnails2.size())
                    {
                        thumbnails1.add(bytes);//image1.setImageBitmap(bytes);
                        addBitmapToMemoryCache(String.valueOf(pns1.get(thumbnails1.size() - 1)), bytes);
                        //getBitmapFromMemCache("" + pns1.get(thumbnails1.size() - 1));//test
                        Log.d(TAG, "addBitmapToMemoryCache : " + pns1.get(thumbnails1.size() - 1));
                    }
                    else if(thumbnails1.size() > thumbnails2.size())
                    {
                        thumbnails2.add(bytes);//image1.setImageBitmap(bytes);
                        addBitmapToMemoryCache(String.valueOf(pns2.get(thumbnails2.size() - 1)), bytes);
                        //getBitmapFromMemCache(""+pns2.get(thumbnails2.size()-1));//test
                        Log.d(TAG, "addBitmapToMemoryCache" + pns2.get(thumbnails1.size()-1));
                    }

                    mAdapter.notifyDataSetChanged();
                    if(count == 10) {
                        count = 0;
                        bottom = false;
                    }
                    count++;
                    //bottom = false;
                }
                else if("search".equals(request))
                {
                    Log.d(TAG, "get search thumbnails");
                    if(backupthumbnails1.size() == backupthumbnails2.size())
                    {
                        backupthumbnails1.add(bytes);//image1.setImageBitmap(bytes);
                    }
                    else if(backupthumbnails1.size() > backupthumbnails2.size())
                    {
                        backupthumbnails2.add(bytes);//image1.setImageBitmap(bytes);
                    }
                    searchAdapter.notifyDataSetChanged();

                }


                if(!imagelist.isShown())
                    setView();

            }
            else
                Log.d("one Image", "bitmap is null");

            //mAdapter.notifyDataSetChanged();
        }
    }

    public Bitmap downloadImage(String imgName) {
        Log.d("one Image", "URI : " + imgName);
        //ByteArrayOutputStream baos = new ByteArrayOutputStream();
        Bitmap bitmap = null;
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            HttpURLConnection con = (HttpURLConnection) ( new URL(imgName)).openConnection();
            con.setDoInput(true);

            con.setRequestProperty("Accept-Encoding", "identity");
            con.connect();

            int responseCode = con.getResponseCode();
            Log.d("one Image", "responseCode : " + responseCode);
            Log.d("one Image", "getContentLength : " + con.getContentLength());

            InputStream is = con.getInputStream();

            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inSampleSize = 16;

//            bitmap = BitmapFactory.decodeStream(is);

            bitmap = BitmapFactory.decodeStream(is, null, options);

            con.disconnect();
        }
        catch(Throwable t) {
            t.printStackTrace();
        }
        return bitmap;
    }

    private void getData()
    {
        //
    }

    @Override
    public void onBackPressed() {
        if(imagelist.getAdapter().equals(searchAdapter))
        {
            imagelist.setAdapter(mAdapter);
            mAdapter.notifyDataSetChanged();
            mPoppyViewHelper.editView.setText("");
        }
        else if(imagelist.getAdapter().equals(mAdapter))
        {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        Log.d(TAG, "keyCode : " + keyCode);

        if(keyCode == 82)
        {
            //new BottomSheet.Builder(this).title("").sheet(R.menu.list).listener(new DialogInterface.OnClickListener() {
            new BottomSheet.Builder(this).sheet(R.menu.list).listener(new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    switch (which) {
                        case R.id.help:

                            break;
                        case R.id.logout:
                            SharedPreferences pref = getSharedPreferences("userInfo", 0);
                            SharedPreferences.Editor prefEdit = pref.edit();

                            prefEdit.putString("username", "");
                            Intent intent = new Intent(VolumeList.this, LoginActivity.class);

                            startActivity(intent);
                            finish();
                            break;
                    }
                }
            }).show();
        }

        return super.onKeyDown(keyCode, event);
    }
}

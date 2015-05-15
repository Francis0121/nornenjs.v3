package com.nornenjs.android;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.*;
import com.nornenjs.android.dto.ResponseVolume;
import com.nornenjs.android.dto.Volume;
import com.nornenjs.android.dto.VolumeFilter;
import com.nornenjs.android.dynamicview.PoppyViewHelper;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;
import org.w3c.dom.Text;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

//import com.cocosw.bottomsheet.*;

public class VolumeList extends Activity {

    private static String TAG="VolumeList";

    private VolumeFilter volumeFilter;

    private ImageAdapter mAdapter;
    private ImageAdapter searchAdapter;

    private List<String> titles1;
    private List<String> titles2;
    private List<Bitmap> thumbnails1;
    private List<Bitmap> thumbnails2;
    private List<Integer> pns1;
    private List<Integer> pns2;

    private List<String> backuptitles1;
    private List<String> backuptitles2;
    private List<Bitmap> backupthumbnails1;
    private List<Bitmap> backupthumbnails2;
    private List<Integer> backuppns1;
    private List<Integer> backuppns2;

    private ListView imagelist;

    private RelativeLayout progressBar;
    private TextView alert;

    private PoppyViewHelper mPoppyViewHelper;

    private View footer;
    private int CurrentPage = 1;
    private int totalPage;

    //private BottomSheet sheet;
    //View v;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_volume_list);

        SharedPreferences pref = getSharedPreferences("userInfo", 0);
        volumeFilter = new VolumeFilter(pref.getString("username",""), "");

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


        mAdapter = new ImageAdapter(titles1, titles2, thumbnails1, thumbnails2, pns1, pns2, VolumeList.this);
        searchAdapter = new ImageAdapter(backuptitles1, backuptitles2, backupthumbnails1, backupthumbnails2, backuppns1, backuppns2, VolumeList.this);

        // Set custom adapter to gridview
        imagelist = (ListView) findViewById(R.id.imagelist);
        imagelist.setAdapter(mAdapter);

        footer = getLayoutInflater().inflate(R.layout.footer, null, false);

        progressBar = (RelativeLayout) findViewById(R.id.progress_layout);

        alert = (TextView) findViewById(R.id.alert);

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
        alert.setVisibility(View.GONE);
        new PostVolumeTask().execute("search", keyword);
        imagelist.setAdapter(searchAdapter);
    }


    private class PostVolumeTask extends AsyncTask<String, Void, ResponseVolume> {


        @Override
        protected void onPreExecute() {
            super.onPreExecute();


        }
        String request;
        @Override
        protected ResponseVolume doInBackground(String... params) {

            request = params[0];
            try {
                // The URL for making the POST request
                final String url = getString(R.string.tomcat) + "/mobile/volume/"+volumeFilter.getUsername()+"/list";


                // Create a new RestTemplate instance
                RestTemplate restTemplate = new RestTemplate();

                // Make the network request, posting the message, expecting a String in response from the server
                ResponseEntity<ResponseVolume> response;
                if("none".equals(params[0]))
                {//첫 요청
                    response = restTemplate.postForEntity(url, volumeFilter, ResponseVolume.class);
                }
                else if("search".equals(params[0]))
                {//검색할때 요청
                    volumeFilter.setTitle(params[1]);
                    response = restTemplate.postForEntity(url, volumeFilter, ResponseVolume.class);
                }else
                {//page 요청이면
                    volumeFilter.setPage(CurrentPage + 1);//임의 값...
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
                Map<String, Object> volumeFilterMap = responseVolume.getVolumeFilter();
                Log.d(TAG, "volumeFilterMap.toString()" + volumeFilterMap.toString());
                Map<String, Integer> num = (Map<String, Integer>)volumeFilterMap.get("pagination");
                totalPage = num.get("numPages");
                Log.d(TAG, "numPages" + num.get("numPages"));
                CurrentPage = num.get("requestedPage");
            }
            else if("page".equals(request))
            {
                Map<String, Object> volumeFilterMap = responseVolume.getVolumeFilter();
                Map<String, Integer> num = (Map<String, Integer>)volumeFilterMap.get("pagination");
                CurrentPage = num.get("requestedPage");
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
                            titles1.add(volume.getTitle() + volume.getInputDate());
                            pns1.add(volume.getPn());
                        }
                        else if(titles1.size() > titles2.size())
                        {
                            titles2.add(volume.getTitle() + volume.getInputDate());
                            pns2.add(volume.getPn());
                        }
                        new GetThumbnail().execute("" + volume.getThumbnailPnList().get(0),"none");
                    }
                    else if("search".equals(request))
                    {
                        if(backuptitles1.size() == backuptitles2.size())
                        {
                            backuptitles1.add(volume.getTitle() + volume.getInputDate());
                            backuppns1.add(volume.getPn());
                        }
                        else if(backuptitles1.size() > backuptitles2.size())
                        {
                            backuptitles2.add(volume.getTitle() + volume.getInputDate());
                            backuppns2.add(volume.getPn());
                        }
                        new GetThumbnail().execute("" + volume.getThumbnailPnList().get(0),"search");
                    }

                    Log.d(TAG, "volume.getThumbnailPnList().get(0) : " + volume.getThumbnailPnList().get(0));
                    //thumbnails.add(R.drawable.head);

                }

                mPoppyViewHelper = new PoppyViewHelper(VolumeList.this);
                View poppyView = mPoppyViewHelper.createPoppyViewOnListView(R.id.searchbar, R.id.imagelist, R.layout.poppyview, new AbsListView.OnScrollListener() {
                    public void onScrollStateChanged(AbsListView view, int scrollState) {

                    }

                    public void onScroll(AbsListView view, int firstVisibleItem, int visibleItemCount, int totalItemCount) {

                    }
                });

            }
        }


    }
    private class GetThumbnail extends AsyncTask<String, Void, Bitmap>{
        int index;
        String request;
        @Override
        protected Bitmap doInBackground(String... params) {
            request = params[1];
            Log.d("one Image", "In doInBackground params0 : " + params[0]);
            Bitmap data = downloadImage(getString(R.string.tomcat) + "/data/thumbnail/" + params[0]);
            Log.d("one Image", "after downloadImage(): ");
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
                    }
                    else if(thumbnails1.size() > thumbnails2.size())
                    {
                        thumbnails2.add(bytes);//image1.setImageBitmap(bytes);
                    }


                    mAdapter.notifyDataSetChanged();
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
            bitmap = BitmapFactory.decodeStream(is);

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
//            sheet = new BottomSheet.Builder(this).darkTheme().title("To " + adapter.getItem(position)).sheet(R.menu.list).listener(new DialogInterface.OnClickListener() {
//                @Override
//                public void onClick(DialogInterface dialog, int which) {
//                    //ListAcitivty.this.onClick(adapter.getItem(position), which);
//                }
//            }).build();
        }
        return super.onKeyDown(keyCode, event);
    }
}

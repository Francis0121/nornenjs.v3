package com.nornenjs.android;

import android.app.Activity;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AbsListView;
import android.widget.Adapter;
import android.widget.ImageView;
import android.widget.ListView;
import com.nornenjs.android.dto.ResponseVolume;
import com.nornenjs.android.dto.Volume;
import com.nornenjs.android.dto.VolumeFilter;
import com.nornenjs.android.dynamicview.PoppyViewHelper;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class VolumeList extends Activity {

    private static String TAG="VolumeList";


    private VolumeFilter volumeFilter;

    private ImageAdapter mAdapter;
    private List<String> titles;
    //private List<Integer> thumbnails;
    private List<Bitmap> thumbnails;
    private List<Integer> pns;

    private ListView imagelist;

    private PoppyViewHelper mPoppyViewHelper;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_volume_list);

        SharedPreferences pref = getSharedPreferences("userInfo", 0);
        volumeFilter = new VolumeFilter(pref.getString("username",""), "");

        titles = new ArrayList<String>();
        thumbnails = new ArrayList<Bitmap>();
        pns = new ArrayList<Integer>();

//        thumbnails.add((ImageView) findViewById(R.id.imageView1));
//        thumbnails.add((ImageView) findViewById(R.id.imageView2));

        //new PostVolumeTask().execute();

        Log.d(TAG, "thumbnails size : " + thumbnails.size());
        Log.d(TAG, "titles size : " + +titles.size());

        mAdapter = new ImageAdapter(VolumeList.this , thumbnails, titles, pns);

        // Set custom adapter to gridview
        imagelist = (ListView) findViewById(R.id.imagelist);
        imagelist.setAdapter(mAdapter);

        new PostVolumeTask().execute();

    }



    private class PostVolumeTask extends AsyncTask<Void, Void, ResponseVolume> {


        @Override
        protected void onPreExecute() {
            super.onPreExecute();


        }

        @Override
        protected ResponseVolume doInBackground(Void... params) {

            try {
                // The URL for making the POST request
                final String url = getString(R.string.tomcat) + "/mobile/volume/"+volumeFilter.getUsername()+"/list";


                // Create a new RestTemplate instance
                RestTemplate restTemplate = new RestTemplate();

                // Make the network request, posting the message, expecting a String in response from the server
                ResponseEntity<ResponseVolume> response = restTemplate.postForEntity(url, volumeFilter, ResponseVolume.class);

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
                Log.d(TAG, volumes.toString());
            }
            Map<String, Object> volumeFilterMap = responseVolume.getVolumeFilter();
            Log.d(TAG, volumeFilterMap.toString());
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
                    titles.add(volume.getTitle() + volume.getInputDate());
                    //thumbnails.add(R.drawable.head);
                    Log.d(TAG, "volume.getThumbnailPnList().get(0) : " + volume.getThumbnailPnList().get(0));
                    //thumbnails.add(R.drawable.head);
                    new GetThumbnail().execute("" + volume.getThumbnailPnList().get(0));
                    pns.add(volume.getPn());
                    Log.d(TAG, "where am i?");
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
        @Override
        protected Bitmap doInBackground(String... params) {

            Log.d("one Image", "In doInBackground params0 : " + params[0]);
            Bitmap data = downloadImage(getString(R.string.tomcat) + "/data/thumbnail/" + params[0]);
            Log.d("one Image", "after downloadImage(): ");
            return data;

        }

        @Override
        protected void onPostExecute(Bitmap bytes) {
            if(bytes != null)//thumbnails.get(index).setImageBitmap(bytes);//image1.setImageBitmap(bytes);
                thumbnails.add(bytes);//image1.setImageBitmap(bytes);
            else
                Log.d("one Image", "bitmap is null");

            mAdapter.notifyDataSetChanged();
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

}

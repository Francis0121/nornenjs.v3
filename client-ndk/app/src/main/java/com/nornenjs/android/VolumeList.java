package com.nornenjs.android;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.AbsListView;
import android.widget.ListView;
import android.widget.Toast;
import com.nornenjs.android.dto.ResponseVolume;
import com.nornenjs.android.dto.Volume;
import com.nornenjs.android.dto.VolumeFilter;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


public class VolumeList extends Activity {

    private static String TAG="VolumeList";


    private VolumeFilter volumeFilter;

    private ImageAdapter mAdapter;
    private List<String> titles;
    private List<Integer> thumbnails;
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
        thumbnails = new ArrayList<Integer>();
        pns = new ArrayList<Integer>();

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

            List<Volume> volumes = responseVolume.getVolumes();
            Log.d(TAG,volumes.toString());

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
                    thumbnails.add(R.drawable.head);
                    pns.add(volume.getPn());
                }

                mAdapter = new ImageAdapter(VolumeList.this ,titles, thumbnails, pns);

                // Set custom adapter to gridview
                imagelist = (ListView) findViewById(R.id.imagelist);
                imagelist.setAdapter(mAdapter);


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

}

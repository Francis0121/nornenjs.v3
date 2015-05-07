package com.nornenjs.android;


import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import com.nornenjs.android.dto.*;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

import java.util.List;


public class PreviewActivity extends Activity {

    private VolumeFilter volumeFilter;
    int pns;

    Volume volumes;
    Data datas;

    Button click;

    private static final String TAG = "PreviewActivity";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_preview);

        click = (Button)findViewById(R.id.clickBtn);
        click.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent;
                intent = new Intent(PreviewActivity.this, JniGLActivity.class);

                intent.putExtra("width", volumes.getWidth());
                intent.putExtra("height", volumes.getHeight());
                intent.putExtra("depth", volumes.getDepth());

                intent.putExtra("savePath", datas.getSavePath());

                startActivity(intent);
                finish();
            }
        });
        Intent intent = getIntent();
        pns = intent.getIntExtra("pns",-1);
        if(pns != -1)
        {
            Log.d(TAG, "pns : "+pns);
        }else
        {
            Log.d(TAG, "pns us -1");
        }

        SharedPreferences pref = getSharedPreferences("userInfo", 0);
        volumeFilter = new VolumeFilter(pref.getString("username",""), "");

        new PostVolumeImageTask().execute();
    }

    private class PostVolumeImageTask extends AsyncTask<Void, Void, ResponseVolumeInfo> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected ResponseVolumeInfo doInBackground(Void... params) {
            try {
                // The URL for making the POST request
                final String url = getString(R.string.tomcat) + "/mobile/volume/"+volumeFilter.getUsername()+"/"+pns;


                // Create a new RestTemplate instance
                RestTemplate restTemplate = new RestTemplate();

                // Make the network request, posting the message, expecting a String in response from the server
                ResponseEntity<ResponseVolumeInfo> response = restTemplate.postForEntity(url, volumeFilter, ResponseVolumeInfo.class);

                // Return the response body to display to the user
                return response.getBody();

            } catch (Exception e) {
                Log.e(TAG, e.getMessage(), e);
            }

            return null;
        }

        @Override
        protected void onPostExecute(ResponseVolumeInfo responseVolume) {
            super.onPostExecute(responseVolume);


            volumes = responseVolume.getVolume();
            Log.d(TAG,volumes.toString());


            List<Integer> thumbnails = responseVolume.getThumbnails();
            Log.d(TAG,thumbnails.toString());


            datas = responseVolume.getData();
            Log.d(TAG,datas.toString());


            if(volumes == null)
            {//통신이 안된 경우
                //Toast.makeText(톧ㅅ녀ㅣ시ㅏ!나!ㅅ니시나사니시나사니시나사니시나ㅏthis, );
                Log.d(TAG, "통신이 안된 경우");
            }
            else
            {


            }
        }

    }
}

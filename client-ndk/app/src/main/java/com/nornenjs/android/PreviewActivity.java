package com.nornenjs.android;


import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import com.nornenjs.android.dto.*;
import org.apache.http.HttpResponse;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.DefaultHttpClient;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

import java.io.*;
import java.net.HttpURLConnection;

import java.net.URL;
import java.util.List;


public class PreviewActivity extends Activity {

    private static final String TAG = "PreviewActivity";

    private VolumeFilter volumeFilter;
    int pns;

    Volume volumes;
    Data datas;
    List<ImageView> thumbnails;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_preview);

        thumbnails.add((ImageView) findViewById(R.id.thumbnail1));
        thumbnails.add((ImageView) findViewById(R.id.thumbnail2));
        thumbnails.add((ImageView) findViewById(R.id.thumbnail3));
        thumbnails.add((ImageView) findViewById(R.id.thumbnail4));
//        click = (Button)findViewById(R.id.clickBtn);
//        click.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//
//                Intent intent;
//                intent = new Intent(PreviewActivity.this, JniGLActivity.class);
//
//                intent.putExtra("width", volumes.getWidth());
//                intent.putExtra("height", volumes.getHeight());
//                intent.putExtra("depth", volumes.getDepth());
//
//                intent.putExtra("savePath", datas.getSavePath());
//
//                startActivity(intent);
//                finish();
//            }
//        });
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
            Log.d(TAG, "volumes : " + volumes.toString());

            List<Integer> thumbnails = responseVolume.getThumbnails();
            Log.d(TAG, "thumbnails : " + thumbnails.toString());

            new GetThumbnails().execute("92");

            //image.setImageBitmap(getImageFromURL("http://localhost:10000/data/thumbnail/" + thumbnails.get(0)));
            Log.d(TAG, "after excute()");

            datas = responseVolume.getData();
            Log.d(TAG,"datas : " + datas.toString());

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


    private class GetThumbnails extends AsyncTask<String, Void, byte[]>{
        @Override
        protected byte[] doInBackground(String... params) {

            Log.d(TAG, "params0 : " + params[0]);
            byte[] data = downloadImage(getString(R.string.tomcat) + "/data/thumbnail/" + params[0]);

            return data;

        }

        @Override
        protected void onPostExecute(byte[] bytes) {
            Bitmap img = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
            thumbnails.get(0).setImageBitmap(img);
        }
    }

    public byte[] downloadImage(String imgName) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try {
            HttpURLConnection con = (HttpURLConnection) ( new URL(imgName)).openConnection();
            con.setDoInput(true);

            con.setRequestProperty("Accept-Encoding", "identity");
            con.connect();

            int responseCode = con.getResponseCode();

            InputStream is = con.getInputStream();
            byte[] b = new byte[1024];

            while ( is.read(b) != -1)
                baos.write(b);

            con.disconnect();
        }
        catch(Throwable t) {
            t.printStackTrace();
        }

        return baos.toByteArray();
    }

}

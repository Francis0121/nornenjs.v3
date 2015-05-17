package com.nornenjs.android;


import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.*;
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
import java.util.*;


public class PreviewActivity extends Activity {

    private static final String TAG = "PreviewActivity";

    private VolumeFilter volumeFilter;
    int pns;


    Volume volumes;
    Data datas;
    List<Bitmap> thumbnails;
    GridView gridview;
    ThumbAdapter thumbAdapter;

    int width, height, depth;
    String savepath = "";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_preview);

        Intent intent = getIntent();
        pns = intent.getIntExtra("pns",-1);
        if(pns != -1)
        {
            Log.d(TAG, "pns : "+pns);
        }else
        {
            Log.d(TAG, "pns us -1");
        }

        thumbnails = new ArrayList<Bitmap>();


        SharedPreferences pref = getSharedPreferences("userInfo", 0);
        volumeFilter = new VolumeFilter(pref.getString("username",""), "");


        thumbAdapter = new ThumbAdapter(thumbnails, PreviewActivity.this);
        gridview = (GridView) findViewById(R.id.previewlist);
        gridview.setAdapter(thumbAdapter);

        gridview.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {

                Log.d("emitTag", "emit position : " + position);
                Intent intent;
                if(position == 1)
                {
                    Log.d(TAG, "go to JNIActivity");
                    intent = new Intent(PreviewActivity.this, JniGLActivity.class);
                }
                else {
                    Log.d(TAG, "go to MprActivity");
                    intent = new Intent(PreviewActivity.this, MprActivity.class);
                }
                intent.putExtra("width", width);
                intent.putExtra("height", height);
                intent.putExtra("depth", depth);
                intent.putExtra("savepath", savepath);

                intent.putExtra("datatype", position-1);
                startActivity(intent);
            }
        });
        new PostVolumeImageTask().execute();

        Log.d(TAG, "thumbnails sie : " + thumbnails.size());

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

            width = responseVolume.getVolume().getWidth();
            height = responseVolume.getVolume().getHeight();
            depth = responseVolume.getVolume().getDepth();
            savepath = responseVolume.getData().getSavePath();
            Log.d(TAG, "getData : " + responseVolume.getData().toString());


            thumbAdapter.text = volumes.getTitle();

            new GetThumbnails().execute("" + thumbnails.get(0), "0");
            new GetThumbnails().execute("" + thumbnails.get(1), "1");
            new GetThumbnails().execute("" + thumbnails.get(2), "2");
            new GetThumbnails().execute("" + thumbnails.get(3), "3");

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
                //????여기 뭐였지???

            }
        }

    }

    private class GetThumbnails extends AsyncTask<String, Void, Bitmap>{
        int index;
        @Override
        protected Bitmap doInBackground(String... params) {

            Log.d(TAG, "params0 : " + params[0]);
            index = Integer.parseInt(params[1]);
            Bitmap data = downloadImage(getString(R.string.tomcat) + "/data/thumbnail/" + params[0]);

            return data;

        }

        @Override
        protected void onPostExecute(Bitmap bytes) {
            if(bytes != null)//thumbnails.get(index).setImageBitmap(bytes);//image1.setImageBitmap(bytes);
            {
                Log.d(TAG, "add bitmap");
                thumbnails.add(bytes);//image1.setImageBitmap(bytes);
            }
            else
                Log.d(TAG, "bitmap is null");

           //
            thumbAdapter.notifyDataSetChanged();
//            if(index == 3) {
//                Log.d(TAG, "notifyDataSetChanged.....thumbsize : " + thumbnails.size());
//                thumbAdapter.notifyDataSetChanged();
//            }
        }
    }

    public Bitmap downloadImage(String imgName) {
        Log.d(TAG, "URI : " + imgName);
        Bitmap bitmap = null;
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            HttpURLConnection con = (HttpURLConnection) ( new URL(imgName)).openConnection();
            con.setDoInput(true);

            con.setRequestProperty("Accept-Encoding", "identity");
            con.connect();

            int responseCode = con.getResponseCode();
            Log.d(TAG, "responseCode : " + responseCode);
            Log.d(TAG, "getContentLength : " + con.getContentLength());

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

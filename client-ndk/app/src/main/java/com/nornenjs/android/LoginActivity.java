package com.nornenjs.android;

import android.app.Activity;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import cn.pedant.SweetAlert.SweetAlertDialog;
import com.fasterxml.jackson.databind.deser.std.DateDeserializers;
import com.nornenjs.android.dto.Actor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

import java.util.Map;
import java.util.Objects;


public class LoginActivity extends Activity implements View.OnClickListener{
    private static String TAG="LogActivity";
    EditText editName,editPasswd;
    String username,userpasswd;

    TextView AlertId, AlertPasswd;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.login_activity);

        editName = (EditText)findViewById(R.id.username);
        editPasswd = (EditText)findViewById(R.id.userpasswd);

        AlertId = (TextView)findViewById(R.id.alertID);
        AlertPasswd = (TextView)findViewById(R.id.alertPasswd);

        Button send = (Button)findViewById(R.id.send);
        send.setOnClickListener(this);
        send.setOnTouchListener(btnTouchListener);

        editName.setOnFocusChangeListener(new View.OnFocusChangeListener() {
            //@Override
            public void onFocusChange(View v, boolean hasFocus) {
                editName.setHint("");
            }
        });


    }


    @Override
    public void onClick(View v) {
        switch(v.getId())
        {
            case R.id.send :
                username = editName.getText().toString();
                userpasswd = editPasswd.getText().toString();

                new PostLoginTask().execute();

                break;

        }
    }

    private View.OnTouchListener btnTouchListener = new View.OnTouchListener() {
        //@Override
        public boolean onTouch(View v, MotionEvent event) {
            Log.d("button","btn pressed");
            Button button = (Button)v;
            if(event.getAction() == MotionEvent.ACTION_DOWN) {
                ((Button) v).setHighlightColor(0xaa111111);
            }
            else if(event.getAction() == MotionEvent.ACTION_UP) {
                ((Button) v).setHighlightColor(0x00000000);
            }
            return false;
        }
    };


    private class PostLoginTask extends AsyncTask<Void, Void, Map<String, Object>>{

        private Actor actor;

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            actor = new Actor(username,userpasswd);

        }

        @Override
        protected Map<String, Object> doInBackground(Void... params) {

            try {
                // The URL for making the POST request
                final String url = getString(R.string.tomcat) + "/mobile/signIn";

                // Create a new RestTemplate instance
                RestTemplate restTemplate = new RestTemplate();

                // Make the network request, posting the message, expecting a String in response from the server
                ResponseEntity<Map> response = restTemplate.postForEntity(url, actor, Map.class);

                // Return the response body to display to the user
                return response.getBody();

            } catch (Exception e) {
                Log.e(TAG, e.getMessage(), e);
            }

            return null;
        }

        @Override
        protected void onPostExecute(Map<String, Object> stringObjectMap) {
            super.onPostExecute(stringObjectMap);

            try
            {
                Log.d(TAG, stringObjectMap.toString());
                if((Boolean)stringObjectMap.get("result"))
                {//intent
                    SharedPreferences pref = getSharedPreferences("userInfo", 0);
                    SharedPreferences.Editor prefEdit = pref.edit();
                    prefEdit.putString("username", username);
                    prefEdit.commit();

                    Intent intent = new Intent(LoginActivity.this, VolumeList.class);
                    startActivity(intent);
                    finish();
                }
                else
                {//error
                    Log.d(TAG, stringObjectMap.toString());
                    Map<String, Object> errorMessage = (Map<String, Object>)stringObjectMap.get("message");

                    if(errorMessage.get("username") != null)
                    {
                        AlertId.setVisibility(View.VISIBLE);
                        AlertId.setText(errorMessage.get("username").toString());
                    }
                    else{
                        AlertId.setVisibility(View.GONE);
                    }
                    if(errorMessage.get("password") != null)
                    {
                        AlertPasswd.setVisibility(View.VISIBLE);
                        AlertPasswd.setText(errorMessage.get("password").toString());
                    }
                    else{
                        AlertPasswd.setVisibility(View.GONE);
                    }
                }

            }catch(Exception e)
            {
                SweetAlertDialog pDialog = new SweetAlertDialog(LoginActivity.this, SweetAlertDialog.WARNING_TYPE);
                pDialog.getProgressHelper().setBarColor(Color.parseColor("#A5DC86"));//A5DC86
                pDialog.getProgressHelper().setRimColor(Color.parseColor("#33485c"));
                pDialog.setTitleText("서버에 접속할 수 없습니다.");
                pDialog.setCancelable(false);
                pDialog.show();

                View view = findViewById(R.id.blank);

                pDialog.setOnDismissListener(new DialogInterface.OnDismissListener() {
                    @Override
                    public void onDismiss(DialogInterface dialog) {
                        finish();
                    }
                });

                e.printStackTrace();
            }
        }
    }
}

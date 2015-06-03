package com.nornenjs.android;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;


public class MainActivity extends Activity {
    String username;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        SharedPreferences pref = getSharedPreferences("userInfo", 0);
        username = pref.getString("username", "");

        Loading();
    }

    private void Loading()
    {
        Handler handler = new Handler(){
            @Override
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                Intent intent;

                if(username == null || username.equals(""))
                {
                    intent = new Intent(MainActivity.this, LoginActivity.class);
                }
                else
                {
                    intent = new Intent(MainActivity.this, VolumeList.class);
                }
                startActivity(intent);
                finish();
            }
        };
        handler.sendEmptyMessageDelayed(0, 1350);
    }

}

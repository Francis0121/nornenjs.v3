package com.nornenjs.android;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;


public class MainActivity extends Activity {
    String username,userpasswd;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        SharedPreferences pref = getSharedPreferences("userInfo", 0);
        //SharedPreferences.Editor prefEdit = pref.edit();
        username = pref.getString("username","");
        userpasswd = pref.getString("userpasswd","");

        Loading();
    }

    private void Loading()
    {
        Handler handler = new Handler(){
            @Override
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                Intent intent;
                intent = new Intent(MainActivity.this, LoginActivity.class);
//                if(username == null || userpasswd.length() == 0)//user정보가 없는 경우
//                {
//                    intent = new Intent(MainActivity.this, LoginActivity.class);
//                }
//                else//user정보가 있는 경우
//                {
//                    intent = new Intent(MainActivity.this, VolumeList.class);
//                }
                startActivity(intent);
                finish();
            }
        };
        handler.sendEmptyMessageDelayed(0, 1350);
    }

}

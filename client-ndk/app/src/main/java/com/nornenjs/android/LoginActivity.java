package com.nornenjs.android;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;


public class LoginActivity extends Activity implements View.OnClickListener{

    EditText editName,editPasswd;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.login_activity);

        editName = (EditText)findViewById(R.id.username);
        editPasswd = (EditText)findViewById(R.id.userpasswd);

        Button send = (Button)findViewById(R.id.send);
        send.setOnClickListener(this);
        send.setOnTouchListener(btnTouchListener);

        editName.setOnFocusChangeListener(new View.OnFocusChangeListener() {
            //@Override
            public void onFocusChange(View v, boolean hasFocus) {
                editName.setHint("");
            }
        });
        editPasswd.setOnFocusChangeListener(new View.OnFocusChangeListener() {
            //@Override
            public void onFocusChange(View v, boolean hasFocus) {
                editPasswd.setHint("");
            }
        });

    }


    //@Override
    public void onClick(View v) {
        switch(v.getId())
        {
            case R.id.send :
                String username,userpasswd;//mainActivity의 변수와 이름 겹침
                username = editName.getText().toString();
                userpasswd = editPasswd.getText().toString();

                if(username.length() != 0 && userpasswd.length() != 0)
                {
                    Log.d("send pressed",username);

                    SharedPreferences pref = getSharedPreferences("userInfo", 0);
                    SharedPreferences.Editor prefEdit = pref.edit();
                    prefEdit.putString("username", username);
                    prefEdit.putString("userpasswd", userpasswd);
                    prefEdit.commit();

                    Intent intent = new Intent(LoginActivity.this, VolumeList.class);
                    startActivity(intent);
                }

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
}

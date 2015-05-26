package com.nornenjs.web.valid;

/**
 * Created by Francis on 2015-04-29.
 */
public class ValidationUtil {
    
    public static Boolean isNull(String value){
        return value == null || value.trim().equals("");
    }
    
    public static Boolean isEmail(String email){
        return !email.matches("^[0-9a-zA-Z]([-_\\.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_\\.]?[0-9a-zA-Z])*\\.[a-zA-Z]{2,3}$");
    }
    
    public static Boolean isUsername(String username){
        return !username.matches("^[a-z0-9_]{4,20}$");
    }

    public static Boolean isPassword(String password){
        return !password.matches("^[a-z0-9!@#$%^&*]{8,20}$");
    }
    
    public static Boolean isChar(String str){
        return !str.matches("^[\\w]{1,20}$");
    }

    public static Boolean isName(String str){
        return !str.matches("^[가-힣a-zA-Z]{1,20}$");
    }
    
}

package com.nornenjs.web.group;

/**
 * Created by Francis on 2015-04-29.
 */
public enum Authority {
    
    ADMIN(1), DOCTOR(2);

    Authority(int value) {
        this.value = value;
    }

    int value;

    public int getValue() {
        return value;
    }
}

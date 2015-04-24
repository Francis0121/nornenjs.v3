package com.nornenjs.web.data;

/**
 * Created by Francis on 2015-04-24.
 */
public enum DataType {
    
    VOLUME(1), IMAGE(2), ETC(3);
    
    int value;

    DataType(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    @Override
    public String toString() {
        return "DataType{" +
                "value=" + value +
                '}';
    }
}

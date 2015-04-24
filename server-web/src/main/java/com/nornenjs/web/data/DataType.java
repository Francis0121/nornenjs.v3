package com.nornenjs.web.data;

/**
 * Created by Francis on 2015-04-24.
 */
public enum DataType {
    
    IMAGE(1), ETC(2);
    
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

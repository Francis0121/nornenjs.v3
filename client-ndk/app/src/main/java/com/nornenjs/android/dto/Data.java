package com.nornenjs.android.dto;

import java.util.List;
import java.util.Map;

/**
 * Created by hyok on 15. 5. 8.
 */
public class Data {

    private Integer pn;

    private Integer type;

    private String username;

    private String name;

    private String savePath;

    private String inputDate;

    public Data() {
    }

    public Data(Integer type, String username, String name, String savePath) {
        this.type = type;
        this.username = username;
        this.name = name;
        this.savePath = savePath;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public Integer getPn() {
        return pn;
    }

    public void setPn(Integer pn) {
        this.pn = pn;
    }

    public Integer getType() {
        return type;
    }

    public void setType(Integer type) {
        this.type = type;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getSavePath() {
        return savePath;
    }

    public void setSavePath(String savePath) {
        this.savePath = savePath;
    }

    public String getInputDate() {
        return inputDate;
    }

    public void setInputDate(String inputDate) {
        this.inputDate = inputDate;
    }

    @Override
    public String toString() {
        return "Data{" +
                "pn=" + pn +
                ", type=" + type +
                ", username='" + username + '\'' +
                ", name='" + name + '\'' +
                ", savePath='" + savePath + '\'' +
                ", inputDate='" + inputDate + '\'' +
                '}';
    }

}
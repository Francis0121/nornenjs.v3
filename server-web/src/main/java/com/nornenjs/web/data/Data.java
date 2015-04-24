package com.nornenjs.web.data;

/**
 * Created by Francis on 2015-04-24.
 */
public class Data {
    
    private Integer pn;
    
    private Integer type;
    
    private String name;
    
    private String savePath;
    
    private String inputDate;

    public Data() {
    }

    public Data(Integer type, String name, String savePath) {
        this.type = type;
        this.name = name;
        this.savePath = savePath;
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
                ", name='" + name + '\'' +
                ", savePath='" + savePath + '\'' +
                ", inputDate='" + inputDate + '\'' +
                '}';
    }
}

package com.nornenjs.android.dto;

import java.util.List;

/**
 * Created by hyok on 15. 5. 8.
 */
public class ResponseVolumeInfo {

    private Volume volume;

    private List<Integer> thumbnails;

    private Data data;


    public ResponseVolumeInfo() {
    }

    public Volume getVolume() {
        return volume;
    }

    public void setVolume(Volume volume) {
        this.volume = volume;
    }

    public List<Integer> getThumbnails() {
        return thumbnails;
    }

    public void setThumbnails(List<Integer> thumbnails) {
        this.thumbnails = thumbnails;
    }

    public Data getData() {
        return data;
    }

    public void setData(Data data) {
        this.data = data;
    }

    @Override
    public String toString() {
        return "ResponseVolumeInfo{" +
                "volume=" + volume +
                ", thumnails=" + thumbnails +
                ", data=" + data +
                '}';
    }
}

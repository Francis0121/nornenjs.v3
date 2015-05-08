package com.nornenjs.android.dto;

import java.util.List;
import java.util.Map;

/**
 * Created by hyok on 15. 5. 8.
 */
public class ResponseVolume {

    private List<Volume> volumes;

    private Map<String, Object> volumeFilter;



    public ResponseVolume() {
    }

    public List<Volume> getVolumes() {
        return volumes;
    }

    public void setVolumes(List<Volume> volumes) {
        this.volumes = volumes;
    }

    public Map<String, Object> getVolumeFilter() {
        return volumeFilter;
    }

    public void setVolumeFilter(Map<String, Object> volumeFilter) {
        this.volumeFilter = volumeFilter;
    }

    @Override
    public String toString() {
        return "ResponseVolume{" +
                "volumes=" + volumes +
                ", volumeFilter=" + volumeFilter +
                '}';
    }

}
package com.nornenjs.web.volume;

/**
 * Created by Francis on 2015-04-24.
 */
public class Volume {
    
    private Integer pn;
    
    private Integer actorPn;
    
    private Integer volumeDataPn;
    
    private String title;
    
    private Integer width;
    
    private Integer height;
    
    private Integer depth;
    
    private String inputDate;

    public Volume() {
    }

    public Volume(Integer actorPn, Integer volumeDataPn, String title, Integer width, Integer height, Integer depth) {
        this.actorPn = actorPn;
        this.volumeDataPn = volumeDataPn;
        this.title = title;
        this.width = width;
        this.height = height;
        this.depth = depth;
    }

    public Integer getPn() {
        return pn;
    }

    public void setPn(Integer pn) {
        this.pn = pn;
    }

    public Integer getActorPn() {
        return actorPn;
    }

    public void setActorPn(Integer actorPn) {
        this.actorPn = actorPn;
    }

    public Integer getVolumeDataPn() {
        return volumeDataPn;
    }

    public void setVolumeDataPn(Integer volumeDataPn) {
        this.volumeDataPn = volumeDataPn;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public Integer getWidth() {
        return width;
    }

    public void setWidth(Integer width) {
        this.width = width;
    }

    public Integer getHeight() {
        return height;
    }

    public void setHeight(Integer height) {
        this.height = height;
    }

    public Integer getDepth() {
        return depth;
    }

    public void setDepth(Integer depth) {
        this.depth = depth;
    }

    public String getInputDate() {
        return inputDate;
    }

    public void setInputDate(String inputDate) {
        this.inputDate = inputDate;
    }

    @Override
    public String toString() {
        return "Volume{" +
                "pn=" + pn +
                ", actorPn=" + actorPn +
                ", volumeDataPn=" + volumeDataPn +
                ", title='" + title + '\'' +
                ", width=" + width +
                ", height=" + height +
                ", depth=" + depth +
                ", inputDate='" + inputDate + '\'' +
                '}';
    }
}

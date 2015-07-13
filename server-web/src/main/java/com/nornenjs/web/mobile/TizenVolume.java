package com.nornenjs.web.mobile;

/**
 * Created by pi on 15. 7. 13.
 */
public class TizenVolume {

    private String pn;

    private String username;

    private String volumeDataPn;

    private String title;

    private String width;

    private String height;

    private String depth;

    private String inputDate;

    public TizenVolume() {
    }

    public TizenVolume(String pn, String username, String volumeDataPn, String title, String width, String height, String depth, String inputDate) {
        this.pn = pn;
        this.username = username;
        this.volumeDataPn = volumeDataPn;
        this.title = title;
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.inputDate = inputDate;
    }

    public String getPn() {
        return pn;
    }

    public String getUsername() {
        return username;
    }

    public String getVolumeDataPn() {
        return volumeDataPn;
    }

    public String getTitle() {
        return title;
    }

    public String getWidth() {
        return width;
    }

    public String getHeight() {
        return height;
    }

    public String getDepth() {
        return depth;
    }

    public String getInputDate() {
        return inputDate;
    }

    @Override
    public String toString() {
        return "TizenVolume{" +
                "pn='" + pn + '\'' +
                ", username='" + username + '\'' +
                ", volumeDataPn='" + volumeDataPn + '\'' +
                ", title='" + title + '\'' +
                ", width='" + width + '\'' +
                ", height='" + height + '\'' +
                ", depth='" + depth + '\'' +
                ", inputDate='" + inputDate + '\'' +
                '}';
    }
}

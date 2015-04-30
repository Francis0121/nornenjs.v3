package com.nornenjs.web.volume.thumbnail;

/**
 * Created by pi on 15. 4. 30.
 */
public class Thumbnail {

    private Integer dataPn;

    private Integer thumbnailPn;

    public Thumbnail() {
    }

    public Thumbnail(Integer dataPn) {
        this.dataPn = dataPn;
    }

    public Integer getDataPn() {
        return dataPn;
    }

    public void setDataPn(Integer dataPn) {
        this.dataPn = dataPn;
    }

    public Integer getThumbnailPn() {
        return thumbnailPn;
    }

    public void setThumbnailPn(Integer thumbnailPn) {
        this.thumbnailPn = thumbnailPn;
    }


    @Override
    public String toString() {
        return "Thumbnail{" +
                "dataPn=" + dataPn +
                ", thumbnailPn=" + thumbnailPn +
                '}';
    }
}

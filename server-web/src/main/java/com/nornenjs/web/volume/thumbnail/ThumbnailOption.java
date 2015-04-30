package com.nornenjs.web.volume.thumbnail;

/**
 * Created by pi on 15. 4. 30.
 */
public class ThumbnailOption {

    private Integer pn;

    private Integer width;

    private Integer height;

    private Integer type;

    private Integer mprType;

    private Integer quality;

    private Float brightness;

    private Float density;

    private Float transferOffset;

    private Float transferScaleX;

    private Float transferScaleY;

    private Float transferScaleZ;

    private Float positionZ;

    private Float rotationX;

    private Float rotationY;

    public ThumbnailOption() {
    }

    public Integer getPn() {
        return pn;
    }

    public void setPn(Integer pn) {
        this.pn = pn;
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

    public Integer getType() {
        return type;
    }

    public void setType(Integer type) {
        this.type = type;
    }

    public Integer getMprType() {
        return mprType;
    }

    public void setMprType(Integer mprType) {
        this.mprType = mprType;
    }

    public Integer getQuality() {
        return quality;
    }

    public void setQuality(Integer quality) {
        this.quality = quality;
    }

    public Float getBrightness() {
        return brightness;
    }

    public void setBrightness(Float brightness) {
        this.brightness = brightness;
    }

    public Float getDensity() {
        return density;
    }

    public void setDensity(Float density) {
        this.density = density;
    }

    public Float getTransferOffset() {
        return transferOffset;
    }

    public void setTransferOffset(Float transferOffset) {
        this.transferOffset = transferOffset;
    }

    public Float getTransferScaleX() {
        return transferScaleX;
    }

    public void setTransferScaleX(Float transferScaleX) {
        this.transferScaleX = transferScaleX;
    }

    public Float getTransferScaleY() {
        return transferScaleY;
    }

    public void setTransferScaleY(Float transferScaleY) {
        this.transferScaleY = transferScaleY;
    }

    public Float getTransferScaleZ() {
        return transferScaleZ;
    }

    public void setTransferScaleZ(Float transferScaleZ) {
        this.transferScaleZ = transferScaleZ;
    }

    public Float getRotationX() {
        return rotationX;
    }

    public void setRotationX(Float rotationX) {
        this.rotationX = rotationX;
    }

    public Float getRotationY() {
        return rotationY;
    }

    public void setRotationY(Float rotationY) {
        this.rotationY = rotationY;
    }

    public Float getPositionZ() {
        return positionZ;
    }

    public void setPositionZ(Float positionZ) {
        this.positionZ = positionZ;
    }

    @Override
    public String toString() {
        return "ThumbnailOption{" +
                "pn=" + pn +
                ", width=" + width +
                ", height=" + height +
                ", type=" + type +
                ", mprType=" + mprType +
                ", quality=" + quality +
                ", brightness=" + brightness +
                ", density=" + density +
                ", transferOffset=" + transferOffset +
                ", transferScaleX=" + transferScaleX +
                ", transferScaleY=" + transferScaleY +
                ", transferScaleZ=" + transferScaleZ +
                ", positionZ=" + positionZ +
                ", rotationX=" + rotationX +
                ", rotationY=" + rotationY +
                '}';
    }
}

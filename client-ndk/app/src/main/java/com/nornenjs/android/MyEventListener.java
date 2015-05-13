package com.nornenjs.android;

/**
 * Created by pi on 15. 3. 20.
 */
public interface MyEventListener {
    void RotationEvent(float rotationX, float rotationY);
    void TranslationEvent(float translationX, float translationY);
    void PinchZoomEvent(float div);
    void GetPng();
    void BackToPreview();
}

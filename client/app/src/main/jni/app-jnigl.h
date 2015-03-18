#ifndef _APP_JNIGL_H_
#define _APP_JNIGL_H_

void nativeDrawIteration(float mx, float my);

void nativeOnTouchEvent(int e, float x, float y);
void nativeOnTrackballEvent(int e, float x, float y);

void nativeOnCreate();
void nativeOnDestroy();
void nativeOnPause();
void nativeOnResume();
//void nativeOnAccelerometer(float x,float y,float z);
//void nativeSendEvent(int action, float x, float y);
void nativeInitGL(int w, int h);
void nativeOnResize(int w, int h);

#endif

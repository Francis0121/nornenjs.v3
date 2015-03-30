#include <jni.h>
#include <android/log.h>

#include "com_nornenjs_android_JniGLActivity.h"
#include "app-jnigl.h"

/* For JNI: C++ compiler need this */
#ifdef __cplusplus
extern "C" {
#endif


/*
 * Class:     Java_com_nornenjs_android_JniGLActivity_nativeOnCreate
 * Method:    nativeOnCreate
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_nornenjs_android_JniGLActivity_nativeOnCreate
  (JNIEnv *, jobject)
{
	nativeOnCreate();
}

/*
 * Class:     Java_com_nornenjs_android_JniGLActivity_nativeOnPause
 * Method:    nativeOnPause
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_nornenjs_android_JniGLActivity_nativeOnPause
  (JNIEnv *, jobject)
{
	nativeOnPause();
}

/*
 * Class:     Java_com_nornenjs_android_JniGLActivity_nativeOnResume
 * Method:    nativeOnResume
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_nornenjs_android_JniGLActivity_nativeOnResume
  (JNIEnv *, jobject)
{
	nativeOnResume();
}

/*
 * Class:     Java_com_nornenjs_android_JniGLActivity_nativeOnDestroy
 * Method:    nativeOnDestroy
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_nornenjs_android_JniGLActivity_nativeOnDestroy
  (JNIEnv *, jobject)
{
	nativeOnDestroy();
}

/*
 * Class:     Java_com_nornenjs_android_JniGLActivity_nativeInitGL
 * Method:    nativeInitGL
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_nornenjs_android_JniGLActivity_nativeInitGL
  (JNIEnv *, jobject)
{
	nativeInitGL(0,0);
}

/*
 * Class:     Java_com_nornenjs_android_JniGLActivity_nativeResize
 * Method:    nativeResize
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_com_nornenjs_android_JniGLActivity_nativeResize
  (JNIEnv *, jobject, jint w, jint h)
{
	nativeOnResize( w, h);
}

/*
 * Class:     Java_com_nornenjs_android_JniGLActivity_nativeDrawIteration
 * Method:    nativeDrawIteration
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_nornenjs_android_JniGLActivity_nativeDrawIteration
  (JNIEnv *, jclass, jfloat mx, jfloat my)
{
	nativeDrawIteration(mx,my);
}

/*
 * Class:     Java_com_nornenjs_android_JniGLActivity_nativeOnTrackballEvent
 * Method:    nativeOnTrackballEvent
 * Signature: (IFF)V
 */
JNIEXPORT void JNICALL Java_com_nornenjs_android_JniGLActivity_nativeOnTrackballEvent
(JNIEnv *, jobject, jint e, jfloat x, jfloat y)
{
	nativeOnTrackballEvent(e,x,y);
}

/*
 * Class:     com_nornenjs_android_JniGLActivity
 * Method:    nativeOnTouchEvent
 * Signature: (IFF)V
 */
JNIEXPORT void JNICALL Java_com_nornenjs_android_JniGLActivity_nativeOnTouchEvent
  (JNIEnv *, jobject, jint e,jfloat x, jfloat y)
{
	nativeOnTouchEvent(e,x,y);
}

/*
 * Class:     Java_com_nornenjs_android_JniGLActivity_nativeInitTextureData
 * Method:    nativeInitTextureData
 * Signature: ([III)V
 */
JNIEXPORT void JNICALL Java_com_nornenjs_android_JniGLActivity_nativeInitTextureData
(JNIEnv *env, jclass thiz, jintArray arr, jint width, jint height)
{
	int *data = env->GetIntArrayElements(arr, 0);

	initTextureData(data, width, height);

	env->ReleaseIntArrayElements((jintArray)arr, data, JNI_ABORT);
}

/*
 * Class:     com_nornenjs_android_JniGLActivity
 * Method:    nativeSetTextureData
 * Signature: ([III)V
 */
JNIEXPORT void JNICALL Java_com_nornenjs_android_JniGLActivity_nativeSetTextureData
(JNIEnv *env, jclass thiz, jintArray arr, jint width, jint height)
{
	int *data = env->GetIntArrayElements(arr, 0);

	setTextureData(data, width, height);

	env->ReleaseIntArrayElements((jintArray)arr, data, JNI_ABORT);
}


#ifdef __cplusplus
}
#endif

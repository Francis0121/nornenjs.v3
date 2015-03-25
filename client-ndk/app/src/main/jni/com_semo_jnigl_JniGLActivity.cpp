#include <jni.h>
#include <android/log.h>

#include "com_semo_jnigl_JniGLActivity.h"

#include "app-jnigl.h"

/* For JNI: C++ compiler need this */
#ifdef __cplusplus
extern "C" {
#endif


/*
 * Class:     com_semo_jnigl_JniGLActivity
 * Method:    nativeOnCreate
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_semo_jnigl_JniGLActivity_nativeOnCreate
  (JNIEnv *, jobject)
{
	nativeOnCreate();
}

/*
 * Class:     com_semo_jnigl_JniGLActivity
 * Method:    nativeOnPause
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_semo_jnigl_JniGLActivity_nativeOnPause
  (JNIEnv *, jobject)
{
	nativeOnPause();
}

/*
 * Class:     com_semo_jnigl_JniGLActivity
 * Method:    nativeOnResume
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_semo_jnigl_JniGLActivity_nativeOnResume
  (JNIEnv *, jobject)
{
	nativeOnResume();
}

/*
 * Class:     com_semo_jnigl_JniGLActivity
 * Method:    nativeOnDestroy
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_semo_jnigl_JniGLActivity_nativeOnDestroy
  (JNIEnv *, jobject)
{
	nativeOnDestroy();
}

/*
 * Class:     com_semo_jnigl_JniGLActivity
 * Method:    nativeInitGL
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_semo_jnigl_JniGLActivity_nativeInitGL
  (JNIEnv *, jobject)
{
	nativeInitGL(0,0);
}

/*
 * Class:     com_semo_jnigl_JniGLActivity
 * Method:    nativeResize
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_com_semo_jnigl_JniGLActivity_nativeResize
  (JNIEnv *, jobject, jint w, jint h)
{
	nativeOnResize( w, h);
}

/*
 * Class:     com_semo_jnigl_JniGLActivity
 * Method:    nativeDrawIteration
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_semo_jnigl_JniGLActivity_nativeDrawIteration
  (JNIEnv *, jclass, jfloat mx, jfloat my)
{
	nativeDrawIteration(mx,my);
}

JNIEXPORT void JNICALL Java_com_semo_jnigl_JniGLActivity_nativeOnTrackballEvent
  (JNIEnv *, jclass, jint e,jfloat x, jfloat y)
{
	nativeOnTrackballEvent(e,x,y);
}

JNIEXPORT void JNICALL Java_com_semo_jnigl_JniGLActivity_nativeOnTouchEvent
  (JNIEnv *, jclass, jint e,jfloat x, jfloat y)
{
	nativeOnTouchEvent(e,x,y);
}

JNIEXPORT void JNICALL Java_com_semo_jnigl_JniGLActivity_nativeInitTextureData
(JNIEnv *env, jobject thiz, jintArray arr, jint width, jint height)
{
	int *data = env->GetIntArrayElements(arr, 0);

	initTextureData(data, width, height);

	env->ReleaseIntArrayElements((jintArray)arr, data, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_semo_jnigl_JniGLActivity_nativeSetTextureData
(JNIEnv *env, jobject thiz, jintArray arr, jint width, jint height)
{
	int *data = env->GetIntArrayElements(arr, 0);

	setTextureData(data, width, height);

	env->ReleaseIntArrayElements((jintArray)arr, data, JNI_ABORT);
}


#ifdef __cplusplus
}
#endif

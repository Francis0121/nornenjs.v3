LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := opengles
LOCAL_SRC_FILES := com_nornenjs_android_JniGLActivity.cpp app-jnigl.cpp com_nornenjs_android_MprActivity.cpp

LOCAL_LDLIBS := -lGLESv1_CM -ldl -llog
                   
include $(BUILD_SHARED_LIBRARY)

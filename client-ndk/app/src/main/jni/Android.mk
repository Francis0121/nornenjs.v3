LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := jnigl
LOCAL_SRC_FILES := com_semo_jnigl_JniGLActivity.cpp app-jnigl.cpp

LOCAL_LDLIBS := -lGLESv1_CM -ldl -llog
                   
include $(BUILD_SHARED_LIBRARY)

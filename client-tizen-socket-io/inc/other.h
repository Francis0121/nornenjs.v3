#ifndef __other_H__
#define __other_H__

#include <app.h>
#include <Elementary.h>
#include <system_settings.h>
#include <efl_extension.h>
#include <dlog.h>

#ifdef  LOG_TAG
#undef  LOG_TAG
#endif
#define LOG_TAG "nornenjs"

#define APPDATA_KEY "AppData" //add

#if !defined(PACKAGE)
#define PACKAGE "org.tizen.other"
#endif

typedef struct appdata
{
	Evas_Object *table;
	Evas_Object *bg;
	Evas_Object *win;
	Evas_Object *glview;
	Evas_Object *conform;
	Ecore_Animator *anim;

	// ~ socekt.io Event handl variable

	/**
	* Touch event - 3d object rotation
	*/
	Eina_Bool mouse_down : 1;
	float rotationX;
	float rotationY;

	/**
	* Multi touch event - 3d object resize
	*/
	Eina_Bool multi_mouse_down : 1;
	float positionZ;

} appdata_s;

void login_cb(void *data, Evas_Object *obj, void *event_info);

#endif /* __other_H__ */


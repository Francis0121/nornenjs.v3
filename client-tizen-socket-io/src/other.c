#include <tizen.h>
#include "socket.hpp"
#include "socket_io_client.hpp"
#include "other.h"
#include <pthread.h>

//add
#include <app.h>
#include <efl_extension.h>
#include "copy_GLES.h"

#define LOG_TAG_SOCKET_IO "socket.io"

static pthread_t thread_id;
static pthread_mutex_t lock;
static Eina_Bool thread_finish = EINA_FALSE;

//typedef struct appdata {
//	Evas_Object *win;
//	Evas_Object *conform;
//	Evas_Object *label;
//} appdata_s;

static Evas_Object*
_glview_create(appdata_s *ad)
{
   Evas_Object *obj;

   /* Create a GLView with an OpenGL-ES 1.1 context */
   obj = elm_glview_version_add(ad->win, EVAS_GL_GLES_1_X);
   elm_table_pack(ad->table, obj, 1, 1, 3, 1);
   evas_object_data_set(obj, APPDATA_KEY, ad);

   elm_glview_mode_set(obj, ELM_GLVIEW_ALPHA | ELM_GLVIEW_DEPTH);
   elm_glview_resize_policy_set(obj, ELM_GLVIEW_RESIZE_POLICY_RECREATE);
   elm_glview_render_policy_set(obj, ELM_GLVIEW_RENDER_POLICY_ON_DEMAND);

   elm_glview_init_func_set(obj, init_gles);
   elm_glview_del_func_set(obj, destroy_gles);
   elm_glview_resize_func_set(obj, resize_gl);
   elm_glview_render_func_set(obj, draw_gl);

   return obj;
}//add

static Eina_Bool
_anim_cb(void *data)
{
   appdata_s *ad = data;

   elm_glview_changed_set(ad->glview);
   return ECORE_CALLBACK_RENEW;
}//add

static void
_destroy_anim(void *data, Evas *evas, Evas_Object *obj, void *event_info)
{
   Ecore_Animator *ani = data;
   ecore_animator_del(ani);
}//add

static void
_close_cb(void *data EINA_UNUSED,
          Evas_Object *obj EINA_UNUSED, void *event_info EINA_UNUSED)
{
   ui_app_exit();
}//add


static void
_win_resize_cb(void *data, Evas *e EINA_UNUSED,
               Evas_Object *obj EINA_UNUSED, void *event_info EINA_UNUSED)
{
   int w, h;
   appdata_s *ad = data;

   evas_object_geometry_get(ad->win, NULL, NULL, &w, &h);
   evas_object_resize(ad->table, w, h);
   evas_object_resize(ad->bg, w, h);
}//add

static void
win_delete_request_cb(void *data, Evas_Object *obj, void *event_info)
{
	ui_app_exit();
}

static void
win_back_cb(void *data, Evas_Object *obj, void *event_info)
{
	appdata_s *ad = data;
	/* Let window go to hide state. */
	elm_win_lower(ad->win);
}

static void
create_base_gui(appdata_s *ad)
{
	/* Window */
	ad->win = elm_win_util_standard_add(PACKAGE, PACKAGE);
	elm_win_autodel_set(ad->win, EINA_TRUE);

	if (elm_win_wm_rotation_supported_get(ad->win)) {
		int rots[4] = { 0, 90, 180, 270 };
		elm_win_wm_rotation_available_rotations_set(ad->win, (const int *)(&rots), 4);
	}

	evas_object_smart_callback_add(ad->win, "delete,request", win_delete_request_cb, NULL);
	eext_object_event_callback_add(ad->win, EEXT_CALLBACK_BACK, win_back_cb, ad);

	/* Conformant */
	ad->conform = elm_conformant_add(ad->win);
	elm_win_indicator_mode_set(ad->win, ELM_WIN_INDICATOR_SHOW);
	elm_win_indicator_opacity_set(ad->win, ELM_WIN_INDICATOR_OPAQUE);
	evas_object_size_hint_weight_set(ad->conform, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
	elm_win_resize_object_add(ad->win, ad->conform);
	evas_object_show(ad->conform);

	/* Label*/
	ad->label = elm_label_add(ad->conform);
	elm_object_text_set(ad->label, rapidjson_test());
	evas_object_size_hint_weight_set(ad->label, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
	elm_object_content_set(ad->conform, ad->label);
	evas_object_show(ad->label);


	int status = 0;

	dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "thread_start");

//	int threadError = 0;
//
//	if ((threadError = pthread_create(&thread_id, NULL, socket_io_client, NULL))){
//			perror("pthread_create!\n");
//			dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "thread_error %d", threadError);
//	}
//
//	dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "finish %d", status);
	/* Show window after base gui is set up */

	evas_object_show(ad->win);

	/////////////////////////////////






	// Thread가 종료되기를 기다린후 Thread의 리턴값을 출력한다.
	//pthread_join(thread_t, (void **)&status);

//	dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "thread_start");
//	socket_io_client();

	/////////////////////////////////
}


static Evas_Object* add_win(const char *name) {
	Evas_Object *win;

	elm_config_accel_preference_set("opengl");
	win = elm_win_util_standard_add(name, "OpenGL example: Cube");

	if (!win)
		return NULL;

	if (elm_win_wm_rotation_supported_get(win)) {
		int rots[4] = { 0, 90, 180, 270 };
		elm_win_wm_rotation_available_rotations_set(win, rots, 4);
	}

	evas_object_show(win);

	return win;
}

static bool
app_create(void *data)
{
	Evas_Object *o, *t;
	   appdata_s *ad = (appdata_s*)data;

	   /* Force OpenGL engine */
	   elm_config_accel_preference_set("opengl");

	   /* Add a window */
	   ad->win = o = elm_win_add(NULL,"glview", ELM_WIN_BASIC);
	   evas_object_smart_callback_add(o, "delete,request", _close_cb, ad);
	   evas_object_event_callback_add(o, EVAS_CALLBACK_RESIZE, _win_resize_cb, ad);
	   eext_object_event_callback_add(o, EEXT_CALLBACK_BACK, _close_cb, ad);
	   evas_object_show(o);

	   /* Add a background */
	   ad->bg = o = elm_bg_add(ad->win);
	   elm_win_resize_object_add(ad->win, ad->bg);
	   elm_bg_color_set(o, 68, 68, 68);
	   evas_object_show(o);

	   /* Add a resize conformant */
	   ad->conform = o = elm_conformant_add(ad->win);
	   elm_win_resize_object_add(ad->win, ad->conform);
	   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, EVAS_HINT_FILL);
	   evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
	   evas_object_show(o);

	   ad->table = t = elm_table_add(ad->win);
	   evas_object_show(t);

	   o = elm_label_add(ad->win);
	   elm_object_text_set(o, "Gles 1.1 Cube");
	   elm_table_pack(t, o, 1, 0, 3, 1);
	   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, EVAS_HINT_FILL);
	   evas_object_size_hint_weight_set(o, 0.00001, 0.00001);
	   evas_object_show(o);

	   o = elm_button_add(ad->win);
	   elm_object_text_set(o, "Quit");
	   evas_object_smart_callback_add(o, "clicked", _close_cb, ad);
	   elm_table_pack(t, o, 1, 9, 3, 1);
	   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, EVAS_HINT_FILL);
	   evas_object_size_hint_weight_set(o, 0.00001, 0.00001);
	   evas_object_show(o);

	   ad->glview = o = _glview_create(ad);
	   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, EVAS_HINT_FILL);
	   evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
	   evas_object_show(o);

	   ad->anim = ecore_animator_add(_anim_cb, ad);
	   evas_object_event_callback_add(ad->glview, EVAS_CALLBACK_DEL, _destroy_anim, ad->anim);

	   return true;
}//add

static void
app_control(app_control_h app_control, void *data)
{

	int status = 0;

	dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "thread_start");

	int threadError = 0;

	if ((threadError = pthread_create(&thread_id, NULL, socket_io_client, NULL))){
			perror("pthread_create!\n");
			dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "thread_error %d", threadError);
	}

	/*
	if ((threadError = pthread_join(thread_id,&thread_return))){
				perror("pthread_join!\n");
				dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "thread_error %d", threadError);
	}
	*/

	dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "finish %d", status);
	/* Show window after base gui is set up */

	/* Handle the launch request. */

}

static void
app_pause(void *data)
{
	/* Take necessary actions when application becomes invisible. */
}

static void
app_resume(void *data)
{
	/* Take necessary actions when application becomes visible. */
}

static void
app_terminate(void *data)
{
	/* Release all resources. */
	turn_off_flag();
}

static void
ui_app_lang_changed(app_event_info_h event_info, void *user_data)
{
	/*APP_EVENT_LANGUAGE_CHANGED*/
	char *locale = NULL;
	system_settings_get_value_string(SYSTEM_SETTINGS_KEY_LOCALE_LANGUAGE, &locale);
	elm_language_set(locale);
	free(locale);
	return;
}

static void
ui_app_orient_changed(app_event_info_h event_info, void *user_data)
{
	/*APP_EVENT_DEVICE_ORIENTATION_CHANGED*/
	return;
}

static void
ui_app_region_changed(app_event_info_h event_info, void *user_data)
{
	/*APP_EVENT_REGION_FORMAT_CHANGED*/
}

static void
ui_app_low_battery(app_event_info_h event_info, void *user_data)
{
	/*APP_EVENT_LOW_BATTERY*/
}

static void
ui_app_low_memory(app_event_info_h event_info, void *user_data)
{
	/*APP_EVENT_LOW_MEMORY*/
}

int
main(int argc, char *argv[])
{
	appdata_s ad = {0,};
	int ret = 0;

	ui_app_lifecycle_callback_s event_callback = {0,};
	app_event_handler_h handlers[5] = {NULL, };


	event_callback.create = app_create;
	event_callback.terminate = app_terminate;
	event_callback.pause = app_pause;
	event_callback.resume = app_resume;
	event_callback.app_control = app_control;

	ui_app_add_event_handler(&handlers[APP_EVENT_LOW_BATTERY], APP_EVENT_LOW_BATTERY, ui_app_low_battery, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_LOW_MEMORY], APP_EVENT_LOW_MEMORY, ui_app_low_memory, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_DEVICE_ORIENTATION_CHANGED], APP_EVENT_DEVICE_ORIENTATION_CHANGED, ui_app_orient_changed, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_LANGUAGE_CHANGED], APP_EVENT_LANGUAGE_CHANGED, ui_app_lang_changed, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_REGION_FORMAT_CHANGED], APP_EVENT_REGION_FORMAT_CHANGED, ui_app_region_changed, &ad);
	ui_app_remove_event_handler(handlers[APP_EVENT_LOW_MEMORY]);

	ret = ui_app_main(argc, argv, &event_callback, &ad);

	int thread_return;

	pthread_join(thread_id,&thread_return);

	if (ret != APP_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "app_main() is failed. err = %d", ret);
	}

	return ret;
}

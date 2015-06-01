#include <app.h>
#include <efl_extension.h>
#include <tizen.h>
#include <pthread.h>
#include <math.h>
#include "other.h"
#include "socket_io_client.hpp"
#include "copy_GLES.h"

#define LOG_TAG_SOCKET_IO "socket.io"

static pthread_t thread_id;

static void
slider_move_cb(void *data, Evas_Object *obj, void *event_info){

	appdata_s *ad = data;
	double brightness = elm_slider_value_get(ad->brightSlider);

	emit_brightness((float) (brightness/100.0));
}

// ~ Mouse event
static void
mouse_down_cb(void *data, Evas *e , Evas_Object *obj , void *event_info){
	Evas_Event_Mouse_Move *ev = (Evas_Event_Mouse_Move *)event_info;
	appdata_s *ad = data;

	ad->mouse_down = EINA_TRUE;

	ad->oldVectorX1 = ev->cur.canvas.x;
	ad->oldVectorY1 = ev->cur.canvas.y;
}

static void
mouse_move_cb(void *data, Evas *e , Evas_Object *obj , void *event_info){
	Evas_Event_Mouse_Move *ev = (Evas_Event_Mouse_Move *)event_info;
	appdata_s *ad = data;

	if(ad->mouse_down && !ad->multi_mouse_down) {
		ad->rotationX += (ev->cur.canvas.x - ev->prev.canvas.x) / 10.0;
		ad->rotationY += (ev->cur.canvas.y - ev->prev.canvas.y) / 10.0;
		emit_rotation(ad->rotationX, ad->rotationY);
	}

	if(ad->multi_mouse_down){
		//dlog_print(DLOG_VERBOSE, LOG_TAG_SOCKET_IO, "Multi touch first point %i %i", ev->cur.canvas.x, ev->cur.canvas.y);
		dlog_print(DLOG_VERBOSE, LOG_TAG_SOCKET_IO, "single move");

		ad->oldVectorX1 = ev->cur.canvas.x;
		ad->oldVectorY1 = ev->cur.canvas.y;

	}
}

static void
mouse_up_cb(void *data, Evas *e , Evas_Object *obj , void *event_info){
	appdata_s *ad = data;

	if(ad->mouse_down){
		emit_quality();
		ad->mouse_down = EINA_FALSE;
	}
}

// ~ Multi Mouse event
static float
spacing(float x1,float y1, float x2, float y2) {

    float x = x1- x2;
    float y = y1 - y2;

    return sqrt(x * x + y * y);
}

static void
multi_mouse_down_cb(void *data, Evas *e, Evas_Object *obj , void *event_info){
	Evas_Event_Multi_Move *ev = (Evas_Event_Multi_Move *) event_info;
	appdata_s *ad = data;

	ad->multi_mouse_down = EINA_TRUE;
	dlog_print(DLOG_VERBOSE, LOG_TAG_SOCKET_IO, "multi down");

	ad->oldVectorX2 = ev->cur.canvas.x;
	ad->oldVectorY2 = ev->cur.canvas.y;

	ad->oldDist = spacing(ad->oldVectorX1,ad->oldVectorY1,ad->oldVectorX2,ad->oldVectorY2);
}

static void
multi_mouse_move_cb(void *data, Evas *e, Evas_Object *obj , void *event_info){

	Evas_Event_Multi_Move *ev = (Evas_Event_Multi_Move *) event_info;
	appdata_s *ad = data;

	if(ad->multi_mouse_down) {

		ad->oldVectorX2 = ev->cur.canvas.x;
		ad->oldVectorY2 = ev->cur.canvas.y;

		ad->newDist = spacing(ad->oldVectorX1,ad->oldVectorY1,ad->oldVectorX2,ad->oldVectorY2);

		// zoom in
		if (ad->newDist - ad->oldDist > 15) {

			ad->oldDist = ad->newDist;
			ad->div -= (((ad->newDist / ad->oldDist) / 50) * 10);

			if (ad->div <= 0.2f) {
				ad->div = 0.2f;
			}

			emit_zoom(ad->div);
		// zoom out
		}else if (ad->oldDist - ad->newDist > 15) {

			ad->oldDist = ad->newDist;
			ad->div += (((ad->newDist / ad->oldDist) / 50) * 10);
			if (ad->div >= 10.0f) {
				ad->div = 10.0f;
			}
			emit_zoom(ad->div);
        }
	}
}

static void
multi_mouse_up_cb(void *data, Evas *e, Evas_Object *obj , void *event_info){
	appdata_s *ad = data;
	if(ad->multi_mouse_down){
		ad->multi_mouse_down = EINA_FALSE;
		emit_quality();
	}
}

static Evas_Object*
_glview_create(appdata_s *ad){
   Evas_Object *obj;

   /* Create a GLView with an OpenGL-ES 1.1 context */
   obj = elm_glview_version_add(ad->win, EVAS_GL_GLES_1_X);
   elm_table_pack(ad->table, obj, 1, 1, 1, 1);
   evas_object_data_set(obj, APPDATA_KEY, ad);

   elm_glview_mode_set(obj, ELM_GLVIEW_ALPHA | ELM_GLVIEW_DEPTH);
   elm_glview_resize_policy_set(obj, ELM_GLVIEW_RESIZE_POLICY_RECREATE);
   elm_glview_render_policy_set(obj, ELM_GLVIEW_RENDER_POLICY_ON_DEMAND);

   elm_glview_init_func_set(obj, init_gles);
   elm_glview_del_func_set(obj, destroy_gles);
   elm_glview_resize_func_set(obj, resize_gl);
   elm_glview_render_func_set(obj, draw_gl);

   return obj;
}

static Eina_Bool
_anim_cb(void *data){
   appdata_s *ad = data;
   elm_glview_changed_set(ad->glview);
   return ECORE_CALLBACK_RENEW;
}

static void
_destroy_anim(void *data, Evas *evas, Evas_Object *obj, void *event_info){
   Ecore_Animator *ani = data;
   ecore_animator_del(ani);
}

static void
_close_cb(void *data EINA_UNUSED, Evas_Object *obj EINA_UNUSED, void *event_info EINA_UNUSED){
   ui_app_exit();
}

static void
_win_resize_cb(void *data, Evas *e EINA_UNUSED, Evas_Object *obj EINA_UNUSED, void *event_info EINA_UNUSED){
   int w, h;
   appdata_s *ad = data;

   evas_object_geometry_get(ad->win, NULL, NULL, &w, &h);
   evas_object_resize(ad->table, w, h);
   evas_object_resize(ad->bg, w, h);
}

static bool
app_create(void *data)
{
	Evas_Object *o, *t;
	appdata_s *ad = (appdata_s*)data;

	ad->div = 2.0f;
	/* Force OpenGL engine */
	elm_config_accel_preference_set("opengl");

	/* Add a window */
	ad->win = o = elm_win_add(NULL,"glview", ELM_WIN_BASIC);
	evas_object_smart_callback_add(o, "delete,request", _close_cb, ad);
	evas_object_event_callback_add(o, EVAS_CALLBACK_RESIZE, _win_resize_cb, ad);
	eext_object_event_callback_add(o, EEXT_CALLBACK_BACK, _close_cb, ad);
	evas_object_show(o);

	/* Add a resize conformant */
	ad->conform = o = elm_conformant_add(ad->win);
	elm_win_resize_object_add(ad->win, ad->conform);
	evas_object_size_hint_align_set(o, EVAS_HINT_FILL, EVAS_HINT_FILL);
	evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
	evas_object_show(o);

	ad->table = t = elm_table_add(ad->win);
	evas_object_show(t);

	ad->glview = o = _glview_create(ad);
	evas_object_size_hint_align_set(o, EVAS_HINT_FILL, EVAS_HINT_FILL);
	evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
	evas_object_show(o);

	ad->anim = ecore_animator_add(_anim_cb, ad);
	evas_object_event_callback_add(ad->glview, EVAS_CALLBACK_DEL, _destroy_anim, ad->anim);

	ad->brightSlider = o = elm_slider_add(ad->win);
	elm_table_pack(t, o, 1, 2, 1, 1);
	evas_object_size_hint_align_set(o, EVAS_HINT_FILL, EVAS_HINT_FILL);
	evas_object_size_hint_weight_set(o, 0.00001, 0.00001);
	elm_slider_indicator_format_set(o, "%1.0f");
	elm_slider_min_max_set(o, 0, 600);
	elm_slider_value_set(o, 200);
	evas_object_smart_callback_add(ad->brightSlider, "changed", slider_move_cb, ad);
	evas_object_show(o);

	// ~ touch event add
	evas_object_event_callback_add(ad->glview, EVAS_CALLBACK_MOUSE_DOWN, mouse_down_cb, ad);
	evas_object_event_callback_add(ad->glview, EVAS_CALLBACK_MOUSE_UP, mouse_up_cb, ad);
	evas_object_event_callback_add(ad->glview, EVAS_CALLBACK_MOUSE_MOVE, mouse_move_cb, ad);

	// ~ multi touch event
	evas_object_event_callback_add(ad->glview, EVAS_CALLBACK_MULTI_DOWN, multi_mouse_down_cb, ad);
	evas_object_event_callback_add(ad->glview, EVAS_CALLBACK_MULTI_MOVE, multi_mouse_move_cb, ad);
	evas_object_event_callback_add(ad->glview, EVAS_CALLBACK_MULTI_UP, multi_mouse_up_cb, ad);

   int status = 0;

	dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "thread_start");

	int threadError = 0;

	if( (threadError = pthread_create(&thread_id, NULL, socket_io_client,(void *)o ) ) ){
		dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "thread_error %d", threadError);
	}

	dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "finish %d", status);
   return true;
}

static void
app_control(app_control_h app_control, void *data){

	/* Show window after base gui is set up */

	/* Handle the launch request. */
}

static void
app_pause(void *data){
	/* Take necessary actions when application becomes invisible. */

}

static void
app_resume(void *data){
	/* Take necessary actions when application becomes visible. */
}

static void
app_terminate(void *data){
	/* Release all resources. */
	turn_off_flag();

   free_que();
   if(image)
	   free(image);
   //예외사항..메모리 생성 후 que에 저장하기 전에 종료사항이 있다면
}

static void
ui_app_lang_changed(app_event_info_h event_info, void *user_data){
	/*APP_EVENT_LANGUAGE_CHANGED*/
	char *locale = NULL;
	system_settings_get_value_string(SYSTEM_SETTINGS_KEY_LOCALE_LANGUAGE, &locale);
	elm_language_set(locale);
	free(locale);
	return;
}

static void
ui_app_orient_changed(app_event_info_h event_info, void *user_data){
	/*APP_EVENT_DEVICE_ORIENTATION_CHANGED*/
	return;
}

static void
ui_app_region_changed(app_event_info_h event_info, void *user_data){
	/*APP_EVENT_REGION_FORMAT_CHANGED*/
}

static void
ui_app_low_battery(app_event_info_h event_info, void *user_data){
	/*APP_EVENT_LOW_BATTERY*/
}

static void
ui_app_low_memory(app_event_info_h event_info, void *user_data){
	/*APP_EVENT_LOW_MEMORY*/
}

int
main(int argc, char *argv[]){
	appdata_s ad = {0,};
	int ret = 0;

	ui_app_lifecycle_callback_s event_callback = {0,};
	app_event_handler_h handlers[5] = {NULL, };

	event_callback.create = app_create;
	event_callback.terminate = app_terminate;
	event_callback.pause = app_pause;
	event_callback.resume = app_resume;
	event_callback.app_control = app_control;

	// ~ Why declare these event handler ?
	ui_app_add_event_handler(&handlers[APP_EVENT_LOW_BATTERY], APP_EVENT_LOW_BATTERY, ui_app_low_battery, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_LOW_MEMORY], APP_EVENT_LOW_MEMORY, ui_app_low_memory, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_DEVICE_ORIENTATION_CHANGED], APP_EVENT_DEVICE_ORIENTATION_CHANGED, ui_app_orient_changed, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_LANGUAGE_CHANGED], APP_EVENT_LANGUAGE_CHANGED, ui_app_lang_changed, &ad);
	ui_app_add_event_handler(&handlers[APP_EVENT_REGION_FORMAT_CHANGED], APP_EVENT_REGION_FORMAT_CHANGED, ui_app_region_changed, &ad);
	ui_app_remove_event_handler(handlers[APP_EVENT_LOW_MEMORY]);

	ret = ui_app_main(argc, argv, &event_callback, &ad);

	int thread_return;
	pthread_join(thread_id, &thread_return);
	if (ret != APP_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "app_main() is failed. err = %d", ret);
	}

	return ret;
}

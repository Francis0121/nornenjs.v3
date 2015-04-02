/*
 * Copyright (c) 2014 Samsung Electronics Co., Ltd All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <app.h>
#include <system_settings.h>
#include <Evas_GL_GLES2_Helpers.h>
#include <efl_extension.h>

#include "nornenjs.h"
#include "nornenjs_utils.h"

EVAS_GL_GLOBAL_GLES2_DEFINE();

/* Define the cube's vertices
   Each vertex consist of x, y, z, r, g, b */
const float cube_vertices[] =
{
	/* front surface is blue */
	0.5,  0.5,  0.5,  0.0, 0.0, 1.0,
	-0.5, -0.5,  0.5, 0.0, 0.0, 1.0,
	0.5, -0.5,  0.5,  0.0, 0.0, 1.0,
	0.5,  0.5,  0.5,  0.0, 0.0, 1.0,
	-0.5,  0.5,  0.5, 0.0, 0.0, 1.0,
	-0.5, -0.5,  0.5, 0.0, 0.0, 1.0,
	/* left surface is green */
	-0.5,  0.5,  0.5, 0.0, 1.0, 0.0,
	-0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
	-0.5, -0.5,  0.5, 0.0, 1.0, 0.0,
	-0.5,  0.5,  0.5, 0.0, 1.0, 0.0,
	-0.5,  0.5, -0.5, 0.0, 1.0, 0.0,
	-0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
	/* top surface is red */
	-0.5,  0.5,  0.5, 1.0, 0.0, 0.0,
	0.5,  0.5, -0.5,  1.0, 0.0, 0.0,
	-0.5,  0.5, -0.5, 1.0, 0.0, 0.0,
	-0.5,  0.5,  0.5, 1.0, 0.0, 0.0,
	0.5,  0.5,  0.5,  1.0, 0.0, 0.0,
	0.5,  0.5, -0.5,  1.0, 0.0, 0.0,
	/* right surface is yellow */
	0.5,  0.5, -0.5,  1.0, 1.0, 0.0,
	0.5, -0.5,  0.5,  1.0, 1.0, 0.0,
	0.5, -0.5, -0.5,  1.0, 1.0, 0.0,
	0.5,  0.5, -0.5,  1.0, 1.0, 0.0,
	0.5,  0.5,  0.5,  1.0, 1.0, 0.0,
	0.5, -0.5,  0.5,  1.0, 1.0, 0.0,
	/* back surface is cyan */
	-0.5,  0.5, -0.5, 0.0, 1.0, 1.0,
	0.5, -0.5, -0.5,  0.0, 1.0, 1.0,
	-0.5, -0.5, -0.5, 0.0, 1.0, 1.0,
	-0.5,  0.5, -0.5, 0.0, 1.0, 1.0,
	0.5,  0.5, -0.5,  0.0, 1.0, 1.0,
	0.5, -0.5, -0.5,  0.0, 1.0, 1.0,
	/* bottom surface is magenta */
	-0.5, -0.5, -0.5, 1.0, 0.0, 1.0,
	0.5, -0.5,  0.5,  1.0, 0.0, 1.0,
	-0.5, -0.5,  0.5, 1.0, 0.0, 1.0,
	-0.5, -0.5, -0.5, 1.0, 0.0, 1.0,
	0.5, -0.5, -0.5,  1.0, 0.0, 1.0,
	0.5, -0.5,  0.5,  1.0, 0.0, 1.0
};

/* Vertext Shader Source */
static const char vertex_shader[] =
   "attribute vec4 vPosition;\n"
   "attribute vec3 inColor;\n"
   "uniform mat4 mvpMatrix;"
   "varying vec3 outColor;\n"
   "void main()\n"
   "{\n"
   "   outColor = inColor;\n"
   "   gl_Position = mvpMatrix * vPosition;\n"
   "}\n";

/* Fragment Shader Source */
static const char fragment_shader[] =
   "#ifdef GL_ES\n"
   "precision mediump float;\n"
   "#endif\n"
   "varying vec3 outColor;\n"
   "void main()\n"
   "{\n"
   "   gl_FragColor = vec4 ( outColor, 1.0 );\n"
   "}\n";

static void
win_back_cb(void *data, Evas_Object *obj, void *event_info)
{
	appdata_s *ad = data;
	/* Let window go to hide state. */
	elm_win_lower(ad->win);
}

static void
win_delete_request_cb(void *data , Evas_Object *obj , void *event_info)
{
	ui_app_exit();
}

static void
init_shaders(appdata_s *ad)
{
	const char *p;

	p = vertex_shader;
	ad->vtx_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(ad->vtx_shader, 1, &p, NULL);
	glCompileShader(ad->vtx_shader);

	p = fragment_shader;
	ad->fgmt_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(ad->fgmt_shader, 1, &p, NULL);
	glCompileShader(ad->fgmt_shader);

	ad->program = glCreateProgram();
	glAttachShader(ad->program, ad->vtx_shader);
	glAttachShader(ad->program, ad->fgmt_shader);
	glBindAttribLocation(ad->program, 0, "vPosition");
	glBindAttribLocation(ad->program, 1, "inColor");

	glLinkProgram(ad->program);
	glUseProgram(ad->program);
	glEnable(GL_DEPTH_TEST);
}

static void
img_pixel_cb(void *data, Evas_Object *obj)
{
	/* Define the model view projection matrix */
	float model[16], mvp[16];
	static float view[16];
	appdata_s *ad = data;

	Evas_Coord w, h;
	evas_object_image_size_get(obj, &w, &h);

	/* Set up the context and surface as the current one */
	evas_gl_make_current(ad->evasgl, ad->sfc, ad->ctx);

	/* Initialize gl stuff just one time. */
	if (ad->initialized == EINA_FALSE) {
		float aspect;
		init_shaders(ad);
		glGenBuffers(1, &ad->vbo);
		glBindBuffer(GL_ARRAY_BUFFER, ad->vbo);
		glBufferData(GL_ARRAY_BUFFER, 3 * 72 * 4, cube_vertices,GL_STATIC_DRAW);
		init_matrix(view);
		if(w > h) {
			aspect = (float)w/h;
			view_set_ortho(view, -1.0 * aspect, 1.0 * aspect, -1.0, 1.0, -1.0, 1.0);
		}
		else {
			aspect = (float)h/w;
			view_set_ortho(view, -1.0, 1.0, -1.0 * aspect,  1.0 *aspect, -1.0, 1.0);
		}
		ad->initialized = EINA_TRUE;
	}

	glViewport(0, 0, w, h);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	init_matrix(model);
	rotate_xyz(model, ad->xangle, ad->yangle, 0.0f);
	multiply_matrix(mvp, view, model);

	glUseProgram(ad->program);
	glBindBuffer(GL_ARRAY_BUFFER, ad->vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, 0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, ad->vbo);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void*)(sizeof(float)*3));
	glEnableVertexAttribArray(1);

	glUniformMatrix4fv( glGetUniformLocation(ad->program, "mvpMatrix"), 1, GL_FALSE, mvp);
	glDrawArrays(GL_TRIANGLES, 0, 36);

	glFlush();
}

static void
img_del_cb(void *data, Evas *e , Evas_Object *obj , void *event_info)
{
	appdata_s *ad = data;
	Ecore_Animator *ani = evas_object_data_get(ad->img, "ani");
	ecore_animator_del(ani);

	/* Free the gl resources when image object is deleted. */
	evas_gl_make_current(ad->evasgl, ad->sfc, ad->ctx);

	glDeleteShader(ad->vtx_shader);
	glDeleteShader(ad->fgmt_shader);
	glDeleteProgram(ad->program);
	glDeleteBuffers(1, &ad->vbo);

	evas_gl_surface_destroy(ad->evasgl, ad->sfc);
	evas_gl_context_destroy(ad->evasgl, ad->ctx);
	evas_gl_config_free(ad->cfg);

	evas_gl_free(ad->evasgl);
}

static Eina_Bool
animate_cb(void *data)
{
	Evas_Object *img = data;

	/* Animate here whenever an animation tick happens and then mark the image as
	   "dirty" meaning it needs an update next time evas renders. it will call the
	   pixel get callback then. */
	evas_object_image_pixels_dirty_set(img, EINA_TRUE);

	return ECORE_CALLBACK_RENEW;
}

static void
mouse_down_cb(void *data, Evas *e , Evas_Object *obj , void *event_info)
{
	appdata_s *ad = data;
	ad->mouse_down = EINA_TRUE;
}

static void
mouse_move_cb(void *data, Evas *e , Evas_Object *obj , void *event_info)
{
	Evas_Event_Mouse_Move *ev;
	ev = (Evas_Event_Mouse_Move *)event_info;
	appdata_s *ad = data;
	float dx = 0, dy = 0;

	if(ad->mouse_down) {
		dx = ev->cur.canvas.x - ev->prev.canvas.x;
		dy = ev->cur.canvas.y - ev->prev.canvas.y;
		ad->xangle += dy;
		ad->yangle += dx;
	}
}

static void
mouse_up_cb(void *data, Evas *e , Evas_Object *obj , void *event_info)
{
	appdata_s *ad = data;
	ad->mouse_down = EINA_FALSE;
}

static void
win_resize_cb(void *data, Evas *e , Evas_Object *obj , void *event_info)
{
	appdata_s *ad = data;

	if(ad->sfc) {
		evas_object_image_native_surface_set(ad->img, NULL);
		evas_gl_surface_destroy(ad->evasgl, ad->sfc);
		ad->sfc = NULL;
	}

	Evas_Coord w,h;
	evas_object_geometry_get(obj, NULL, NULL, &w, &h);
	evas_object_image_size_set(ad->img, w, h);
	evas_object_resize(ad->img, w, h);
	evas_object_show(ad->img);

	if(!ad->sfc) {
		Evas_Native_Surface ns;

		ad->sfc = evas_gl_surface_create(ad->evasgl, ad->cfg, w, h);
		evas_gl_native_surface_get(ad->evasgl, ad->sfc, &ns);
		evas_object_image_native_surface_set(ad->img, &ns);
		evas_object_image_pixels_dirty_set(ad->img, EINA_TRUE);
	}
}

static void
init_evasgl(appdata_s *ad)
{
	Ecore_Animator *ani;

	/* Set config of the surface for evas gl */
	ad->cfg = evas_gl_config_new();
	ad->cfg->color_format = EVAS_GL_RGB_888;
	ad->cfg->depth_bits = EVAS_GL_DEPTH_BIT_24;
	ad->cfg->stencil_bits = EVAS_GL_STENCIL_NONE;
	ad->cfg->options_bits = EVAS_GL_OPTIONS_NONE;

	/* Get the window size */
	Evas_Coord w,h;
	evas_object_geometry_get(ad->win, NULL, NULL, &w, &h);

	/* Get the evas gl handle for doing gl things */
	ad->evasgl = evas_gl_new(evas_object_evas_get(ad->win));

	/* Create a surface and context */
	ad->sfc = evas_gl_surface_create(ad->evasgl, ad->cfg, w, h);
	ad->ctx = evas_gl_context_create(ad->evasgl, NULL);

	EVAS_GL_GLOBAL_GLES2_USE(ad->evasgl, ad->ctx);

	/* Set rotation variables */
	ad->xangle = 45.0f;
	ad->yangle = 45.0f;
	ad->mouse_down = EINA_FALSE;
	ad->initialized = EINA_FALSE;

	/* Set up the image object. A filled one by default. */
	ad->img = evas_object_image_filled_add(evas_object_evas_get(ad->win));
	evas_object_event_callback_add(ad->img, EVAS_CALLBACK_DEL, img_del_cb, ad);
	evas_object_image_pixels_get_callback_set(ad->img, img_pixel_cb, ad);

	/* Add Mouse Event Callbacks */
	evas_object_event_callback_add(ad->img, EVAS_CALLBACK_MOUSE_DOWN, mouse_down_cb, ad);
	evas_object_event_callback_add(ad->img, EVAS_CALLBACK_MOUSE_UP, mouse_up_cb, ad);
	evas_object_event_callback_add(ad->img, EVAS_CALLBACK_MOUSE_MOVE, mouse_move_cb, ad);
	
	ani = ecore_animator_add(animate_cb, ad->img);
	evas_object_data_set(ad->img, "ani", ani);
}

Evas_Object *
add_win(const char *name)
{
	Evas_Object *win;

	elm_config_accel_preference_set("opengl");
	win = elm_win_util_standard_add(name, "UI Template");
	if (!win)
		return NULL;

	evas_object_show(win);

	return win;
}

static bool
app_create(void *data)
{
	/* Hook to take necessary actions before main event loop starts
	   Initialize UI resources and application's data
	   If this function returns true, the main loop of application starts
	   If this function returns false, the application is terminated. */

	appdata_s *ad;
	Evas_Object *win;

	if (!data)
		return false;

	ad = data;

	win = add_win(ad->name);
	if (!win)
		return false;

	ad->win = win;

	init_evasgl(ad);

	eext_object_event_callback_add(win, EEXT_CALLBACK_BACK, win_back_cb, ad);
	evas_object_event_callback_add(ad->win, EVAS_CALLBACK_RESIZE, win_resize_cb, ad);
	evas_object_smart_callback_add(ad->win, "delete,request", win_delete_request_cb, NULL);

	return true;
}

static void
app_control(app_control_h app_control, void *data)
{
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
	if (ret != APP_ERROR_NONE) {
		dlog_print(DLOG_ERROR, LOG_TAG, "The application failed to start, and returned %d", ret);
	}

	return ret;
}

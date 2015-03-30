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

#ifndef __nornenjs_H__
#define __nornenjs_H__

#include <dlog.h>
#include <Elementary.h>

#ifdef  LOG_TAG
#undef  LOG_TAG
#endif

#define LOG_TAG "nornenjs"

typedef struct appdata {
	const char *name;
	Evas_Object *win;

	/* GL related data here... */
	Evas_GL         *evasgl;
	Evas_GL_Context *ctx;
	Evas_GL_Surface *sfc;
	Evas_GL_Config  *cfg;
	Evas_Object     *img;

	unsigned int     program;
	unsigned int     vtx_shader;
	unsigned int     fgmt_shader;
	unsigned int     vbo;

	float            xangle;
	float            yangle;
	Eina_Bool        mouse_down : 1;
	Eina_Bool        initialized : 1;
} appdata_s;

#endif /* __nornenjs_H__ */

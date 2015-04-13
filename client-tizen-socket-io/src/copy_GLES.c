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

#include <math.h>
#include <Elementary_GL_Helpers.h>
#include <image_util.h>

#include "socket.hpp"
#include "socket_io_client.hpp"
#include "other.h"

#define LOG_TAG "socket.io.opengl"
#define SAMPLE_FILENAME "/tmp/dog.jpg"

#define ONEP  +1.0
#define ONEN  -1.0
#define ZERO   0.0
#define Z_POS_INC 0.01f

char* textBuf = NULL;
int sizeBuf;

static void set_perspective(Evas_Object *obj, float fovDegree, int w, int h, float zNear,  float zFar)
{
   ELEMENTARY_GLVIEW_USE(obj);

   glViewport(0, 0, w, h);
   float ratio = (float)w / (float)h;
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glFrustumf(-ratio, ratio, -1, 1, 1, 10);
}


void
init_gles(Evas_Object *obj)
{
	dlog_print(DLOG_VERBOSE, LOG_TAG, "init_gles");
   int w, h;
   appdata_s *ad;

   ELEMENTARY_GLVIEW_USE(obj);
   ad = evas_object_data_get(obj, APPDATA_KEY);

   glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

   glGenTextures(1, ad->tex_ids);

   /* Create and map texture 1 */
   glBindTexture(GL_TEXTURE_2D, ad->tex_ids[0]);

	glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

   glShadeModel(GL_SMOOTH);

   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);

   glEnable(GL_DEPTH_TEST);

   elm_glview_size_get(obj, &w, &h);
   set_perspective(obj, 60.0f, w, h, 1.0f, 400.0f);
}



void destroy_gles(Evas_Object *obj)
{
	dlog_print(DLOG_VERBOSE, LOG_TAG, "destroy_gles");
   appdata_s *ad;

   ELEMENTARY_GLVIEW_USE(obj);
   ad = evas_object_data_get(obj, APPDATA_KEY);

   if (ad->tex_ids[0])
   {
      glDeleteTextures(1, &(ad->tex_ids[0]));
      ad->tex_ids[0] = 0;
   }

   if (ad->tex_ids[1])
   {
      glDeleteTextures(1, &(ad->tex_ids[1]));
      ad->tex_ids[1] = 0;
   }
}

void resize_gl(Evas_Object *obj)
{
	dlog_print(DLOG_VERBOSE, LOG_TAG, "resize_gl");
   int w, h;

   elm_glview_size_get(obj, &w, &h);

   set_perspective(obj, 60.0f, w, h, 1.0f, 400.0f);
}

static void draw_cube(Evas_Object *obj)
{
   appdata_s *ad;

   ELEMENTARY_GLVIEW_USE(obj);
   ad = evas_object_data_get(obj, APPDATA_KEY);

   static const float VERTICES[] =
   {
		   -1.0f	, -1.0f, 0.0f,	// 3, Left Bottom
		   1.0f	, -1.0f, 0.0f,	// 2, Right Bottom
		   -1.0f	, 1.0f	, 0.0f, 	// 0, Left Top
		   1.0f	, 1.0f	, 0.0f		// 1, Right Top
   };

   static const float TEXTURE_COORD[] =
   {
		   0.0f, 1.0f,
		   1.0f, 1.0f,
		   0.0f, 0.0f,
		   1.0f, 0.0f,
   };


   glEnableClientState(GL_VERTEX_ARRAY);
   glVertexPointer(3, GL_FLOAT, 0, VERTICES);

   glEnableClientState(GL_TEXTURE_COORD_ARRAY);
   glTexCoordPointer(2, GL_FLOAT, 0, TEXTURE_COORD);

   glEnable(GL_TEXTURE_2D);
   glBindTexture(GL_TEXTURE_2D, ad->tex_ids[ad->current_tex_index]);

   glMatrixMode(GL_MODELVIEW);

   glLoadIdentity();
   glTranslatef(0, 0.0f, -2.0f);

   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

   glDisable(GL_TEXTURE_2D);
   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

void draw_gl(Evas_Object *obj)
{
	appdata_s *ad;
	ELEMENTARY_GLVIEW_USE(obj);
	ad = evas_object_data_get(obj, APPDATA_KEY);

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   draw_cube(obj);
	if(textBuf != NULL){

		int bufWidth = 0, bufHeight = 0;
		unsigned int decodeBufSize = 0;

		unsigned char *image = NULL;

		int error = 0;
		error = image_util_decode_jpeg_from_memory((unsigned char *)textBuf, sizeBuf, IMAGE_UTIL_COLORSPACE_RGBA8888, &image, &bufWidth, &bufHeight, &decodeBufSize);

		if(error != -2)
		{
			glBindTexture(GL_TEXTURE_2D, ad->tex_ids[0]);//ad->current_tex_index
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, bufWidth, bufHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);

			glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}

		free(image);
	}

}

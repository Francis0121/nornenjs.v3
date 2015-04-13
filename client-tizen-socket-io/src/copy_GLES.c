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
#include "socket.hpp"
#include "socket_io_client.hpp"
#include "other.h"
#define LOG_TAG_SOCKET_IO "socket.io"
#define ONEP  +1.0
#define ONEN  -1.0
#define ZERO   0.0
#define Z_POS_INC 0.01f

extern const unsigned short IMAGE_565_128_128_1[];
extern const unsigned short IMAGE_4444_128_128_1[];

static Evas_Object *obj2;

static void
set_perspective(Evas_Object *obj, float fovDegree, int w, int h, float zNear,  float zFar)
{
   ELEMENTARY_GLVIEW_USE(obj);

   glViewport(0, 0, w, h);
   float ratio = (float) w / (float)h;
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   //glFrustumf(fxdXMin, fxdXMax, fxdYMin, fxdYMax, zNear, zFar);
   glFrustumf(-ratio, ratio, -1, 1, 1, 10);
}

void
init_gles(Evas_Object *obj)
{
   int w, h;
   appdata_s *ad;
   const unsigned char *texture_ids;
  //texture_ids = (unsigned char*)texture_getter();
   dlog_print(DLOG_FATAL, LOG_TAG, "success!!!! %d %d ", texture_ids, texture_ids);
   ELEMENTARY_GLVIEW_USE(obj);
   ad = evas_object_data_get(obj, APPDATA_KEY);

   glGenTextures(1, ad->tex_ids);

   /* Create and map texture 1 */
   glBindTexture(GL_TEXTURE_2D, ad->tex_ids[0]);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 128, 128, 0, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4, IMAGE_4444_128_128_1);
  // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE,   (void*)texture_ids);

   glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   glShadeModel(GL_SMOOTH);

   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);

   glEnable(GL_DEPTH_TEST);
  // glDepthFunbeforec(GL_LESS);

   elm_glview_size_get(obj, &w, &h);
   set_perspective(obj, 60.0f, w, h, 1.0f, 400.0f);

   // obj2 = obj;//add
}
void
genTex(Evas_Object *obj){

	//char *tex;
	//appdata_s *ad;
	//char *tex = texture_getter();

	//ELEMENTARY_GLVIEW_USE(obj);
	//ad = evas_object_data_get(obj, APPDATA_KEY);

	//glBindTexture(GL_TEXTURE_2D, ad->tex_ids[1]);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, (void *)tex);
	  // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE,   (void*)texture_ids);

//	glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//   glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//   glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//   glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	//dlog_print(DLOG_FATAL, LOG_TAG, "texture_getter value c %d ", tex[1]);
	//dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "quit~~~~~~!!!!!!~~~~~~~~~~~~~~~~~");

}

void setTextureData(char* tex , Evas_Object *obj)
{
	appdata_s *ad;
	ELEMENTARY_GLVIEW_USE(obj);
	ad = evas_object_data_get(obj, APPDATA_KEY);
	dlog_print(DLOG_FATAL, LOG_TAG, "another success~~%d %d %d %d %d %d %d %d", tex[0], tex[1], tex[2], tex[3], tex[4], tex[100],tex[200],tex[9300]);
	//glGenTextures(1, ad->g_textureName);
	glBindTexture(GL_TEXTURE_2D, ad->tex_ids[0]);
	glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 128, 128, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);

	//dlog_print(DLOG_FATAL, LOG_TAG_SOCKET_IO, "quit~~~~~~!!!!!!~~~~~~~~~~~~~~~~~");

	//dlog_print(DLOG_FATAL, LOG_TAG, "success!!!! %d %d ", tex[0], tex[2]);
	//dlog_print(DLOG_FATAL, LOG_TAG, "another success %d %d %d %d %d %d %d %d", tex[0], tex[1], tex[2], tex[3], tex[4], tex[100],tex[200],tex[9300]);

//	glGenTextures(1, &g_textureName);

//	appdata_s *ad;
//	ad = evas_object_data_get(obj, APPDATA_KEY);
//   glBindTexture(GL_TEXTURE_2D, ad->tex_ids[1]);
//   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 128, 128, 0, GL_RGB, GL_UNSIGNED_BYTE, (void *)tex);
//   glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//   glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

//	glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//   glTexParameterx(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

}

void
destroy_gles(Evas_Object *obj)
{
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

void
resize_gl(Evas_Object *obj)
{
   int w, h;

   elm_glview_size_get(obj, &w, &h);

   set_perspective(obj, 60.0f, w, h, 1.0f, 400.0f);
}

//static void
//draw_cube1(Evas_Object *obj)
//{
//   static int angle = 0;
//
//   ELEMENTARY_GLVIEW_USE(obj);
//
//   static const float VERTICES[] =
//   {
//      ONEN, ONEP, ONEN, // 0
//      ONEP, ONEP, ONEN, // 1
//      ONEN, ONEN, ONEN, // 2
//      ONEP, ONEN, ONEN, // 3
//      ONEN, ONEP, ONEP, // 4
//      ONEP, ONEP, ONEP, // 5
//      ONEN, ONEN, ONEP, // 6
//      ONEP, ONEN, ONEP  // 7
//   };
//
//   static const float VERTEX_COLOR[] =
//   {
//      0, 0, 0, 0,
//      0, 0, 0, 0,
//      0, 0, 0, 0,
//      0, 0, 0, 0,
//      0, 0, 0, 0,
//      0, 0, 0, 0,
//      0, 0, 0, 0,
//      ZERO, ZERO, ZERO, ONEP
//   };
//
//   static const unsigned short INDEX_BUFFER[] =
//   {
//      0, 1, 2, 2, 1, 3,
//      1, 5, 3, 3, 5, 7,
//      5, 4, 7, 7, 4, 6,
//      4, 0, 6, 6, 0, 2,
//      4, 5, 0, 0, 5, 1,
//      2, 3, 6, 6, 3, 7
//   };
//
//   glEnableClientState(GL_VERTEX_ARRAY);
//   glVertexPointer(3, GL_FLOAT, 0, VERTICES);
//
//   glEnableClientState(GL_COLOR_ARRAY);
//   glColorPointer(4, GL_FLOAT, 0, VERTEX_COLOR);
//
//   glMatrixMode(GL_MODELVIEW);
//   glLoadIdentity();
//   glTranslatef(0, -0.7f, -8.0f);
//
//   angle = (angle + 1) % (360 * 3);
//   //glRotatef((float)angle / 3, 1.0f, 0, 0);
//  // glRotatef((float)angle, 0, 0, 1.0f);
//
//   glDrawElements(GL_TRIANGLES, 6 * (3 * 2), GL_UNSIGNED_SHORT, &INDEX_BUFFER[0]);
//
//   glDisableClientState(GL_VERTEX_ARRAY);
//   glDisableClientState(GL_COLOR_ARRAY);
//}

static void
draw_cube(Evas_Object *obj)
{
   appdata_s *ad;


   ELEMENTARY_GLVIEW_USE(obj);
   ad = evas_object_data_get(obj, APPDATA_KEY);

   static const float VERTICES[] =
   {
		   -1.0f	, -1.0f, 0.0f,	// 3, Left Bottom
		   1.0f	, -1.0f, 0.0f,	// 2, Right Bottom
		   -1.0f	, 1.0f	, 0.0f, // 0, Left Top
		   1.0f	, 1.0f	, 0.0f	// 1, Right Top
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

static void
draw_cube2(char * temp)//Evas_Object *obj
{
   appdata_s *ad;


   ELEMENTARY_GLVIEW_USE(obj2);
   ad = evas_object_data_get(obj2, APPDATA_KEY);

   static const float VERTICES[] =
   {
		   -1.0f	, -1.0f, 0.0f,	// 3, Left Bottom
		   1.0f	, -1.0f, 0.0f,	// 2, Right Bottom
		   -1.0f	, 1.0f	, 0.0f, // 0, Left Top
		   1.0f	, 1.0f	, 0.0f	// 1, Right Top
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

//void
//draw_interface(char * temp)//Evas_Object *obj
//{
//	draw_gl(temp, obj2);
//}

void
draw_gl(Evas_Object *obj)
{
	ELEMENTARY_GLVIEW_USE(obj);

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

   //dlog_print(DLOG_FATAL, LOG_TAG, "drawgl");
   //draw_cube2(obj);
   draw_cube(obj);
}

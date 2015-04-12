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
#define LOG_TAG "other"

#define APPDATA_KEY "AppData"//add

#if !defined(PACKAGE)
#define PACKAGE "org.tizen.other"
#endif

typedef struct appdata
{
   Evas_Object *table, *bg;
   Evas_Object *win;
   Evas_Object *glview;
   Ecore_Animator *anim;
   Evas_Object *conform;

   Evas_Object *label;//add
   GLuint tex_ids[2];
   int current_tex_index;
} appdata_s;
//add

#endif /* __other_H__ */



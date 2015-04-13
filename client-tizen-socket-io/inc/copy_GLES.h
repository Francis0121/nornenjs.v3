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

#ifndef __copy_GLES_H__
#define __copy_GLES_H__

void init_gles(Evas_Object *obj);
void draw_gl(Evas_Object *obj);
void resize_gl(Evas_Object *obj);
void destroy_gles(Evas_Object *obj);
void genTex(Evas_Object *obj);
#endif // __copy_GLES_H__

#ifdef __cplusplus
extern "C" {
#endif
void setTextureData(char* tex ,Evas_Object *obj );
#ifdef __cplusplus
}
#endif

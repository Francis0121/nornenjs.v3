/*
 * loginc.c
 *
 *  Created on: Jun 1, 2015
 *      Author: pi
 */

/*
 * Copyright (c) 2014 Samsung Electronics Co., Ltd All Rights Reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

#include "other.h"

static Evas_Object *
create_login_contents(Evas_Object *parent)
{
	Evas_Object *layout, *contents;

	/* Full view layout */
	layout = elm_layout_add(parent);
	evas_object_size_hint_weight_set(layout, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
	evas_object_size_hint_align_set(layout, EVAS_HINT_FILL, EVAS_HINT_FILL);

	/* Create elm_layout and set its style as nocontents/text */
	contents = elm_layout_add(layout);
	elm_layout_theme_set(contents, "layout", "nocontents", "default");
	evas_object_size_hint_weight_set(contents, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
	evas_object_size_hint_align_set(contents, EVAS_HINT_FILL, EVAS_HINT_FILL);

	elm_layout_signal_emit(contents, "text,disabled", "");
	elm_layout_signal_emit(contents, "align.center", "elm");

	elm_object_part_content_set (layout, "contents", contents);

	return layout;
}

void
login_cb(void *data, Evas_Object *obj, void *event_info)
{
	Evas_Object *contents;
	Evas_Object *nf = data;

	contents = create_login_contents(nf);
	elm_naviframe_item_push(nf, "Nocontents", NULL, NULL, contents, NULL);
}


/*
 * socket_io_client.hpp
 *
 *  Created on: Apr 7, 2015
 *      Author: hyok
 */
#include <Elementary.h>
#include <dlog.h>
static int LOOP_FLAG = 1;

#ifdef __cplusplus
extern "C"
#endif
void turn_off_flag();

#ifdef __cplusplus
extern "C"
#endif
void socket_io_client(void *object);

#ifdef __cplusplus
extern "C"
#endif
char* texture_getter();

#ifdef __cplusplus
extern "C"
#endif
void emit_quality();

#ifdef __cplusplus
extern "C"
#endif
void emit_otf(float otf);

#ifdef __cplusplus
extern "C"
#endif
void emit_otf_end(float otf);

#ifdef __cplusplus
extern "C"
#endif
void emit_brightness(float brightness);

#ifdef __cplusplus
extern "C"
#endif
void emit_rotation(float rotationX, float rotationY);

#ifdef __cplusplus
extern "C"
#endif
void emit_zoom(float positionZ);

#ifdef __cplusplus
extern "C"
#endif
void free_que();

#ifdef __cplusplus
extern "C"
#endif
void fresh_que();

#ifdef __cplusplus
extern "C"
#endif
void que_in(unsigned char * inputBuf);

#ifdef __cplusplus
extern "C"
#endif
unsigned char * que_pop();

#ifdef __cplusplus
extern "C"
#endif
unsigned char *image;
int sizeBuf;
int err;
int bufWidth;
int bufHeight;


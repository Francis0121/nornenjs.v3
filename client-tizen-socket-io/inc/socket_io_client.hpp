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
unsigned char *image;
int sizeBuf;
int err;
int bufWidth;
int bufHeight;
unsigned int decodeBufSize;
unsigned char *decoded_image;

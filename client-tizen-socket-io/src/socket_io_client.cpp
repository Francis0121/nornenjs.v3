#include "sio_client.h"
#include "socket_io_client.hpp"

#include <functional>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <string>
#include <pthread.h>
#include <unistd.h>

#include <image_util.h>


#define LOG_TAG "socket.io.client"

using namespace sio;
using namespace std;

std::mutex _lock;
std::condition_variable_any _cond;
bool connect_finish = false;

unsigned char * deprecated[6];
int count = -1;


class connection_listener
{
    sio::client &handler;

public:

    connection_listener(sio::client& h):handler(h)
    {
    	dlog_print(DLOG_VERBOSE, LOG_TAG, "Socket.io client call constructor");
    }

    void on_connected()
    {
        _lock.lock();
        _cond.notify_all();
        connect_finish = true;
        _lock.unlock();
        dlog_print(DLOG_VERBOSE, LOG_TAG, "Socket.io client connect");
    }

    void on_close(client::close_reason const& reason)
    {
        dlog_print(DLOG_VERBOSE, LOG_TAG, "Socket.io client closed");
    }

    void on_fail()
    {
        dlog_print(DLOG_VERBOSE, LOG_TAG, "Socket.io client failed");
    }
};

extern "C" {
	/**
	 * LOOP FLAG - Terminater application - while statement finish
	 */
	void turn_off_flag()
	{
		LOOP_FLAG = 0;
	}
}

extern "C" {

	void socket_io_client(void *object)
	{
		err = -9;//초기화
		dlog_print(DLOG_VERBOSE, LOG_TAG, "Socket.io function start");

		Evas_Object *evas_object = (Evas_Object *)object;
		sio::client h;
		connection_listener l(h);
		h.set_connect_listener(std::bind(&connection_listener::on_connected, &l));
		h.set_close_listener(std::bind(&connection_listener::on_close, &l,std::placeholders::_1));
		h.set_fail_listener(std::bind(&connection_listener::on_fail, &l));
		h.connect("http://112.108.40.164:5000");

		_lock.lock();
		if(!connect_finish)
		{
			_cond.wait(_lock);
		}
		_lock.unlock();
		dlog_print(DLOG_VERBOSE, LOG_TAG, "Socket.io connect finish");

		h.emit("init", "");
		dlog_print(DLOG_VERBOSE, LOG_TAG, "Emit \"init\" message\n");

		h.bind_event("stream_buf", [&](string const& name, message::ptr const& data, bool isAck,message::ptr &ack_resp){//message
			_lock.lock();

			dlog_print(DLOG_VERBOSE, LOG_TAG, "Bind event \"stream_buf\"\n");

			int size = data->get_map()["stream"]->get_map()["size"]->get_int();
			shared_ptr<const string> s_binary = data->get_map()["stream"]->get_map()["buffer"]->get_binary();
			string buffer = *s_binary;
			char* textBuf = NULL;
			textBuf = (char *)buffer.c_str();

			err = -9;
			err = image_util_decode_jpeg_from_memory((unsigned char *)textBuf, sizeBuf, IMAGE_UTIL_COLORSPACE_RGBA8888, &image, &bufWidth, &bufHeight, &decodeBufSize);
			sizeBuf = size;

			//error이면 memory 생성되지 않는가?
			//그러면 ++count를 if문 안으로 넣어 줘야 함. 그래도 생성된다면 여기에 있는게 맞고.
			deprecated[++count] = image;

			if(err == 0)//IMAGE_UTIL_ERROR_NONE != error
			{
				if(count == 5)
				{
					for(int i = 0; i < 5; i++)
					{
						free(deprecated[i]);
						deprecated[i] = NULL;
					}
					deprecated[0] = deprecated[5];
					count = -1;
				}

			}

			_lock.unlock();
		});

		dlog_print(DLOG_VERBOSE, LOG_TAG, "Bind event listener\n");

		while(LOOP_FLAG){}

		dlog_print(DLOG_VERBOSE, LOG_TAG, "Socket.io function close");


	}
}

extern "C" {
	void free_que()
	{
		for(int i = 0; i < 6; i++)//or count
		{
			if(deprecated[i])
			{
				free(deprecated[i]);
				deprecated[i] = NULL;
			}
		}
	}
}

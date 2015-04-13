#include "sio_client.h"
#include "socket_io_client.hpp"

#include <functional>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <string>
#include <pthread.h>
#include <unistd.h>

#define LOG_TAG "socket.io.client"

using namespace sio;
using namespace std;

extern "C" void setTextureData(char * tex,  Evas_Object *obj);
extern "C" void draw_interface(char * temp);

std::mutex _lock;
std::condition_variable_any _cond;
bool connect_finish = false;

char *chTexture;

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
			chTexture = (char *)buffer.c_str();
			setTextureData(chTexture, evas_object);

			dlog_print(DLOG_VERBOSE, LOG_TAG, "Buffer size : %d", size);
			dlog_print(DLOG_VERBOSE, LOG_TAG, "Texture address : %d", &chTexture);

			_lock.unlock();
		});

		dlog_print(DLOG_VERBOSE, LOG_TAG, "Bind event listener\n");

		while(LOOP_FLAG){

		}

		dlog_print(DLOG_VERBOSE, LOG_TAG, "Socket.io function close");


	}
}

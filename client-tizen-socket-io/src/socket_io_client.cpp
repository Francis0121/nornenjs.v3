#include "sio_client.h"
#include "socket_io_client.hpp"

#define HIGHLIGHT(__O__) std::cout<<"\e[1;31m"<<__O__<<"\e[0m"<<std::endl
#define EM(__O__) std::cout<<"\e[1;30;1m"<<__O__<<"\e[0m"<<std::endl
#include <functional>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <string>

#include <pthread.h>
#include <unistd.h>

#define LOG_TAG "socket.io"

using namespace sio;
using namespace std;

std::mutex _lock;

std::condition_variable_any _cond;
bool connect_finish = false;
int participants = 0;

class connection_listener
{
    sio::client &handler;

public:

    connection_listener(sio::client& h):
    handler(h)
    {
    }


    void on_connected()
    {
        _lock.lock();
        _cond.notify_all();
        connect_finish = true;
        _lock.unlock();
    }
    void on_close(client::close_reason const& reason)
    {
        std::cout<<"sio closed "<<std::endl;
        dlog_print(DLOG_FATAL, LOG_TAG, "sio closed");
        //exit(0);
    }

    void on_fail()
    {
        std::cout<<"sio failed "<<std::endl;
        dlog_print(DLOG_FATAL, LOG_TAG, "sio closed");
        //exit(0);
    }
};
extern "C" {
	void turn_off_flag()
	{
		LOOP_FLAG = 0;
	}
}
extern "C" {

	void socket_io_client()
	{


		sio::client h;
		connection_listener l(h);
		dlog_print(DLOG_FATAL, LOG_TAG, "Connect Start");
		h.set_connect_listener(std::bind(&connection_listener::on_connected, &l));
		dlog_print(DLOG_FATAL, LOG_TAG, "Set ConnectListener");

		h.set_close_listener(std::bind(&connection_listener::on_close, &l,std::placeholders::_1));
		dlog_print(DLOG_FATAL, LOG_TAG, "Set ClosetListener");

		h.set_fail_listener(std::bind(&connection_listener::on_fail, &l));
		dlog_print(DLOG_FATAL, LOG_TAG, "Set FaileListener");

		h.connect("http://112.108.40.164:5000");
		dlog_print(DLOG_FATAL, LOG_TAG, "Connect");

		_lock.lock();
		dlog_print(DLOG_FATAL, LOG_TAG, "Lock");
		if(!connect_finish)
		{
			dlog_print(DLOG_FATAL, LOG_TAG, "!!!");
			_cond.wait(_lock);
		}
		dlog_print(DLOG_FATAL, LOG_TAG, "unlock");
		_lock.unlock();

		dlog_print(DLOG_FATAL, LOG_TAG, "emit connectMessage");
		h.emit("connectMessage", "{\"project\":\"rapidjson\",\"stars\":10}");

		dlog_print(DLOG_FATAL, LOG_TAG, "bind connectMessage");
		h.bind_event("connectMessage", [&](string const& name, message::ptr const& data, bool isAck,message::ptr &ack_resp){
			_lock.lock();

			unsigned int pid = (unsigned) getpid();
			dlog_print(DLOG_FATAL, LOG_TAG, "bind_event [connectMessage] %u", pid);

			string error = data->get_map()["error"]->get_string();
			dlog_print(DLOG_FATAL, LOG_TAG, "connectMessage :: %s", error.c_str());

			_lock.unlock();
	    });

		dlog_print(DLOG_FATAL, LOG_TAG, "1");

		dlog_print(DLOG_FATAL, LOG_TAG, "2");

		dlog_print(DLOG_FATAL, LOG_TAG, "3");
		/*
		h.bind_event("test1", [&](string const& name, message::ptr const& data, bool isAck,message::ptr &ack_resp){
			_lock.lock();

			unsigned int pid = (unsigned) getpid();
			dlog_print(DLOG_FATAL, LOG_TAG, "bind_event [test1] %u", pid);

			//double user = data->get_map()["message"]->get_double();
			//dlog_print(DLOG_FATAL, LOG_TAG, "test_1 :: %f", user);

			string user = data->get_map()["message"]->get_string();
			dlog_print(DLOG_FATAL, LOG_TAG, "test_1 :: %s", user.c_str());

			_lock.unlock();
	    });*/

		dlog_print(DLOG_FATAL, LOG_TAG, "emit connectMessage");
		h.emit("init", "");

		h.bind_event("stream", [&](string const& name, message::ptr const& data, bool isAck,message::ptr &ack_resp){
					_lock.lock();

					unsigned int pid = (unsigned) getpid();
					dlog_print(DLOG_FATAL, LOG_TAG, "bind_event [test1] %u", pid);
					message::flag f = data->get_flag();
					dlog_print(DLOG_FATAL, LOG_TAG, "before switch");

					switch(f)
					    {
					    case message::flag_integer:
					    	 dlog_print(DLOG_FATAL, LOG_TAG, "flag_integer");
					        break;
					    case message::flag_double:
					    	 dlog_print(DLOG_FATAL, LOG_TAG, "flag_double");
					        break;
					    case message::flag_string:
					    	 dlog_print(DLOG_FATAL, LOG_TAG, "flag_string");
					        break;
					    case message::flag_binary:
					    	 dlog_print(DLOG_FATAL, LOG_TAG, "flag_binary");
					    	 break;
					    case message::flag_array:
							 dlog_print(DLOG_FATAL, LOG_TAG, "flag_array");
							 break;
					    case message::flag_object:
							 dlog_print(DLOG_FATAL, LOG_TAG, "flag_object");
							 break;
					    default:
					        printf("??\n");
					    }
					dlog_print(DLOG_FATAL, LOG_TAG, "test_1 :: %s", "test~~");

					/*
					vector<shared_ptr<const string>> s_binary;
					s_binary.push_back(data->get_binary());

					for(const auto&  p : s_binary)
					{
						string user;

					}*/

					//dlog_print(DLOG_FATAL, LOG_TAG, "test_1 :: %s", user.c_str());

					//string user = data->get_map()["message"]->get_string();
					//shared_ptr<const string> s_binary = data->get_map()["message"]->get_binary();
					//소멸주의

					//string user = data->get_map()["message"]->get_string();
					//dlog_print(DLOG_FATAL, LOG_TAG, "test_1 :: %s", user.c_str());
					//int user = data->get_map()["message"]->get_int();
					//dlog_print(DLOG_FATAL, LOG_TAG, "test_1 :: %d", user);

					_lock.unlock();
			    });

		dlog_print(DLOG_FATAL, LOG_TAG, "4");

		dlog_print(DLOG_FATAL, LOG_TAG, "5");

		dlog_print(DLOG_FATAL, LOG_TAG, "6");

		unsigned int pidThread = (unsigned) getpid();
		dlog_print(DLOG_FATAL, LOG_TAG, "close %u", pidThread);

		while(LOOP_FLAG){
		}

		dlog_print(DLOG_FATAL, LOG_TAG, "close");

//		1. --
//		pthread_exit(0);

// 		2. --
//		h.sync_close();
//		h.clear_con_listeners();

	}
}

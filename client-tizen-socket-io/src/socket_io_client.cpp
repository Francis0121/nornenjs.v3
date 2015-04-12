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
#include <bitset>

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


		dlog_print(DLOG_FATAL, LOG_TAG, "emit connectMessage");
		h.emit("init", "");

		h.bind_event("stream", [&](string const& name, message::ptr const& data, bool isAck,message::ptr &ack_resp){//message
					_lock.lock();

					unsigned int pid = (unsigned) getpid();
					dlog_print(DLOG_FATAL, LOG_TAG, "bind_event [test1] %u", pid);


					vector<message::ptr> arr;
					arr = data->get_map()["stream"]->get_vector();
					dlog_print(DLOG_FATAL, LOG_TAG, "arr.size :  %d", arr.size());

					int count = 0;

					int * texture = new int[arr.size()];

					for(const auto&  p : arr)
					{
						texture[count++] = p->get_int();
					}

					delete[] texture;
					_lock.unlock();
			    });
		h.bind_event("test2", [&](string const& name, message::ptr const& data, bool isAck,message::ptr &ack_resp){//message
							_lock.lock();

							unsigned int pid = (unsigned) getpid();
							dlog_print(DLOG_FATAL, LOG_TAG, "bind_event [test2] %u", pid);

							//int num1 = data->get_map()["stream"]->get_map()["num"]->get_int(); get size
							int d = data->get_map()["stream"]->get_map()["size"]->get_int();
							dlog_print(DLOG_FATAL, LOG_TAG, "key size : %d", d);

							shared_ptr<const string> s_binary = data->get_map()["stream"]->get_map()["buf"]->get_binary();
							string user = *s_binary;

							//user.c_str() == char * 로 byte가 들어잇음. opengl에서 참조하면 됨.

							//dlog_print(DLOG_FATAL, LOG_TAG, "success %d %d %d %d %d %d %d %d", user.c_str()[0], user.c_str()[1], user.c_str()[2], user.c_str()[3], user.c_str()[4], user.c_str()[100],user.c_str()[200],user.c_str()[9300]);

							_lock.unlock();
					    });


		unsigned int pidThread = (unsigned) getpid();
		dlog_print(DLOG_FATAL, LOG_TAG, "close %u", pidThread);

		while(LOOP_FLAG){
		}

		dlog_print(DLOG_FATAL, LOG_TAG, "close");


	}
}

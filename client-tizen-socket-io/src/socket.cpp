/*
 * socket.cpp
 *
 *  Created on: Apr 7, 2015
 *      Author: hyok
 */
#include "socket.hpp"
#include <iostream>
#include "sio_client.h"

// ~ Include
// ~ Rapidjson
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
// ~ Websocketpp
//#include <websocketpp/config/asio_no_tls.hpp>
//#include <websocketpp/server.hpp>

// ~ Type
// ~ Rapidjson
using namespace rapidjson;
// ~ Websocketpp
//typedef websocketpp::server<websocketpp::config::asio> server;


extern "C" {
	char* rapidjson_test() {
		// 1. Parse a JSON string into DOM.
		const char* json = "{\"project\":\"rapidjson\",\"stars\":10}";
		Document d;
		d.Parse(json);

		// 2. Modify it by DOM.
		Value& s = d["stars"];
		s.SetInt(s.GetInt() + 1);

		// 3. Stringify the DOM
		StringBuffer buffer;
		Writer<StringBuffer> writer(buffer);
		d.Accept(writer);

		// Output {"project":"rapidjson","stars":11}
		std::cout << buffer.GetString() << std::endl;
		dlog_print(DLOG_FATAL, "socket.c", "Rapidson Test::%s", buffer.GetString());

		return "Rapdjson include success";
	}
}

//void on_message(websocketpp::connection_hdl, server::message_ptr msg) {
//	std::cout << msg->get_payload() << std::endl;
//	//dlog_print(DLOG_FATAL, "socket.c", "Websocketpp Message::%s", msg->get_payload());
//}
//
//extern "C" {
//
//	char * websocket_test() {
//
//		dlog_print(DLOG_FATAL, "socket.c", "Websocketpp Test::");
//
//		server print_server;
//
//		print_server.set_message_handler(&on_message);
//
//		print_server.init_asio();
//		print_server.listen(9002);
//		print_server.start_accept();
//
//		dlog_print(DLOG_FATAL, "socket.c", "Websocketpp Accept::");
//
//		return "Websocketpp include success";
//	}
//}

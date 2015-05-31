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


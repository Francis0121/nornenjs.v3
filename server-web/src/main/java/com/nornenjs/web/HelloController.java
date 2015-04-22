package com.nornenjs.web;

import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

@Controller
public class HelloController {
	
	@RequestMapping(value = "/", method = RequestMethod.GET)
	public String printWelcome(ModelMap model) {
		model.addAttribute("message", "Hello world!");
		return "hello";
	}

	@RequestMapping(value ="/signIn", method = RequestMethod.GET)
	public String indexPage() {
		return "user/signIn";
	}

	@RequestMapping(value = "/noPermission", method = RequestMethod.GET)
	public String permissionPage() {
		return "user/noPermission";
	}
}
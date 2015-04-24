package com.nornenjs.web.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

/**
 * Created by Francis on 2015-04-22.
 */
@Controller
public class UserController {
    
    @RequestMapping(value ={"/", "/signIn"}, method = RequestMethod.GET)
    public String signInPage() {
        return "user/signIn";
    }

    @RequestMapping(value = "/noPermission", method = RequestMethod.GET)
    public String permissionPage() {
        return "user/noPermission";
    }
    
}

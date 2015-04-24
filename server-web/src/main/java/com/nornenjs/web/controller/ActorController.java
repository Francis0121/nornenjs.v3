package com.nornenjs.web.controller;

import com.nornenjs.web.actor.Actor;
import com.nornenjs.web.actor.ActorInfo;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

/**
 * Created by Francis on 2015-04-22.
 */
@Controller
public class ActorController {
    
    @RequestMapping(value ={"/", "/signIn"}, method = RequestMethod.GET)
    public String signInPage() {
        return "user/signIn";
    }

    @RequestMapping(value = "/noPermission", method = RequestMethod.GET)
    public String permissionPage() {
        return "user/noPermission";
    }
    
    @RequestMapping(value = "/join", method = RequestMethod.GET)
    public String joinPage(Model model){
        model.addAttribute("actor", new Actor());
        model.addAttribute("actorInfo", new ActorInfo());
        return "user/join";
    }
    
}
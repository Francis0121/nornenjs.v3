package com.nornenjs.web.controller;

import com.nornenjs.web.actor.Actor;
import com.nornenjs.web.actor.ActorInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
    
    private static Logger logger = LoggerFactory.getLogger(ActorController.class);
    
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
        model.addAttribute("actorInfo", new ActorInfo());
        return "user/join";
    }
    
    @RequestMapping(value = "/join", method = RequestMethod.POST)
    public String joinPost(@ModelAttribute ActorInfo actorInfo){
        logger.debug(actorInfo.toString());
        return "redirect:/";
    }
    
    @RequestMapping(value = "/forgotPassword", method = RequestMethod.GET)
    public String forgotPasswordPage(Model model){
        model.addAttribute("actorInfo", new ActorInfo());
        return "user/forgotPassword";
    }
    
    @RequestMapping(value = "/forgotPassword", method = RequestMethod.POST)
    public String forgotPassword(@ModelAttribute ActorInfo actorInfo){
        logger.debug(actorInfo.toString());
        return "redirect:/forgotPassword";
    }
    
    @RequestMapping(value = "/myInfo", method = RequestMethod.GET)
    public String myInfoPage(Model model){
        model.addAttribute("actor", new ActorInfo(new Actor("username", "qwertyuijhgfd23456789@!@", true), "myemail@nornenjs.com", "성근", "김", "2015-04-30", "2015-04-31"));
        return "user/myInfo";
    }

    @RequestMapping(value = "/setting", method = RequestMethod.GET)
    public String settingPage() {
        return "user/setting";
    }
    
}
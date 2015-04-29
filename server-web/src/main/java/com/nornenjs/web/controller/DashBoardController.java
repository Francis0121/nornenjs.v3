package com.nornenjs.web.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

/**
 * Created by Francis on 2015-04-27.
 */
@Controller
@PreAuthorize("hasRole('ROLE_DOCTOR')")
public class DashBoardController {
    
    private static Logger logger = LoggerFactory.getLogger(DashBoardController.class);
    
    @RequestMapping(value = "/dashboard", method = RequestMethod.GET)
    public String dashBoardPage(){
        return "dashboard";
    }
}

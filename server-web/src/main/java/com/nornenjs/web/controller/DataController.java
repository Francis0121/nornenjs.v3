package com.nornenjs.web.controller;

import com.nornenjs.web.data.Data;
import com.nornenjs.web.data.DataService;
import com.nornenjs.web.data.MultipartFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.User;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.context.request.WebRequest;

/**
 * Created by Francis on 2015-04-30.
 */
@Controller
@PreAuthorize("hasRole('ROLE_DOCTOR')")
@RequestMapping(value = "/data")
public class DataController {
    
    private static Logger logger = LoggerFactory.getLogger(DataController.class);
    
    @Autowired
    private DataService dataService;

    @ResponseBody
    @RequestMapping(value = "/upload", method = RequestMethod.POST)
    public Integer upload(@ModelAttribute MultipartFile multipartFile, WebRequest request){
        if(multipartFile.getFiledata() == null || multipartFile.getFiledata().getSize() == 0){
            return -1;
        }
        User user = (User) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        String username = user.getUsername();
        logger.debug(username);
        Data data = dataService.uploadData(multipartFile.getFiledata(), username);
        logger.debug(data.toString());
        
        return data.getPn();
    }
    
}

package com.nornenjs.web.controller;

import com.nornenjs.web.volume.VolumeFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

/**
 * Created by Francis on 2015-04-26.
 */
@Controller
@RequestMapping(value = "/volume")
public class VolumeController {
    
    private static Logger logger = LoggerFactory.getLogger(VolumeController.class);
    
    @RequestMapping(value = {"", "/list"}, method = RequestMethod.GET)
    public String listPage(@ModelAttribute VolumeFilter volumeFilter){
        return "volume/list";
    }
}

package com.nornenjs.web.controller;

import com.nornenjs.web.data.Data;
import com.nornenjs.web.data.DataService;
import com.nornenjs.web.data.MultipartFile;
import com.nornenjs.web.util.Publisher;
import com.nornenjs.web.volume.Volume;
import com.nornenjs.web.volume.thumbnail.Thumbnail;
import com.nornenjs.web.volume.thumbnail.ThumbnailOption;
import com.nornenjs.web.volume.thumbnail.ThumbnailService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.User;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.context.request.WebRequest;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
    @Autowired
    private ThumbnailService thumbnailService;
    @Autowired
    private Publisher publisher;

    @ResponseBody
    @RequestMapping(value = "/upload", method = RequestMethod.POST)
    public Data upload(@ModelAttribute MultipartFile multipartFile, WebRequest request){
        if(multipartFile.getFiledata() == null || multipartFile.getFiledata().getSize() == 0){
            return null;
        }
        User user = (User) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        String username = user.getUsername();
        logger.debug(username);
        Data data = dataService.uploadData(multipartFile.getFiledata(), username);
        logger.debug(data.toString());
        
        return dataService.selectOne(data.getPn());
    }

    @ResponseBody
    @RequestMapping(value= "/thumbnail", method = RequestMethod.POST)
    public Map<String, Object> publish(@RequestBody Volume volume){
        // ~ STEP 02 섬네일 생성 요청을 한다.
        logger.debug(volume.toString());

        Data data = dataService.selectOne(volume.getVolumeDataPn());
        logger.debug(data.toString());
        List<ThumbnailOption> thumbnailOptionList = thumbnailService.selectList();
        logger.debug(thumbnailOptionList.toString());

        Map<String, Object> map = new HashMap<String, Object>();
        map.put("data", data);
        map.put("volume", volume);
        map.put("thumbnailOptionList", thumbnailOptionList);
        publisher.makeThumbnail(map);

        return map;
    }

    @ResponseBody
    @RequestMapping(value="/polling/{dataPn}", method =RequestMethod.GET)
    public List<Integer> polling(@PathVariable("dataPn") Integer dataPn){
        List<Integer> thumbnailPns = dataService.selectVolumeThumbnailPn(new Thumbnail(dataPn));
        return thumbnailPns;
    }


}

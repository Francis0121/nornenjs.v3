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
import org.springframework.context.ApplicationContext;
import org.springframework.core.io.Resource;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.User;
import org.springframework.stereotype.Controller;
import org.springframework.util.FileCopyUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.context.request.WebRequest;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URLEncoder;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Francis on 2015-04-30.
 */
@Controller
@RequestMapping(value = "/data")
public class DataController {
    
    private static Logger logger = LoggerFactory.getLogger(DataController.class);
    
    @Autowired
    private DataService dataService;
    @Autowired
    private ThumbnailService thumbnailService;
    @Autowired
    private Publisher publisher;
    @Autowired
    private ApplicationContext appContext;

    @ResponseBody
    @PreAuthorize("hasRole('ROLE_DOCTOR')")
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
    @PreAuthorize("hasRole('ROLE_DOCTOR')")
    @RequestMapping(value= "/thumbnail", method = RequestMethod.POST)
    public Map<String, Object> publish(@RequestBody Volume volume){
        // ~ STEP 02 섬네일 생성 요청을 한다.
        logger.debug(volume.toString());

        Data data = dataService.selectOne(volume.getVolumeDataPn());
        logger.debug(data.toString());
        List<ThumbnailOption> thumbnailOptionList = thumbnailService.selectList();
        logger.debug(thumbnailOptionList.toString());
        User user = (User) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        String username = user.getUsername();

        Map<String, Object> map = new HashMap<String, Object>();
        map.put("username", username);
        map.put("data", data);
        map.put("volume", volume);
        map.put("thumbnailOptionList", thumbnailOptionList);
        map.put("savePaths", dataService.makeSavePaths(thumbnailOptionList.size()));
        publisher.makeThumbnail(map);

        return map;
    }

    @ResponseBody
    @PreAuthorize("hasRole('ROLE_DOCTOR')")
    @RequestMapping(value="/polling/{dataPn}", method =RequestMethod.GET)
    public List<Integer> polling(@PathVariable("dataPn") Integer dataPn){
        List<Integer> thumbnailPns = dataService.selectVolumeThumbnailPn(new Thumbnail(dataPn));
        return thumbnailPns;
    }


    @RequestMapping(value = "/thumbnail/{pn}", method = RequestMethod.GET)
    public void downloadSnapshotImage(@PathVariable(value = "pn") Integer pn,
                                      HttpServletRequest request,
                                      HttpServletResponse response) {
        Data data = dataService.selectOne(pn);
        download(request, response, data);
    }

    private void download(HttpServletRequest request, HttpServletResponse response, Data data){
        try {
            String originalName;
            java.io.File snapshotFile;
            if(data == null){
                Resource resource = appContext.getResource("/resources/image/icon/empty.png");
                snapshotFile = resource.getFile();
                originalName = "none.png";
            }else{
                snapshotFile = new File(data.getSavePath());
                originalName = data.getName();
                originalName = URLEncoder.encode(originalName, "utf-8");
            }
            InputStream is = new FileInputStream(snapshotFile);
            response.setHeader("Content-Disposition", "attachment; filename=" + originalName);
            FileCopyUtils.copy(is, response.getOutputStream());
            response.flushBuffer();
        }catch (IOException e){
            throw  new RuntimeException(e);
        }
    }
}

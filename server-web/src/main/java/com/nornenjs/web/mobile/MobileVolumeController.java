package com.nornenjs.web.mobile;

import com.nornenjs.web.actor.ActorService;
import com.nornenjs.web.volume.Volume;
import com.nornenjs.web.volume.VolumeFilter;
import com.nornenjs.web.volume.VolumeService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by pi on 15. 5. 6.
 */
@Controller
@ResponseBody
@RequestMapping("/mobile/volume/{username}")
public class MobileVolumeController {

    private static Logger logger = LoggerFactory.getLogger(MobileVolumeController.class);

    @Autowired
    private VolumeService volumeService;
    @Autowired
    private ActorService actorService;

    @RequestMapping(value = "/list", method = RequestMethod.POST)
    public Map<String, Object> listPage(@RequestBody VolumeFilter volumeFilter,
                                        @PathVariable("username") String username){
        Map<String, Object> response = new HashMap<String, Object>();
        if(!actorService.selectUsernameExist(username)) {
            response.put("volumes", null);
            return response;
        }
        logger.debug(volumeFilter.toString());
        volumeFilter.setUsername(username);

        List<Volume> volumes = volumeService.selectList(volumeFilter);
        response.put("volumes", volumes);
        response.put("volumeFilter", volumeFilter);
        logger.debug(volumes.toString());
        return response;
    }

    @RequestMapping(value = "/{volumePn}", method = RequestMethod.POST)
    public Map<String, Object> renderingPage(@PathVariable("username") String username,
                                @PathVariable Integer volumePn){
        Map<String, Object> response = new HashMap<String, Object>();
        if(!actorService.selectUsernameExist(username)) {
            response.put("volume", null);
            return response;
        }
        response.putAll(volumeService.selectVolumeInformation(volumePn));
        return response;
    }

}

package com.nornenjs.web.mobile;

import com.nornenjs.web.actor.ActorService;
import com.nornenjs.web.util.Pagination;
import com.nornenjs.web.volume.Volume;
import com.nornenjs.web.volume.VolumeFilter;
import com.nornenjs.web.volume.VolumeService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by pi on 15. 5. 6.c
 */
@Controller
@ResponseBody
@RequestMapping("/tizen/")
public class TizenVolumeController {

    private static Logger logger = LoggerFactory.getLogger(TizenVolumeController.class);

    @Autowired
    private VolumeService volumeService;

    @RequestMapping(value = "/list", method = RequestMethod.POST)
    public Map<String, Object> listPage(){
        logger.debug("TizenVolumeController-listPage");

        VolumeFilter volumeFilter = new VolumeFilter();
        Pagination pagination = volumeFilter.getPagination();
        pagination.setNumItemsPerPage(50);
        volumeFilter.setUsername("nornenjs");
        volumeFilter.setFrom("1800-01-01");
        List<Volume> volumes = volumeService.selectList(volumeFilter);

        List<TizenVolume> tizenVolumes = new ArrayList<TizenVolume>();
        for(Volume volume: volumes){
            tizenVolumes.add(new TizenVolume(volume.getPn().toString(), volume.getUsername(),
                    volume.getVolumeDataPn().toString(), volume.getTitle(),
                    volume.getWidth().toString(), volume.getHeight().toString(), volume.getDepth().toString(), volume.getInputDate()));
        }

        Map<String, Object> response = new HashMap<String, Object>();
        response.put("volumes", tizenVolumes);
        logger.debug(response.toString());
        return response;
    }

    @RequestMapping(value = "/data/{volumePn}", method = RequestMethod.POST)
    public Map<String, Object> renderingPage(@PathVariable Integer volumePn){
        logger.debug("TizenVolumeController-renderingPage");
        Map<String, Object> response = new HashMap<String, Object>();
        response.putAll(volumeService.selectVolumeInformation(volumePn));
        logger.debug(response.toString());
        return response;
    }

}

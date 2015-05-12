package com.nornenjs.web.volume;

import com.nornenjs.web.util.CRUDService;

import java.util.Map;

/**
 * Created by Francis on 2015-04-24.
 */
public interface VolumeService extends CRUDService<Volume, VolumeFilter>{
    
    Integer updateData(Volume updateVolume);

    Integer selectMaxVolume();

    Map<String, Object> selectVolumeInformation(Integer volumePn);

    Boolean selectVolumeIsExist(Volume volume);

    void deleteVolumeAndFile(Volume volume);
}

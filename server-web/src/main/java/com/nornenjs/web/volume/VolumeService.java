package com.nornenjs.web.volume;

import com.nornenjs.web.util.CRUDService;

/**
 * Created by Francis on 2015-04-24.
 */
public interface VolumeService extends CRUDService<Volume, VolumeFilter>{
    
    Integer updateData(Volume updateVolume);
}

package com.nornenjs.test.volume;

import com.nornenjs.web.volume.Volume;
import com.nornenjs.web.volume.VolumeFilter;
import com.nornenjs.web.volume.VolumeService;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

/**
 * Created by Francis on 2015-04-24.
 */
@Service
public class MockVolumeServiceImpl implements VolumeService{
    
    @Override
    public Integer updateData(Volume updateVolume) {
        return null;
    }

    @Override
    public Integer selectMaxVolume() {
        return null;
    }

    @Override
    public Map<String, Object> selectVolumeInformation(Integer volumePn) {
        return null;
    }

    @Override
    public Boolean selectVolumeIsExist(Volume volume) {
        return null;
    }

    @Override
    public void deleteVolumeAndFile(Volume volume) {

    }

    @Override
    public Volume selectOne(Integer pn) {
        return null;
    }

    @Override
    public Integer selectCount(VolumeFilter volumeFilter) {
        return null;
    }

    @Override
    public List<Volume> selectList(VolumeFilter volumeFilter) {
        return null;
    }

    @Override
    public Integer insert(Volume volume) {
        return null;
    }

    @Override
    public Integer update(Volume volume) {
        return null;
    }

    @Override
    public Integer delete(Integer pn) {
        return null;
    }
}

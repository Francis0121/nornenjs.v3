package com.nornenjs.web.volume;

import com.nornenjs.web.data.Data;
import com.nornenjs.web.data.DataService;
import com.nornenjs.web.util.Pagination;
import com.nornenjs.web.volume.thumbnail.Thumbnail;
import org.mybatis.spring.support.SqlSessionDaoSupport;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Francis on 2015-04-24.
 */
@Service
public class VolumeServiceImpl extends SqlSessionDaoSupport implements VolumeService{

    @Autowired
    private DataService dataService;
    
    @Override
    public Volume selectOne(Integer pn) {
        return getSqlSession().selectOne("volume.selectOne", pn);
    }

    @Override
    public Integer selectCount(VolumeFilter volumeFilter) {
        return getSqlSession().selectOne("volume.selectCount", volumeFilter);
    }

    @Override
    public List<Volume> selectList(VolumeFilter volumeFilter) {
        Pagination pagination = volumeFilter.getPagination();
        Integer count = selectCount(volumeFilter);
        pagination.setNumItems(count);
        if(count.equals(0)){
            return new ArrayList<Volume>();
        }
        return getSqlSession().selectList("volume.selectList", volumeFilter);
    }

    @Override
    public Integer insert(Volume volume) {
        return getSqlSession().insert("volume.insert", volume);
    }

    @Override
    public Integer update(Volume volume) {
        return getSqlSession().update("volume.update", volume);
    }
    
    @Override
    public Integer updateData(Volume updateVolume) {
        return getSqlSession().update("volume.updateData", updateVolume);
    }

    @Override
    public Integer selectMaxVolume() {
        return getSqlSession().selectOne("volume.selectMaxVolume");
    }

    @Override
    public Map<String, Object> selectVolumeInformation(Integer volumePn) {
        Map<String, Object> result = new HashMap<String, Object>();

        Volume volume = selectOne(volumePn);
        Data data = dataService.selectOne(volume.getVolumeDataPn());
        List<Integer> thumbnails = dataService.selectVolumeThumbnailPn(new Thumbnail(volume.getVolumeDataPn()));

        result.put("volume", volume);
        result.put("data", data);
        result.put("thumbnails", thumbnails);
        return result;
    }

    @Override
    public Integer delete(Integer pn) {
        return getSqlSession().delete("volume.delete", pn);
    }

}

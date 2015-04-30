package com.nornenjs.web.volume;

import com.nornenjs.web.util.Pagination;
import org.mybatis.spring.support.SqlSessionDaoSupport;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Francis on 2015-04-24.
 */
@Service
public class VolumeServiceImpl extends SqlSessionDaoSupport implements VolumeService{
    
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
    public Integer delete(Integer pn) {
        return getSqlSession().delete("volume.delete", pn);
    }

}

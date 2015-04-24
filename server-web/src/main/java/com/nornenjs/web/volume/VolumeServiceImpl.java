package com.nornenjs.web.volume;

import org.mybatis.spring.support.SqlSessionDaoSupport;
import org.springframework.stereotype.Service;

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
        return null;
    }

    @Override
    public List<Volume> selectList(VolumeFilter volumeFilter) {
        return null;
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
    public Integer delete(Integer pn) {
        return getSqlSession().delete("volume.delete", pn);
    }

}

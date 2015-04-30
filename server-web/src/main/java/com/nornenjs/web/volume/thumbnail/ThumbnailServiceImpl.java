package com.nornenjs.web.volume.thumbnail;

import org.mybatis.spring.support.SqlSessionDaoSupport;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Created by pi on 15. 4. 30.
 */
@Service
public class ThumbnailServiceImpl extends SqlSessionDaoSupport implements ThumbnailService {


    @Override
    public ThumbnailOption selectOne(Integer pn) {
        return null;
    }

    @Override
    public Integer selectCount(Object o) {
        return null;
    }

    @Override
    public List<ThumbnailOption> selectList(Object o) {
        return null;
    }

    @Override
    public Integer insert(ThumbnailOption thumbnailOption) {
        return null;
    }

    @Override
    public Integer update(ThumbnailOption thumbnailOption) {
        return null;
    }

    @Override
    public Integer delete(Integer pn) {
        return null;
    }

    @Override
    public List<ThumbnailOption> selectList() {
        return getSqlSession().selectList("thumbnail.selectList");
    }
}

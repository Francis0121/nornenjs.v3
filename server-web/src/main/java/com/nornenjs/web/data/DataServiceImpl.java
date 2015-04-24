package com.nornenjs.web.data;

import org.mybatis.spring.support.SqlSessionDaoSupport;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Created by Francis on 2015-04-24.
 */
@Service
public class DataServiceImpl extends SqlSessionDaoSupport implements DataService{
    
    // ~ Data
    
    @Override
    public Data selectOne(Integer pn) {
        return getSqlSession().selectOne("data.selectOne", pn);
    }

    @Override
    public Integer selectCount(DataFilter dataFilter) {
        return null;
    }

    @Override
    public List<Data> selectList(DataFilter dataFilter) {
        return null;
    }

    @Override
    public Integer insert(Data data) {
        return getSqlSession().insert("data.insert", data);
    }

    @Override
    public Integer update(Data data) {
        return null;
    }

    @Override
    public Integer delete(Integer pn) {
        return getSqlSession().delete("data.delete", pn);
    }

}

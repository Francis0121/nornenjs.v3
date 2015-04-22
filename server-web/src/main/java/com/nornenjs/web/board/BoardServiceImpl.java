package com.nornenjs.web.board;

import org.mybatis.spring.support.SqlSessionDaoSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Created by Francis on 2015-04-22.
 */
@Service
public class BoardServiceImpl extends SqlSessionDaoSupport implements BoardService{
    
    private static Logger logger = LoggerFactory.getLogger(BoardServiceImpl.class);

    @Override
    public Board selectOne(Integer pn) {
        return null;
    }

    @Override
    public Integer selectCount(BoardFilter boardFilter) {
        return null;
    }

    @Override
    public List<Board> selectList(BoardFilter boardFilter) {
        return null;
    }

    @Override
    public Integer insert(Board board) {
        return null;
    }

    @Override
    public Integer update(Board board) {
        return null;
    }

    @Override
    public Integer delete(Integer pn) {
        return null;
    }
}

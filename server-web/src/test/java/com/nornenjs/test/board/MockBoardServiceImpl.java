package com.nornenjs.test.board;

import com.nornenjs.web.board.Board;
import com.nornenjs.web.board.BoardFilter;
import com.nornenjs.web.board.BoardService;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Created by Francis on 2015-04-23.
 */
@Service(value = "boardServiceImpl")
public class MockBoardServiceImpl implements BoardService{

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

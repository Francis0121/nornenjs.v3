package com.nornenjs.test.board;

import com.nornenjs.web.board.Board;
import com.nornenjs.web.board.BoardFilter;
import com.nornenjs.web.board.BoardService;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Francis on 2015-04-23.
 */
@Service
public class MockBoardServiceImpl implements BoardService{

    private Integer autoIncrmentValue = 0;
    private List<Board> boards = new ArrayList<Board>();
    
    @Override
    public Board selectOne(Integer pn) {
        return boards.get(pn);
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
        board.setPn(autoIncrmentValue++);
        boards.add(board);
        return 1;
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

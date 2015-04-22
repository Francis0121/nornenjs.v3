package com.nornenjs.web.board;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.transaction.annotation.Transactional;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

/**
 * Created by Francis on 2015-04-23.
 */
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration("file:src/test/resources/spring/root-context.xml")
public class BoardServiceTest {
    
    private static Logger logger = LoggerFactory.getLogger(BoardServiceTest.class);
    
    @Autowired
    private BoardService boardService;
    
    @Test
    @Transactional
    public void 게시판_입력_테스트() throws Exception{
        Board board = new Board("제목", "내용");
        boardService.insert(board);
        logger.debug(board.toString());
        
        Integer boardPn = board.getPn();
        Board getBoard = boardService.selectOne(boardPn);
        logger.debug(getBoard.toString());

        assertThat(board.getPn(), is(getBoard.getPn()));
        assertThat(board.getTitle(), is(getBoard.getTitle()));
        assertThat(board.getContent(), is(getBoard.getContent()));
        
        assertThat(getBoard.getInsertDate(),is(getBoard.getUpdateDate()));
    }
    
}

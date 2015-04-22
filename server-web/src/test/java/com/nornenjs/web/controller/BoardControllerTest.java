package com.nornenjs.web.controller;

import com.nornenjs.web.board.Board;
import com.nornenjs.web.board.BoardService;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.context.WebApplicationContext;

import static org.hamcrest.Matchers.hasProperty;
import static org.hamcrest.Matchers.is;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Created by Francis on 2015-04-23.
 */
@RunWith(SpringJUnit4ClassRunner.class)
@WebAppConfiguration
@ContextConfiguration({"file:src/test/resources/spring/root-test-context.xml", "file:src/test/resources/spring/servlet-context.xml"})
public class BoardControllerTest {

    private static Logger logger = LoggerFactory.getLogger(BoardControllerTest.class);
    
    @Qualifier("mockBoardServiceImpl")
    @Autowired
    private BoardService boardService;

    @Autowired
    protected WebApplicationContext wac;
    private MockMvc mockMvc;

    @Before
    public void Before() {
        this.mockMvc = MockMvcBuilders.webAppContextSetup(this.wac).build();
    }

    @Test
    public void 게시판_컨트롤러_테스트() throws Exception{
        Board board = new Board("제목", "내용");
        boardService.insert(board);
        
        mockMvc.perform(get("/board/" + board.getPn()))
                .andExpect(status().isOk())
                .andExpect(view().name("board/home"))
                .andExpect(forwardedUrl("/WEB-INF/pages/board/home.jsp"))
                .andExpect(model().attributeExists("board"))
                .andExpect(model().attribute("board", hasProperty("title", is(board.getTitle()))))
                .andExpect(model().attribute("board", hasProperty("content", is(board.getContent()))));
        
    }
    
}

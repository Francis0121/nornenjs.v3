package com.nornenjs.web.controller;

import com.nornenjs.web.board.Board;
import com.nornenjs.web.board.BoardService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

/**
 * Created by Francis on 2015-04-23.
 */
@Controller
@RequestMapping("/board")
@PreAuthorize("hasRole('ROLE_DOCTOR')")
public class BoardController {
    
    private static Logger logger = LoggerFactory.getLogger(BoardController.class);
    
    @Autowired
    private BoardService boardService;
    
    @RequestMapping(value="/{boardPn}", method = RequestMethod.GET)
    public String selectOneBoard(Model model, @PathVariable("boardPn") Integer boardPn){
        Board board = boardService.selectOne(boardPn);
        logger.debug(board.toString());
        model.addAttribute("board", board);
        return "board/home";
    }
}

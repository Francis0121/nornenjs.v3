package com.nornenjs.web.controller;

import com.nornenjs.web.board.BoardService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;

/**
 * Created by Francis on 2015-04-23.
 */
@Controller
public class BoardController {
    
    @Autowired
    private BoardService boardService;
}

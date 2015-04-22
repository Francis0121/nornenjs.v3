package com.nornenjs.web.controller;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.context.WebApplicationContext;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Created by Francis on 2015-03-24.
 */
@RunWith(SpringJUnit4ClassRunner.class)
@WebAppConfiguration
@ContextConfiguration({"file:src/test/resources/spring/root-context.xml", "file:src/test/resources/spring/servlet-context.xml"})
public class UserControllerTest {

    private static Logger logger = LoggerFactory.getLogger(UserControllerTest.class);

    @Autowired
    protected WebApplicationContext wac;
    private MockMvc mockMvc;
    
    @Before
    public void Before() {
        this.mockMvc = MockMvcBuilders.webAppContextSetup(this.wac).build();
    }
    
    @Test
    public void 컨트롤러_테스트() throws Exception{
        mockMvc.perform(get("/signIn")).andExpect(status().isOk());
    }
}

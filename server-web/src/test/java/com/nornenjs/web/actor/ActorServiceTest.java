package com.nornenjs.web.actor;

import org.junit.Before;
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
public class ActorServiceTest {

    private static Logger logger = LoggerFactory.getLogger(ActorServiceTest.class);

    @Autowired
    private ActorService actorService;
    
    @Before
    public void Before(){
        logger.debug("Initial actor service");
        
    }

    @Test
    @Transactional
    public void 사용자_입력() throws Exception{
        Actor actor1 = new Actor("user01", "1q2w3e4r!", true);
        actorService.insert(actor1);

        Actor actor2 = new Actor("user02", "2w3e4r5t@", true);
        actorService.insert(actor2);
        
        Actor getActor1 = actorService.selectOne(actor1.getPn());
        Actor getActor2 = actorService.selectOne(actor2.getPn());
        
        Compare.actor(actor1, getActor1);
        Compare.actor(actor2, getActor2);
    }
    
    @Test
    @Transactional
    public void 사용자_선택() throws  Exception{
        
    }
    
    @Test
    @Transactional
    public void 사용자_수정() throws Exception{

    }
    
}

class Compare {
    
    public static void actor(Actor actor, Actor getActor){
        assertThat(actor.getPn(), is(getActor.getPn()));
        assertThat(actor.getUsername(), is(getActor.getUsername()));
        assertThat(actor.getPassword(), is(getActor.getPassword()));
        assertThat(actor.getEnabled(), is(getActor.getEnabled()));
    }
    
}
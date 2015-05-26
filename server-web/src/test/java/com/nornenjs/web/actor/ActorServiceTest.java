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
@ContextConfiguration("file:src/main/resources/spring/root-context.xml")
public class ActorServiceTest {

    private static Logger logger = LoggerFactory.getLogger(ActorServiceTest.class);

    @Autowired
    private ActorService actorService;

    private Actor actor1 = new Actor("user01", "1q2w3e4r!", true);
    private Actor actor2 = new Actor("user02", "2w3e4r5t@", true);
    
    @Before
    public void Before(){
        //logger.debug("Initial actor service");
        //actorService.insert(actor1);
        //actorService.insert(actor2);
    }

    @Test
    @Transactional
    public void 사용자_입력_및_선택() throws Exception{
        Actor getActor = actorService.selectOne(actor1.getPn());
        Compare.actor(actor1, getActor);
        getActor = actorService.selectOne(actor2.getPn());
        Compare.actor(actor2, getActor);
    }
    
    @Test
    @Transactional
    public void 사용자_수정() throws Exception{
        actor1.setUsername("userUpdate01");
        actor1.setPassword("update1q2w3e4r!");
        actor1.setEnabled(false);
        actorService.update(actor1);
        
        actor2.setUsername("userUpdate02");
        actor2.setPassword("update2w3e4r5t@");
        actor2.setEnabled(false);
        actorService.update(actor2);

        Actor getActor = actorService.selectOne(actor1.getPn());
        Compare.actor(actor1, getActor);
        getActor = actorService.selectOne(actor2.getPn());
        Compare.actor(actor2, getActor);
    }
    
    @Test
    @Transactional
    public void 사용자_삭제() throws Exception{
        actorService.delete(actor1.getPn());
        Actor getActor = actorService.selectOne(actor1.getPn());
        assertThat(null, is(getActor));
        
        getActor = actorService.selectOne(actor2.getPn());
        Compare.actor(actor2, getActor);
    }

    @Test
    public void 사용자_비밀번호_초기화()throws Exception{
        Actor actor = new Actor();
        actor.setEnabled(true);
        actor.setUsername("nornenjs");
        actor.setChangePassword("1q2w3e4r!");

        actorService.updatePassword(actor);
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
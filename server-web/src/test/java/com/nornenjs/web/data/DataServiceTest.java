package com.nornenjs.web.data;

import com.nornenjs.web.actor.Actor;
import com.nornenjs.web.actor.ActorService;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.transaction.annotation.Transactional;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

/**
 * Created by Francis on 2015-04-24.
 */
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration("file:src/test/resources/spring/root-context.xml")
public class DataServiceTest {

    private static Logger logger = LoggerFactory.getLogger(DataServiceTest.class);
    
    @Autowired
    private ActorService actorService;
    
    @Autowired
    private DataService dataService;
    
    private Data data1;
    private Data data2;
    private Actor actor1 = new Actor("user01", "1q2w3e4r!", true);
    @Before
    public void Before(){
        actorService.insert(actor1);
        
        data1 = new Data(DataType.IMAGE.getValue(), actor1.getUsername(), "name", "path");
        data2 = new Data(DataType.IMAGE.getValue(), actor1.getUsername(), "name2", "path2");
        
        dataService.insert(data1);
        dataService.insert(data2);
    }
    
    @Test
    @Transactional
    public void 파일_입력_및_조회() throws Exception{
        Data getData = dataService.selectOne(data1.getPn());
        Compare.data(data1, getData);
        getData = dataService.selectOne(data2.getPn());
        Compare.data(data2, getData);
    }
    
    @Test
    @Transactional
    public void 파일_삭제() throws Exception{
        dataService.delete(data1.getPn());
        Data getData = dataService.selectOne(data2.getPn());
        Compare.data(data2, getData);
    }
    
}

class Compare{
    
    public static void data(Data data, Data getData){
        assertThat(data.getPn(), is(getData.getPn()));
        assertThat(data.getType(), is(getData.getType()));
        assertThat(data.getName(), is(getData.getName()));
        assertThat(data.getSavePath(), is(getData.getSavePath()));
    }
    
}

package com.nornenjs.web.data;

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
    private DataService dataService;
    
    private Data data1 = new Data(DataType.IMAGE.getValue(), "name", "path");
    private Data data2 = new Data(DataType.IMAGE.getValue(), "name2", "path2");
    
    @Before
    public void Before(){
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

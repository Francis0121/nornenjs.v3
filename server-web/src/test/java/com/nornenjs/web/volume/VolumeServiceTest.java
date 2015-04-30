package com.nornenjs.web.volume;

import com.nornenjs.web.actor.Actor;
import com.nornenjs.web.actor.ActorFilter;
import com.nornenjs.web.actor.ActorService;
import com.nornenjs.web.data.Data;
import com.nornenjs.web.data.DataService;
import com.nornenjs.web.data.DataType;
import com.nornenjs.web.util.DateUtil;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

/**
 * Created by Francis on 2015-04-24.
 */
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration("file:src/test/resources/spring/root-context.xml")
public class VolumeServiceTest {
    
    private static Logger logger = LoggerFactory.getLogger(VolumeServiceTest.class);
    
    @Autowired
    private VolumeService volumeService;
    @Autowired
    private ActorService actorService;
    @Autowired
    private DataService dataService;

    private Actor actor = new Actor("username", "password", true);
    private Data data1;
    private Data data2;
    private Volume volume1;
    private Volume volume2;
    
    @Before
    public void Before() throws Exception{
        actorService.insert(actor);

        data1 = new Data(DataType.IMAGE.getValue(), actor.getUsername(), "name", "savePath");
        data2 = new Data(DataType.IMAGE.getValue(), actor.getUsername(), "name2", "savePath2");
        dataService.insert(data1);
        dataService.insert(data2);
        volume1 = new Volume(actor.getPn(), data1.getPn(), "title", 100, 100, 10);
        volume2 = new Volume(actor.getPn(), data2.getPn(), "title", 100, 100, 10);
        
        volumeService.insert(volume1);
        volumeService.insert(volume2);
    }
    
    @Test
    @Transactional
    public void 볼륨_입력_및_조회() throws Exception{
        Volume getVolume = volumeService.selectOne(volume1.getPn());
        Compare.volume(volume1, getVolume);
        getVolume = volumeService.selectOne(volume2.getPn());
        Compare.volume(volume2, getVolume);
    }
    
    @Test
    @Transactional
    public void 볼륨_삭제() throws Exception{
        volumeService.delete(volume1.getPn());
        Volume getVolume = volumeService.selectOne(volume1.getPn());
        assertThat(null, is(getVolume));
        
        getVolume = volumeService.selectOne(volume2.getPn());
        Compare.volume(volume2, getVolume);
    }
    
    @Test
    @Transactional
    public void 볼륨_제목_수정() throws Exception{
        Volume updateVolume = new Volume();
        updateVolume.setTitle("updateTitle");
        updateVolume.setPn(volume1.getPn());
        volumeService.update(updateVolume);
        
        Volume getVolume = volumeService.selectOne(volume1.getPn());
        assertThat(updateVolume.getTitle(), is(getVolume.getTitle()));
        
        getVolume = volumeService.selectOne(volume2.getPn());
        Compare.volume(volume2, getVolume);
    }
    
    @Test
    @Transactional
    public void 볼륨_데이터_수정() throws Exception{
        Data updateData = new Data(DataType.IMAGE.getValue(), actor.getUsername(), "updateName", "savePath");
        dataService.insert(updateData);
        
        Volume updateVolume = new Volume(null, updateData.getPn(), null, 200, 200, 20);
        updateVolume.setPn(volume1.getPn());
        volumeService.updateData(updateVolume);
        
        Volume getVolume = volumeService.selectOne(volume1.getPn());
        assertThat(updateVolume.getVolumeDataPn(), is(getVolume.getVolumeDataPn()));
        assertThat(updateVolume.getDepth(), is(getVolume.getDepth()));
        assertThat(updateVolume.getWidth(), is(getVolume.getWidth()));
        assertThat(updateVolume.getHeight(), is(getVolume.getHeight()));
        
        getVolume = volumeService.selectOne(volume2.getPn());
        Compare.volume(volume2, getVolume);
    }
    
    
    @Test
    @Transactional
    public void 볼륨_조회() throws Exception{
        Actor actor = new Actor("actor", "password", true);
        actorService.insert(actor);
        
        for(int i=0; i<20; i++){
            Data data = new Data(DataType.IMAGE.getValue(), actor.getUsername(),"name", "savepath");
            dataService.insert(data);
            Volume volume = new Volume(actor.getPn(), data.getPn(), "title"+i, 100, 100, 10);
            volumeService.insert(volume);
        }

        VolumeFilter volumeFilter = new VolumeFilter();
        volumeFilter.setActorPn(actor.getPn()); 
        List<Volume> volumes = volumeService.selectList(volumeFilter);
        assertThat(10, is(volumes.size()));

        volumeFilter.setFrom(DateUtil.getToday("YYYY-MM-DD"));
        volumeFilter.setTo(DateUtil.getToday("YYYY-MM-DD"));
        volumes = volumeService.selectList(volumeFilter);
        assertThat(10, is(volumes.size()));
        
        volumeFilter.setTitle("title2");
        volumes = volumeService.selectList(volumeFilter);
        assertThat(1, is(volumes.size()));
    }
    
    
}

class Compare {
    
    public static void volume(Volume volume, Volume getVolume){
        assertThat(volume.getPn(), is(getVolume.getPn()));
        assertThat(volume.getActorPn(), is(getVolume.getActorPn()));
        assertThat(volume.getVolumeDataPn(), is(getVolume.getVolumeDataPn()));
        assertThat(volume.getDepth(), is(getVolume.getDepth()));
        assertThat(volume.getWidth(), is(getVolume.getWidth()));
        assertThat(volume.getHeight(), is(getVolume.getHeight()));
        assertThat(volume.getTitle(), is(getVolume.getTitle()));
    }
    
}


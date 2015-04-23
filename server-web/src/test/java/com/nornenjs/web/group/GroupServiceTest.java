package com.nornenjs.web.group;

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

import java.util.ArrayList;
import java.util.List;

import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.is;
import static org.junit.Assert.assertThat;

/**
 * Created by Francis on 2015-04-23.
 */
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration("file:src/test/resources/spring/root-context.xml")
public class GroupServiceTest {
    
    private static Logger logger = LoggerFactory.getLogger(GroupServiceTest.class);
    
    @Autowired
    private ActorService actorService;
    
    private Actor actor = new Actor("groupActor", "1q2w3e4r!", true);
    
    @Autowired
    private GroupService groupService;
    
    private Groups groups1 = new Groups("Admin");
    private Groups groups2 = new Groups("User");
    
    private GroupAuthorities groupAuthorities1;
    private GroupAuthorities groupAuthorities2;
    
    
    @Before
    public void Before() throws  Exception{
        logger.debug("Initial group service");
        groupService.insert(groups1);
        groupService.insert(groups2);
        
        actorService.insert(actor);
    }
    
    @Test
    @Transactional
    public void 그룹_입력_및_선택() throws Exception{
        Groups getGroups = groupService.selectOne(groups1.getPn());
        Compare.groups(groups1, getGroups);
        getGroups = groupService.selectOne(groups2.getPn());
        Compare.groups(groups2, getGroups);
    }
    
    @Test
    @Transactional
    public void 그룹_권한_입력_및_조회() throws  Exception{
        groupAuthorities1 = new GroupAuthorities(groups1.getPn(), "ADMIN");
        groupService.insertGroupAuthorities(groupAuthorities1);
        
        groupAuthorities2 = new GroupAuthorities(groups1.getPn(), "USER");
        groupService.insertGroupAuthorities(groupAuthorities2);
        
        List<GroupAuthorities> list = groupService.selectGroupAuthorities(groups1.getPn());
        
        List<String> groupNames = new ArrayList<String>();
        for(GroupAuthorities groupAuthorities : list){
            groupNames.add(groupAuthorities.getAuthority());
        }
    
        assertThat(list.size(), is(2));
        assertThat(groupNames, contains(groupAuthorities1.getAuthority(), groupAuthorities2.getAuthority()));
    }
    
    @Test
    @Transactional
    public void 그룹_유저_입력_및_조회() throws  Exception{
        GroupMembers groupMembers = new GroupMembers(actor.getUsername(), groups1.getPn());
        groupService.insertGroupMembers(groupMembers);
        
        GroupMembers getGroupMembers = groupService.selectGroupMembers(groupMembers.getPn());
        Compare.groupMembers(groupMembers, getGroupMembers);
    }

}

class Compare{
    
    public static void groups(Groups groups, Groups getGroups) {
        assertThat(groups.getPn(), is(getGroups.getPn()));
        assertThat(groups.getGroupName(), is(getGroups.getGroupName()));
    }
    
    public static void groupMembers(GroupMembers groupMembers, GroupMembers getGroupMembers){
        assertThat(groupMembers.getPn(), is(getGroupMembers.getPn()));
        assertThat(groupMembers.getGroupPn(), is(getGroupMembers.getGroupPn()));
        assertThat(groupMembers.getUsername(), is(getGroupMembers.getUsername()));
    }
}


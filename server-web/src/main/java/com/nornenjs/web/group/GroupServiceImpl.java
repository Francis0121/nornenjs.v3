package com.nornenjs.web.group;

import org.mybatis.spring.support.SqlSessionDaoSupport;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Francis on 2015-04-23.
 */
@Service
public class GroupServiceImpl extends SqlSessionDaoSupport implements GroupService{
    
    // ~ Groups
    
    @Override
    public Groups selectOne(Integer pn) {
        return getSqlSession().selectOne("group.selectOne", pn);
    }

    @Override
    public Integer selectCount(GroupsFilter groupsFilter) {
        return null;
    }

    @Override
    public List<Groups> selectList(GroupsFilter groupsFilter) {
        return null;
    }

    @Override
    public Integer insert(Groups groups) {
        return getSqlSession().insert("group.insert", groups);
    }

    @Override
    public Integer update(Groups groups) {
        return null;
    }

    @Override
    public Integer delete(Integer pn) {
        return null;
    }

    // ~ GroupMembers
    @Override
    public GroupMembers selectGroupMembers(Integer pn) {
        return getSqlSession().selectOne("group.selectGroupMembers", pn);
    }

    @Override
    public Integer insertGroupMembers(GroupMembers groupMembers) {
        return getSqlSession().insert("group.insertGroupMembers", groupMembers);
    }

    // ~ GroupAuthorities


    @Override
    public List<GroupAuthorities> selectGroupAuthorities(Integer groupPn) {
        return getSqlSession().selectList("group.selectGroupAuthorities", groupPn);
    }

    @Override
    public Integer insertGroupAuthorities(GroupAuthorities groupAuthorities) {
        return getSqlSession().insert("group.insertGroupAuthorities", groupAuthorities);
    }
    
    // ~ Join

    public List<GrantedAuthority> selectGroupAuthoritiesInfo(String username){
        List<Groups> groups = getSqlSession().selectList("group.selectGroupAuthoritiesInfo", username);
        List<GrantedAuthority> grantedAuthorities = new ArrayList<GrantedAuthority>();
        for(Groups g : groups){
            grantedAuthorities.add(new SimpleGrantedAuthority(g.getGroupName()));
        }
        return grantedAuthorities;
    }
}

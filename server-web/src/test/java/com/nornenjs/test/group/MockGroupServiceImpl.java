package com.nornenjs.test.group;

import com.nornenjs.web.group.*;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Created by Francis on 2015-04-24.
 */
@Service
public class MockGroupServiceImpl implements GroupService{
    @Override
    public GroupMembers selectGroupMembers(Integer pn) {
        return null;
    }

    @Override
    public Integer insertGroupMembers(GroupMembers groupMembers) {
        return null;
    }

    @Override
    public List<GroupAuthorities> selectGroupAuthorities(Integer groupPn) {
        return null;
    }

    @Override
    public Integer insertGroupAuthorities(GroupAuthorities groupAuthorities) {
        return null;
    }

    @Override
    public List<GrantedAuthority> selectGroupAuthoritiesInfo(String username) {
        return null;
    }

    @Override
    public Groups selectOne(Integer pn) {
        return null;
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
        return null;
    }

    @Override
    public Integer update(Groups groups) {
        return null;
    }

    @Override
    public Integer delete(Integer pn) {
        return null;
    }
}

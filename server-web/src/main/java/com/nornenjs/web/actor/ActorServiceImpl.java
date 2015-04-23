package com.nornenjs.web.actor;

import org.mybatis.spring.support.SqlSessionDaoSupport;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Created by Francis on 2015-04-23.
 */
@Service
public class ActorServiceImpl extends SqlSessionDaoSupport implements ActorService, UserDetailsService{
    
    // ~ Actor service
    @Override
    public Actor selectOne(Integer pn) {
        return null;
    }

    @Override
    public Integer selectCount(ActorFilter actorFilter) {
        return null;
    }

    @Override
    public List<Actor> selectList(ActorFilter actorFilter) {
        return null;
    }

    @Override
    public Integer insert(Actor actor) {
        return null;
    }

    @Override
    public Integer update(Actor actor) {
        return null;
    }

    @Override
    public Integer delete(Integer pn) {
        return null;
    }

    // ~ UserDetailService
    
    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        return null;
    }
}

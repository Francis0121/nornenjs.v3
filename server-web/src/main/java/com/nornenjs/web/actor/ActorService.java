package com.nornenjs.web.actor;

import com.nornenjs.web.util.CRUDService;

import java.util.List;

/**
 * Created by Francis on 2015-04-23.
 */
public interface ActorService extends CRUDService<Actor, ActorFilter>{


    Integer insertAuthorities(Authorities authorities);
    
    List<Authorities> selectAuthoritieses(String username);
    
    
    
}

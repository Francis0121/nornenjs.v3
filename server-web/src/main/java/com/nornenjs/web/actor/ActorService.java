package com.nornenjs.web.actor;

import com.nornenjs.web.util.CRUDService;

/**
 * Created by Francis on 2015-04-23.
 */
public interface ActorService extends CRUDService<Actor, ActorFilter>{

    Boolean selectUsernameExist(String username);

    Boolean selectEmailExist(String email);

    Boolean createActor(ActorInfo actorInfo);

    String selectUsernameFromEmail(String username);

    String selectEmailFromUsername(String email);

    ActorInfo selectOneFromUsername(String username);

    Integer updateActorInfo(ActorInfo actorInfo);
}

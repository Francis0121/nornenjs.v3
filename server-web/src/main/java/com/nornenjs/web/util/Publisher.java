package com.nornenjs.web.util;

import com.google.gson.Gson;
import com.nornenjs.web.actor.ActorInfo;
import com.nornenjs.web.actor.ActorService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;


/**
 * Created by pi on 15. 4. 30.
 */
@Component
public class Publisher {

    private static Logger logger = LoggerFactory.getLogger(Publisher.class);

    @Autowired
    private StringRedisTemplate publishTemplate;
    @Autowired
    private ActorService actorService;


    public void testPublish() {
        try {
            Gson gson = new Gson();
            ActorInfo actorInfo = actorService.selectActorInfoFromUsername("nornenjs");
            String json = gson.toJson(actorInfo);
            logger.debug(json.toString());
            publishTemplate.convertAndSend("testPublish", json);
        } catch (Exception e) {
            e.printStackTrace();
            logger.error("Redis 서버가 작동하지 않고 있습니다. Redis 서버를 실행시켜야 합니다");
        }
    }
}

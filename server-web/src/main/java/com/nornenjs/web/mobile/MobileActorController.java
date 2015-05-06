package com.nornenjs.web.mobile;

import com.nornenjs.web.actor.Actor;
import com.nornenjs.web.actor.ActorInfo;
import com.nornenjs.web.actor.ActorService;
import com.nornenjs.web.valid.Message;
import com.nornenjs.web.valid.SignInValidator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Controller;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by pi on 15. 5. 6.
 */
@Controller
@ResponseBody
@RequestMapping("/mobile")
public class MobileActorController {

    private static Logger logger = LoggerFactory.getLogger(MobileActorController.class);

    @Autowired
    private AuthenticationManager authenticationManager;
    @Autowired
    private ActorService actorService;
    @Autowired
    private SignInValidator signInValidator;
    @Autowired
    private Message message;

    @RequestMapping(value = "/signIn", method = RequestMethod.POST)
    public Map<String, Object> mobileSignIn(@RequestBody Actor actor, BindingResult result){
        signInValidator.validate(actor, result);

        try {
            UsernamePasswordAuthenticationToken token = new UsernamePasswordAuthenticationToken(
                    actor.getUsername(), actor.getPassword());
            Authentication auth = authenticationManager.authenticate(token);
            SecurityContextHolder.getContext().setAuthentication(auth);
        } catch (AuthenticationException e) {
            logger.debug("Mobile sign in password error");
            result.rejectValue("password", "actor.password.wrong");
        }

        Map<String, Object> response = new HashMap<String, Object>();
        if(result.hasErrors()){
            response.put("result", false);
            response.put("message", message.getMessageFromResult(result));
            return response;
        }

        response.put("result", true);
        response.put("username", actor.getUsername());
        return response;
    }

    @RequestMapping(value = "/myInfo/{username}", method = RequestMethod.POST)
    public Map<String, Object> mobileMyInfo(@PathVariable("username") String username){
        Map<String, Object> response = new HashMap<String, Object>();
        if(actorService.selectUsernameExist(username)) {
            response.put("actorInfo", null);
            return response;
        }

        ActorInfo actorInfo = actorService.selectActorInfoFromUsername(username);
        response.put("actorInfo", actorInfo);
        return response;
    }

}

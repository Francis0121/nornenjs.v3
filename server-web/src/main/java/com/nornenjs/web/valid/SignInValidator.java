package com.nornenjs.web.valid;

import com.nornenjs.web.actor.Actor;
import com.nornenjs.web.actor.ActorInfo;
import com.nornenjs.web.actor.ActorService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.validation.Errors;
import org.springframework.validation.Validator;

/**
 * Created by pi on 15. 5. 6.
 */
@Component
public class SignInValidator implements Validator {

    @Autowired
    private ActorService actorService;

    @Override
    public boolean supports(Class<?> aClass) {
        return ActorInfo.class.isAssignableFrom(aClass);
    }

    @Override
    public void validate(Object object, Errors errors) {
        Actor actor = (Actor) object;

        String username = actor.getUsername();
        if(ValidationUtil.isNull(username)){
            errors.rejectValue("username", "actorInfo.actor.username.empty");
        }else{
            Boolean isEmail = !ValidationUtil.isEmail(username);
            Boolean isFormat = isEmail || !ValidationUtil.isUsername(username);
            if(!isFormat){
                errors.rejectValue("username", "actorInfo.actor.username.wrong");
            }else{
                if(isEmail){
                    if(!actorService.selectEmailExist(username)){
                        errors.rejectValue("username", "actor.username.notExist");
                    }else{
                        actor.setUsername(actorService.selectUsernameFromEmail(username));
                    }
                }else{
                    if(!actorService.selectUsernameExist(username)){
                        errors.rejectValue("username", "actor.username.notExist");
                    }
                }
            }
        }

        String password = actor.getPassword();
        if(ValidationUtil.isNull(password)){
            errors.rejectValue("password", "actorInfo.actor.password.empty");
        }
    }
}

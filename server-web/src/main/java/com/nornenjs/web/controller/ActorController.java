package com.nornenjs.web.controller;

import com.nornenjs.web.actor.Actor;
import com.nornenjs.web.actor.ActorInfo;
import com.nornenjs.web.actor.ActorService;
import com.nornenjs.web.valid.SignInValidator;
import com.nornenjs.web.valid.ValidationUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.context.HttpSessionSecurityContextRepository;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.validation.Errors;
import org.springframework.validation.Validator;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.context.request.WebRequest;


/**
 * Created by Francis on 2015-04-22.
 */
@Controller
public class ActorController {
    
    private static Logger logger = LoggerFactory.getLogger(ActorController.class);
    
    @Autowired
    private ActorService actorService;
    @Autowired
    private SignInValidator signInValidator;
    @Autowired
    private AuthenticationManager authenticationManager;
    
    @RequestMapping(value ={"/", "/signIn"}, method = RequestMethod.GET)
    public String signInPage(Model model) {
        model.addAttribute("actor", new Actor());
        return "user/signIn";
    }

    @RequestMapping(value = "/signIn", method = RequestMethod.POST)
    public String formLoginPage(WebRequest request, @ModelAttribute Actor actor, BindingResult result) {
        signInValidator.validate(actor, result);

        if (result.hasErrors()) {
            logger.debug(result.getAllErrors().toString());
            return "user/signIn";
        }

        try {
            
            UsernamePasswordAuthenticationToken token = new UsernamePasswordAuthenticationToken(
                    actor.getUsername(), actor.getPassword());
            Authentication auth = authenticationManager.authenticate(token);
            SecurityContextHolder.getContext().setAuthentication(auth);

            request.setAttribute(
                    HttpSessionSecurityContextRepository.SPRING_SECURITY_CONTEXT_KEY,
                    SecurityContextHolder.getContext(),
                    WebRequest.SCOPE_SESSION);

        } catch (AuthenticationException e) {
            logger.debug("loginPasswordError");
            result.rejectValue("password", "actor.password.wrong");
            return "user/signIn";
        }

        return "redirect:/dashboard";
    }

    @RequestMapping(value = "/noPermission", method = RequestMethod.GET)
    public String permissionPage() {
        return "user/noPermission";
    }
    
    @RequestMapping(value = "/join", method = RequestMethod.GET)
    public String joinPage(Model model){
        model.addAttribute("actorInfo", new ActorInfo());
        return "user/join";
    }
    
    @RequestMapping(value = "/join", method = RequestMethod.POST)
    public String joinPost(@ModelAttribute ActorInfo actorInfo, BindingResult result){
        new ActorInfoJoinValidator().validate(actorInfo, result);
        if(result.hasErrors()){
            return "user/join";
        }else{
            actorService.createActor(actorInfo);
            logger.debug(actorInfo.toString());
            return "redirect:/";
        }
    }
    
    @RequestMapping(value = "/forgotPassword", method = RequestMethod.GET)
    public String forgotPasswordPage(Model model, @RequestParam(required = false) String email){
        model.addAttribute("actorInfo", new ActorInfo());
        model.addAttribute("isSuccessEmail", email);
        return "user/forgotPassword";
    }
    
    @RequestMapping(value = "/forgotPassword", method = RequestMethod.POST)
    public String forgotPassword(Model model, @ModelAttribute ActorInfo actorInfo, BindingResult result){
        logger.debug(actorInfo.toString());
        new Validator(){
            @Override
            public boolean supports(Class<?> aClass) {
                return ActorInfo.class.isAssignableFrom(aClass);
            }

            @Override
            public void validate(Object object, Errors errors) {
                ActorInfo actorInfo = (ActorInfo) object;
                String email = actorInfo.getEmail();
                
                if(ValidationUtil.isNull(email)){
                    errors.rejectValue("email", "actorInfo.email.forgot.empty");
                }else{
                    Boolean isEmail = !ValidationUtil.isEmail(email);
                    Boolean isFormat = isEmail || !ValidationUtil.isUsername(email);
                    if(!isFormat){
                        errors.rejectValue("email", "actorInfo.email.forgot.wrong");
                    }else{
                        if(isEmail){
                            if(!actorService.selectEmailExist(email)){
                                errors.rejectValue("email", "actorInfo.email.forgot.notExist");
                            }
                        }else{
                            if(!actorService.selectUsernameExist(email)){
                                errors.rejectValue("email", "actorInfo.email.forgot.notExist");
                            }else{
                                actorInfo.setEmail(actorService.selectEmailFromUsername(email));
                            }
                        }
                    }
                }
            }
        }.validate(actorInfo, result);
        
        String email = actorInfo.getEmail();
        logger.debug(email);
        if(result.hasErrors()) {
            return "user/forgotPassword";
        }else{
            // TODO send mail
            logger.debug(actorInfo.toString());
            model.addAttribute("email", email);
            return "redirect:/forgotPassword";
        }
    }
    
    @PreAuthorize("hasRole('ROLE_DOCTOR')")
    @RequestMapping(value = "/myInfo/{username}", method = RequestMethod.GET)
    public String myInfoPage(Model model, @PathVariable("username") String username){
        ActorInfo actorInfo = actorService.selectActorInfoFromUsername(username);
        model.addAttribute("actorInfo", actorInfo);
        model.addAttribute("actor", new Actor(username));
        return "user/myInfo";
    }

    @PreAuthorize("hasRole('ROLE_DOCTOR')")
    @RequestMapping(value = "/myInfo/{username}", method = RequestMethod.POST)
    public String myInfoPagePost(Model model, @PathVariable("username") String username,
                                 @ModelAttribute ActorInfo actorInfo, BindingResult result){
        logger.debug(actorInfo.toString());
        new Validator(){
            @Override
            public boolean supports(Class<?> aClass) {
                return ActorInfo.class.isAssignableFrom(aClass);
            }

            @Override
            public void validate(Object object, Errors errors) {
                ActorInfo actorInfo = (ActorInfo) object;

                String lastName = actorInfo.getLastName();
                if(ValidationUtil.isNull(lastName)){
                    errors.rejectValue("lastName", "actorInfo.lastName.empty");
                }else{
                    if(ValidationUtil.isChar(lastName)){
                        errors.rejectValue("lastName", "actorInfo.lastName.wrong");
                    }
                }
                        
                String firstName = actorInfo.getFirstName();
                if(ValidationUtil.isNull(firstName)){
                    errors.rejectValue("firstName", "actorInfo.firstName.empty");
                }else{
                    if(ValidationUtil.isChar(firstName)){
                        errors.rejectValue("firstName", "actorInfo.firstName.wrong");
                    }
                }
            }
        }.validate(actorInfo, result);
        
        if(result.hasErrors()){
            model.addAttribute("actor", new Actor(username));
            return "user/myInfo";
        }else{
            actorService.updateActorInfo(actorInfo);
            return "redirect:/myInfo/"+username;
        }
    }

    @PreAuthorize("hasRole('ROLE_DOCTOR')")
    @RequestMapping(value = "/myInfo/{username}", method = RequestMethod.PUT)
    public String passwordChangePost(Model model, @PathVariable("username") String username,
                                 @ModelAttribute Actor actor, BindingResult result){
        logger.debug(actor.toString());        
        new Validator(){
            @Override
            public boolean supports(Class<?> aClass) {
                return Actor.class.isAssignableFrom(aClass);
            }

            @Override
            public void validate(Object object, Errors errors) {
                Actor actor = (Actor) object;
                
                String password = actor.getPassword();
                if(ValidationUtil.isNull(password)){
                    errors.rejectValue("password", "actor.password.empty");
                }else{
                    if(!actorService.selectIsRightPassword(actor)){
                        errors.rejectValue("password", "actor.password.wrong");
                    }
                }
                
                String changePassword = actor.getChangePassword();
                if(ValidationUtil.isNull(changePassword)){
                    errors.rejectValue("changePassword", "actor.changePassword.empty");
                }else{
                    if(ValidationUtil.isPassword(changePassword)){
                        errors.rejectValue("changePassword", "actor.changePassword.wrong");
                    }
                }
            }
            
        }.validate(actor, result);
        
        if(result.hasErrors()){
            ActorInfo actorInfo = actorService.selectActorInfoFromUsername(username);
            model.addAttribute("actorInfo", actorInfo);
            return "user/myInfo";
        }else{
            actorService.updatePassword(actor);
            return "redirect:/myInfo/"+username;
        }
    }

    @PreAuthorize("hasRole('ROLE_DOCTOR')")
    @RequestMapping(value = "/setting", method = RequestMethod.GET)
    public String settingPage() {
        
        return "user/setting";
    }


    private class ActorInfoJoinValidator implements Validator{

        @Override
        public boolean supports(Class<?> aClass) {
            return ActorInfo.class.isAssignableFrom(aClass);
        }

        @Override
        public void validate(Object object, Errors errors) {
            ActorInfo actorInfo = (ActorInfo) object;
            Actor actor = actorInfo.getActor();

            String username = actor.getUsername();
            if(ValidationUtil.isNull(username)){
                errors.rejectValue("actor.username", "actorInfo.actor.username.empty");
            }else{
                if(ValidationUtil.isUsername(username)){
                    errors.rejectValue("actor.username", "actorInfo.actor.username.wrong");
                }else{
                    if(actorService.selectUsernameExist(username)){
                        errors.rejectValue("actor.username", "actorInfo.actor.username.exist");
                    }
                }
            }

            String password = actor.getPassword();
            if(ValidationUtil.isNull(password)){
                errors.rejectValue("actor.password", "actorInfo.actor.password.empty");
            }else{
                if(ValidationUtil.isPassword(password)){
                    errors.rejectValue("actor.password", "actorInfo.actor.password.wrong");
                }
            }

            String email = actorInfo.getEmail();
            if(ValidationUtil.isNull(email)){
                errors.rejectValue("email", "actorInfo.email.empty");
            }else{
                if(ValidationUtil.isEmail(email)){
                    errors.rejectValue("email", "actorInfo.email.wrong");
                }else{
                    if(actorService.selectEmailExist(email)){
                        errors.rejectValue("email", "actorInfo.email.exist");
                    }
                }
            }

            String firstName = actorInfo.getFirstName();
            if(ValidationUtil.isNull(firstName)){
                errors.rejectValue("firstName", "actorInfo.firstName.empty");
            }else{
                if(ValidationUtil.isChar(firstName)){
                    errors.rejectValue("firstName", "actorInfo.firstName.wrong");
                }
            }

            String lastName = actorInfo.getLastName();
            if(ValidationUtil.isNull(lastName)){
                errors.rejectValue("lastName", "actorInfo.lastName.empty");
            }else{
                if(ValidationUtil.isChar(lastName)){
                    errors.rejectValue("lastName", "actorInfo.lastName.wrong");
                }
            }
        }
    }
}
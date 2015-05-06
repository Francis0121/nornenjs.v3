package com.nornenjs.web.valid;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.i18n.LocaleContextHolder;
import org.springframework.context.support.ResourceBundleMessageSource;
import org.springframework.stereotype.Component;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Created by pi on 15. 5. 6.
 */
@Component
public class Message {

    @Qualifier(value = "messageSource")
    @Autowired
    private ResourceBundleMessageSource messageSource;

    public String getValue(String key){
        Locale locale = LocaleContextHolder.getLocale();
        messageSource.setDefaultEncoding("utf-8");
        return messageSource.getMessage(key, new Object[0], locale);
    }

    public Map<String, Object> getMessageFromResult(BindingResult result){
        Map<String, Object> msg = new HashMap<String, Object>();
        for(FieldError error : result.getFieldErrors()){
            msg.put(error.getField(), getValue(error.getCode()));
        }
        return msg;
    }
}

package com.nornenjs.web.util;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

/**
 * Created by Francis on 2015-04-30.
 */
@Service
public class PropertiesService {
    
    private @Value("${data.rootUploadPath}")
    String rootUploadPath;

    public String getRootUploadPath() {
        return rootUploadPath;
    }
    
}

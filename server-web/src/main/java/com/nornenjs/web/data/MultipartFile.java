package com.nornenjs.web.data;

import org.springframework.web.multipart.commons.CommonsMultipartFile;

/**
 * Created by Francis on 2015-04-30.
 */
public class MultipartFile {
    
    private CommonsMultipartFile filedata;

    public CommonsMultipartFile getFiledata() {
        return filedata;
    }

    public void setFiledata(CommonsMultipartFile filedata) {
        this.filedata = filedata;
    }

    @Override
    public String toString() {
        return "MultipartFile{" +
                "filedata=" + filedata +
                '}';
    }
}

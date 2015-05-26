package com.nornenjs.web.util;

import java.io.File;

/**
 * Created by pi on 15. 5. 12.
 */
public class FileUtil {

    public static Boolean delFileRecursive(String path) {

        File delDir = new File(path);

        if(delDir.isDirectory()) {
            File[] allFiles = delDir.listFiles();

            for(File delAllDir : allFiles) {
                delFileRecursive(delAllDir.getAbsolutePath());
            }
        }

        return delDir.delete();
    }
}

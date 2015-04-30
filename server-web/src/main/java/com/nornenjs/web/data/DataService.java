package com.nornenjs.web.data;

import com.nornenjs.web.util.CRUDService;
import com.nornenjs.web.volume.thumbnail.Thumbnail;
import org.springframework.web.multipart.commons.CommonsMultipartFile;

import java.util.List;

/**
 * Created by Francis on 2015-04-24.
 */
public interface DataService extends CRUDService<Data, DataFilter>{
    
    String transferFile(CommonsMultipartFile multipartFile, String ... paths);

    Data uploadData(CommonsMultipartFile multipartFile, String username);

    List<Integer> selectVolumeThumbnailPn(Thumbnail thumbnail);

}

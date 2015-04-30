package com.nornenjs.web.volume.thumbnail;

import com.nornenjs.web.util.CRUDService;

import java.util.List;

/**
 * Created by pi on 15. 4. 30.
 */
public interface ThumbnailService extends CRUDService<ThumbnailOption, Object>{

    List<ThumbnailOption> selectList();
}

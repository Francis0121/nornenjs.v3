package com.nornenjs.web.data;

import com.nornenjs.web.util.PropertiesService;
import com.nornenjs.web.volume.thumbnail.Thumbnail;
import org.mybatis.spring.support.SqlSessionDaoSupport;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.commons.CommonsMultipartFile;

import java.io.File;
import java.util.List;
import java.util.UUID;

/**
 * Created by Francis on 2015-04-24.
 */
@Service
public class DataServiceImpl extends SqlSessionDaoSupport implements DataService{
    
    @Autowired
    private PropertiesService propertiesService;
    
    // ~ Data
    
    @Override
    public Data selectOne(Integer pn) {
        return getSqlSession().selectOne("data.selectOne", pn);
    }

    @Override
    public Integer selectCount(DataFilter dataFilter) {
        return null;
    }

    @Override
    public List<Data> selectList(DataFilter dataFilter) {
        return null;
    }

    @Override
    public Integer insert(Data data) {
        return getSqlSession().insert("data.insert", data);
    }

    @Override
    public Integer update(Data data) {
        return null;
    }

    @Override
    public Integer delete(Integer pn) {
        return getSqlSession().delete("data.delete", pn);
    }

    @Override
    public String transferFile(CommonsMultipartFile multipartFile, String... paths) {
        String savePath = getFreeFilePath(paths);

        File file = new File(savePath);
        try {
            multipartFile.transferTo(file);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return savePath;
    }

    private String getFreeFilePath(String[] paths) {
        String path = propertiesService.getRootUploadPath() + "/";

        for (int i = 0; i < paths.length; ++i) {
            path += paths[i] + "/";
        }

        String randomPath = null;
        while (randomPath == null) {
            UUID randomUUID = UUID.randomUUID();
            java.io.File testPath = new java.io.File(path + randomUUID);
            if (testPath.exists() == false) {
                randomPath = testPath.getPath();

                testPath = new java.io.File(path);
                if (testPath.exists() == false) {
                    testPath.mkdirs();
                }
            }
        }
        return randomPath;
    }

    @Override
    public Data uploadData(CommonsMultipartFile multipartFile, String username) {
        
        String savePath = transferFile(multipartFile, "data");
        Data data = new Data(DataType.VOLUME.getValue(), username, multipartFile.getOriginalFilename(), savePath);
        insert(data);
        logger.debug(data.toString());
        
        return data;
    }

    @Override
    public List<Integer> selectVolumeThumbnailPn(Thumbnail thumbnail) {
        return getSqlSession().selectList("data.selectVolumeThumbnailPn", thumbnail);
    }
}

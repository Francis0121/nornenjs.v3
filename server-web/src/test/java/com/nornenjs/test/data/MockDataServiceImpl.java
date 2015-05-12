package com.nornenjs.test.data;

import com.nornenjs.web.data.Data;
import com.nornenjs.web.data.DataFilter;
import com.nornenjs.web.data.DataService;
import com.nornenjs.web.volume.thumbnail.Thumbnail;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.commons.CommonsMultipartFile;

import java.util.List;

/**
 * Created by Francis on 2015-04-24.
 */
@Service
public class MockDataServiceImpl implements DataService{
    @Override
    public Data selectOne(Integer pn) {
        return null;
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
        return null;
    }

    @Override
    public Integer update(Data data) {
        return null;
    }

    @Override
    public Integer delete(Integer pn) {
        return null;
    }

    @Override
    public String transferFile(CommonsMultipartFile multipartFile, String... paths) {
        return null;
    }

    @Override
    public Data uploadData(CommonsMultipartFile multipartFile, String username) {
        return null;
    }

    @Override
    public List<Integer> selectVolumeThumbnailPn(Thumbnail thumbnail) {
        return null;
    }

    @Override
    public List<String> makeSavePaths(Integer size) {
        return null;
    }

    @Override
    public Data selectVolumeSavePath(Integer pn) {
        return null;
    }

    @Override
    public List<String> selectThumbnailSavePath(Integer pn) {
        return null;
    }
}

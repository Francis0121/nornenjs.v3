package com.nornenjs.web.util;

import java.util.List;

/**
 * Created by Francis on 2015-04-22.
 */
public interface CRUDService<Object,Filter> {

    /**
     * Select object
     *
     * @param pn
     *  Database primary key
     * @return
     *  Object
     */
    Object selectOne(Integer pn);

    /**
     * Select count objects use filter
     * @param filter
     *  Filter object extend AbstractListFilter 
     * @return
     *  count
     */
    Integer selectCount(Filter filter);

    /**
     * Select list objects use filter 
     * @param filter
     *  Filter object extend AbstractListFilter
     * @return
     *  List 
     */
    List<Object> selectList(Filter filter);

    /**
     * Insert object 
     * @param object
     *  Object 
     * @return
     *  Insert column count equal 1
     */
    Integer insert(Object object);

    /**
     * Update object 
     * @param object
     *  Object 
     * @return
     *  Update column count equal 1
     */
    Integer update(Object object);

    /**
     * Delete object use primary key 
     * @param pn
     *  Database primary key
     * @return
     *  Delete column count equal 1
     */
    Integer delete(Integer pn);
    
}

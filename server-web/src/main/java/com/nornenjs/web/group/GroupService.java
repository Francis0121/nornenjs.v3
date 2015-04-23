package com.nornenjs.web.group;

import com.nornenjs.web.util.CRUDService;

import java.util.List;

/**
 * Created by Francis on 2015-04-23.
 */
public interface GroupService extends CRUDService<Groups, GroupsFilter>{
    
    GroupMembers selectGroupMembers(Integer pn);
    
    Integer insertGroupMembers(GroupMembers groupMembers);
    
    List<GroupAuthorities> selectGroupAuthorities(Integer groupPn);
    
    Integer insertGroupAuthorities(GroupAuthorities groupAuthorities);

}

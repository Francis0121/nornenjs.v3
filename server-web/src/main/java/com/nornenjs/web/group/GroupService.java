package com.nornenjs.web.group;

import com.nornenjs.web.util.CRUDService;
import org.springframework.security.core.GrantedAuthority;

import java.util.List;

/**
 * Created by Francis on 2015-04-23.
 */
public interface GroupService extends CRUDService<Groups, GroupsFilter>{
    
    // ~ GroupMember
    GroupMembers selectGroupMembers(Integer pn);
    
    Integer insertGroupMembers(GroupMembers groupMembers);
    
    // ~ GroupAuthority
    List<GroupAuthorities> selectGroupAuthorities(Integer groupPn);
    
    Integer insertGroupAuthorities(GroupAuthorities groupAuthorities);

    // ~ Join
    List<GrantedAuthority> selectGroupAuthoritiesInfo(String username);
}

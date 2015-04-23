package com.nornenjs.web.group;

/**
 * Created by Francis on 2015-04-23.
 */
public class GroupAuthorities {
    
    private Integer groupId;
    
    private String authority;

    public GroupAuthorities() {
    }

    public GroupAuthorities(Integer groupId, String authority) {
        this.groupId = groupId;
        this.authority = authority;
    }

    public Integer getGroupId() {
        return groupId;
    }

    public void setGroupId(Integer groupId) {
        this.groupId = groupId;
    }

    public String getAuthority() {
        return authority;
    }

    public void setAuthority(String authority) {
        this.authority = authority;
    }

    @Override
    public String toString() {
        return "GroupAuthorities{" +
                "groupId=" + groupId +
                ", authority='" + authority + '\'' +
                '}';
    }
}

package com.nornenjs.web.group;

/**
 * Created by Francis on 2015-04-23.
 */
public class GroupAuthorities {
    
    private Integer groupPn;
    
    private String authority;

    public GroupAuthorities() {
    }

    public GroupAuthorities(Integer groupPn, String authority) {
        this.groupPn = groupPn;
        this.authority = authority;
    }

    public Integer getGroupPn() {
        return groupPn;
    }

    public void setGroupPn(Integer groupPn) {
        this.groupPn = groupPn;
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
                "groupPn=" + groupPn +
                ", authority='" + authority + '\'' +
                '}';
    }
}

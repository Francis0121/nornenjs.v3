package com.nornenjs.web.group;

/**
 * Created by Francis on 2015-04-23.
 */
public class GroupMembers {
    
    private Integer pn;
    
    private String username;
    
    private Integer groupPn;

    public GroupMembers() {
        
    }

    public GroupMembers(String username, Integer groupPn) {
        this.username = username;
        this.groupPn = groupPn;
    }

    public Integer getPn() {
        return pn;
    }

    public void setPn(Integer pn) {
        this.pn = pn;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public Integer getGroupPn() {
        return groupPn;
    }

    public void setGroupPn(Integer groupPn) {
        this.groupPn = groupPn;
    }

    @Override
    public String toString() {
        return "GroupMembers{" +
                "pn=" + pn +
                ", username='" + username + '\'' +
                ", groupPn=" + groupPn +
                '}';
    }
}

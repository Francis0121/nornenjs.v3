package com.nornenjs.web.group;

/**
 * Created by Francis on 2015-04-23.
 */
public class GroupMembers {
    
    private Integer id;
    
    private String username;
    
    private Integer groupId;

    public GroupMembers() {
        
    }
    
    public GroupMembers(String username, Integer groupId) {
        this.username = username;
        this.groupId = groupId;
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public Integer getGroupId() {
        return groupId;
    }

    public void setGroupId(Integer groupId) {
        this.groupId = groupId;
    }

    @Override
    public String toString() {
        return "GroupMembers{" +
                "id=" + id +
                ", username='" + username + '\'' +
                ", groupId=" + groupId +
                '}';
    }
}

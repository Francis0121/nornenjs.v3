package com.nornenjs.web.group;

/**
 * Created by Francis on 2015-04-23.
 */
public class Groups {
    
    private Integer id;
    
    private String groupName;

    public Groups() {
    }

    public Groups(String groupName) {
        this.groupName = groupName;
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getGroupName() {
        return groupName;
    }

    public void setGroupName(String groupName) {
        this.groupName = groupName;
    }

    @Override
    public String toString() {
        return "Groups{" +
                "id=" + id +
                ", groupName='" + groupName + '\'' +
                '}';
    }
}

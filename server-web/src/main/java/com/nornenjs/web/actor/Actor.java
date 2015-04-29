package com.nornenjs.web.actor;

/**
 * Created by Francis on 2015-04-23.
 */
public class Actor {
    
    private Integer pn;
    
    private String username;
    
    private String password;
    
    private String changePassword;
    
    private Boolean enabled;
    
    public Actor() {
    }

    public Actor(String username, String password, Boolean enabled) {
        this.username = username;
        this.password = password;
        this.enabled = enabled;
    }

    public Actor(String username) {
        this.username = username;
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

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public Boolean getEnabled() {
        return enabled;
    }

    public void setEnabled(Boolean enabled) {
        this.enabled = enabled;
    }

    public String getChangePassword() {
        return changePassword;
    }

    public void setChangePassword(String changePassword) {
        this.changePassword = changePassword;
    }

    @Override
    public String toString() {
        return "Actor{" +
                "pn=" + pn +
                ", username='" + username + '\'' +
                ", password='" + password + '\'' +
                ", changePassword='" + changePassword + '\'' +
                ", enabled=" + enabled +
                '}';
    }
}

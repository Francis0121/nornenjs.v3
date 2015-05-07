package com.nornenjs.android.dto;

/**
 * Created by hyok on 15. 5. 8.
 */
public class Actor {

    private Integer pn;
    private String username;
    private String password;
    private String changePassword;
    private Boolean enabled;

    public Actor() {
    }

    public Actor(String username, String password) {
        this.username = username;
        this.password = password;
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

    public String getChangePassword() {
        return changePassword;
    }

    public void setChangePassword(String changePassword) {
        this.changePassword = changePassword;
    }

    public Boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(Boolean enabled) {
        this.enabled = enabled;
    }

    @Override
    public String toString() {
        return "Actor{"+
                "pn=" + pn +
                ", username='" + username + '\'' +
                ", password='" + password + '\'' +
                ", changePassword='" + changePassword + '\'' +
                ", enabled=" + enabled +
                '}';
    }
}

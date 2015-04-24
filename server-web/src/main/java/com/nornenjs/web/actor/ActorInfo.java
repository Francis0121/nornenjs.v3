package com.nornenjs.web.actor;

/**
 * Created by Francis on 2015-04-24.
 */
public class ActorInfo {
    
    private Actor actor;
    
    private String email;
    
    private String firstName;
    
    private String lastName;
    
    private String inputDate;
    
    private String updateDate;

    public ActorInfo() {
    }

    public ActorInfo(Actor actor, String email, String firstName, String lastName, String inputDate, String updateDate) {
        this.actor = actor;
        this.email = email;
        this.firstName = firstName;
        this.lastName = lastName;
        this.inputDate = inputDate;
        this.updateDate = updateDate;
    }

    public Actor getActor() {
        return actor;
    }

    public void setActor(Actor actor) {
        this.actor = actor;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getInputDate() {
        return inputDate;
    }

    public void setInputDate(String inputDate) {
        this.inputDate = inputDate;
    }

    public String getUpdateDate() {
        return updateDate;
    }

    public void setUpdateDate(String updateDate) {
        this.updateDate = updateDate;
    }

    @Override
    public String toString() {
        return "ActorInfo{" +
                "actor=" + actor +
                ", email='" + email + '\'' +
                ", firstName='" + firstName + '\'' +
                ", lastName='" + lastName + '\'' +
                ", inputDate='" + inputDate + '\'' +
                ", updateDate='" + updateDate + '\'' +
                '}';
    }
}

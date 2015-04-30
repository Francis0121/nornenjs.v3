package com.nornenjs.web.volume;

import com.nornenjs.web.util.AbstractListFilter;
import com.nornenjs.web.util.DateUtil;

import java.util.Date;

/**
 * Created by Francis on 2015-04-24.
 */
public class VolumeFilter extends AbstractListFilter{
    
    private String username;
    
    private String title;
    
    private String from;
    
    private String to;

    public VolumeFilter() {
        this.from = "2015-01-01";
        this.to = DateUtil.getToday("YYYY-MM-DD");
    }

    public VolumeFilter(String username, String title, String from, String to) {
        this.username = username;
        this.title = title;
        this.from = from;
        this.to = to;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getFrom() {
        return from;
    }

    public void setFrom(String from) {
        this.from = from;
    }

    public String getTo() {
        return to;
    }

    public void setTo(String to) {
        this.to = to;
    }

    @Override
    public String toString() {
        return "VolumeFilter{" +
                "username='" + username + '\'' +
                ", title='" + title + '\'' +
                ", from='" + from + '\'' +
                ", to='" + to + '\'' +
                '}';
    }
}

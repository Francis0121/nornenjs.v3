package com.nornenjs.web.board;

/**
 * Created by Francis on 2015-04-23.
 */
public class Board {
    
    private Integer pn;
    
    private String title;
    
    private String content;
    
    private String insertDate;
    
    private String updateDate;

    public Board() {
    }

    public Board(String title, String content) {
        this.title = title;
        this.content = content;
    }

    public Integer getPn() {
        return pn;
    }

    public void setPn(Integer pn) {
        this.pn = pn;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public String getInsertDate() {
        return insertDate;
    }

    public void setInsertDate(String insertDate) {
        this.insertDate = insertDate;
    }

    public String getUpdateDate() {
        return updateDate;
    }

    public void setUpdateDate(String updateDate) {
        this.updateDate = updateDate;
    }

    @Override
    public String toString() {
        return "Board{" +
                "pn=" + pn +
                ", title='" + title + '\'' +
                ", content='" + content + '\'' +
                ", insertDate='" + insertDate + '\'' +
                ", updateDate='" + updateDate + '\'' +
                '}';
    }
}

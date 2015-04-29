<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>
<%@ include file="../../layout/nav.jspf" %>

<section class="layoutContentSectionWrap">
    
    <section class="layoutContentSection">

        <form:form commandName="volumeFilter" htmlEscape="true" cssClass="volumeFilterForm">
            <form:hidden path="actorPn"/>
            <ul>
                <li class="searchIcon">
                    <img src="${cp}/resources/image/icon/search.png"/>
                </li>
                <li>
                    <form:input path="title" cssClass="search" placeholder="검색"/>
                </li>
                <li class="dateIcon">
                    <img src="${cp}/resources/image/icon/calendar.png"/>
                </li>
                <li>
                    <form:input path="from" readonly="true" cssClass="date"/>
                </li>
                <li>
                    <form:input path="to" readonly="true" cssClass="date"/>
                </li>
            </ul>
            
        </form:form>
        
        <article class="volumeListArticle">
            
            <ul class="list">
                <c:forEach begin="0" end="19" varStatus="i">
                <li class="one">
                    <figure>

                        <ul class="volumeListSlider">
                            <li><img src="${cp}/resources/image/sample/01.jpg" /></li>
                            <li><img src="${cp}/resources/image/sample/02.jpg" /></li>
                        </ul>
                        
                        <figcaption>
                            <a href="${cp}/volume/${i.count}" class="name">Skull</a><span class="number">225 x 225 x 100</span><br/>
                            <span class="date">2015.04.26</span>
                        </figcaption>
                    </figure>
                </li>
                </c:forEach>
            </ul>
            
        </article>
            
    </section>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
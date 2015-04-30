<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>
<%@ include file="../../layout/nav.jspf" %>

<section class="layoutContentSectionWrap">
    
    <section class="layoutContentSection">

        <form:form commandName="volumeFilter" htmlEscape="true" cssClass="volumeFilterForm">
            <form:hidden path="username"/>
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
                <c:forEach items="${volumes}" var="volume" varStatus="loop">
                <li class="one" data-pn="<c:out value="${volume.pn}"/>">
                    <figure>

                        <ul class="volumeListSlider">
                            <li><img src="${cp}/resources/image/sample/01.jpg" /></li>
                            <li><img src="${cp}/resources/image/sample/02.jpg" /></li>
                        </ul>
                        
                        <figcaption>
                            <a href="${cp}/volume/${loop.count}" class="name"><c:out value="${volume.title}"/></a> <a href="${cp}/volume/page/${volume.pn}" class="volumeUpdateBtn">수정</a><br/>
                            <span class="number"><c:out value="${volume.width}"/> x <c:out value="${volume.height}"/> x <c:out value="${volume.depth}"/></span> <span class="date">[ 2015.04.26 ]</span>
                        </figcaption>
                    </figure>
                </li>
                </c:forEach>
            </ul>
            
        </article>
            
    </section>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
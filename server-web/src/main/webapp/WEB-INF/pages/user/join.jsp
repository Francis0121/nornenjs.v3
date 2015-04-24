<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>

<section class="layoutFullSection">
    
    <article class="layoutJoinArticle">
        
        <header>
            <h1>Join Nornenjs</h1>
        </header>
        
        <form:form commandName="actorInfo"  action="${cp}/join" method="post" htmlEscape="true" cssClass="actorJoinForm">
            <ul>
                <lI>
                    <form:label path="actor.username">아이디</form:label>
                    <form:input path="actor.username"/>
                </lI>
                <li>
                    <form:label path="email">이메일주소</form:label>
                    <form:input path="email"/>
                </li>
                <li>
                    <form:label path="actor.password">비밀번호</form:label>
                    <form:input path="actor.password"/>
                </li>
                <li>
                    <form:label path="lastName">성</form:label>
                    <form:input path="lastName"/>
                </li>
                <li>
                    <form:label path="firstName">이름</form:label>
                    <form:input path="firstName"/>
                </li>
                <li>
                    <button type="submit">회원가입</button>
                </li>
            </ul>
        </form:form>
        
    </article>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
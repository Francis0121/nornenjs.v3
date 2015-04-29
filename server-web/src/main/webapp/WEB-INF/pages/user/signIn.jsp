<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>
<section class="layoutFullSection">
    
    <article class="layoutSignInArticle">
        
        <header class="actorHeader">
            <a href="${cp}/"><h1>Nornen<span class="colorOrange">js</span></h1></a>
            <h2>Volume Rendering system</h2>
        </header>
        
        <form:form commandName="actor" action="${cp}/signIn" method="post" htmlEscape="true" cssClass="actorForm">
            <sec:authorize access="isAnonymous()">
            <ul>
                <li>
                    <form:input path="username" placeholder="아이디 또는 이메일" maxlength="200" cssClass="text"/>
                    <form:errors path="username" cssClass="error"/>
                </li>
                <li>
                    <form:password path="password" placeholder="비밀번호" maxlength="20" cssClass="text"/>
                    <form:errors path="password" cssClass="error"/>
                </li>
                <li class="divide">
                    <div>
                        <input class="checkbox" type="checkbox" name="nornenjsRememberme" id="nornenjsRememberme"/>
                        <label for="nornenjsRememberme" class="colorGray size">자동 로그인</label>
                    </div>
                    <div>
                        <a href="${cp}/join" class="colorGrayHover size">회원가입</a>
                        <a href="${cp}/forgotPassword" class="colorGrayHover size">비밀번호 찾기</a>
                    </div>
                </li>
                <li class="button">
                    <button type="submit">로그인</button>
                </li>
            </ul>
            </sec:authorize>
            <sec:authorize access="hasRole('ROLE_DOCTOR')">
            <ul>
                <li>
                    <sec:authentication property="principal.username" var="username"/>
                    <span class="username">Welcome! <c:out value="${username}"/></span>
                </li>
            </ul>
            </sec:authorize>
        </form:form>
        
    </article>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
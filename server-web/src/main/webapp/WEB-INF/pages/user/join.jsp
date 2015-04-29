<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>

<section class="layoutFullSection">
    
    <article class="layoutJoinArticle">
        
        <header class="actorJoinHeader">
            <a href="${cp}/"><h1>Nornen<span class="colorOrange">js</span></h1></a>
            <a href="${cp}/join"><h2>회원가입</h2></a>
        </header>
        
        <form:form commandName="actorInfo"  action="${cp}/join" method="post" htmlEscape="true" cssClass="actorJoinForm">
            <ul>
                <lI>    
                    <form:label path="actor.username">아이디</form:label>
                    <form:input path="actor.username" placeholder="영문, 숫자만 입력가능합니다." maxlength="20"/>
                    <form:errors path="actor.username" cssClass="error"/>
                </lI>
                <li>
                    <form:label path="email">이메일주소</form:label>
                    <form:input path="email" placeholder="email@nornenjs.com" maxlength="200"/>
                    <form:errors path="email" cssClass="error"/>
                </li>
                <li>
                    <form:label path="actor.password">비밀번호</form:label>
                    <form:password path="actor.password" placeholder="문자,숫자,특수문자로 구성된 8자리 이상 비밀번호를 입력해주세요." maxlength="20"/>
                    <form:errors path="actor.password" cssClass="error"/>
                </li>
                <li>
                    <label for="confirmPassword">비밀번호 확인</label>
                    <input type="password" id="confirmPassword" name="confirmPassword" placeholder="동일한 비밀번호를 입력해주세요."/>
                </li>
                <li>
                    <form:label path="lastName">성</form:label>
                    <form:input path="lastName" placeholder="성을 입력해주세요." maxlength="20"/>
                    <form:errors path="lastName" cssClass="error"/>
                </li>
                <li>
                    <form:label path="firstName">이름</form:label>
                    <form:input path="firstName" placeholder="이름을 입력해주세요." maxlength="20"/>
                    <form:errors path="firstName" cssClass="error"/>
                </li>
                <li class="actorJoinBtn">
                    <button type="submit" class="orangeButton">회원가입</button>
                </li>
            </ul>
        </form:form>
        
    </article>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
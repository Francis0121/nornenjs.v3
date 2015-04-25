<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>

<section class="layoutFullSection">
    
    <article class="layoutForgotArticle">

        <header class="actorForgotHeader">
            <a href="${cp}/"><h1>Nornen<span class="colorOrange">js</span></h1></a>
            <a href="${cp}/forgotPassword"><h2>비밀번호 찾기</h2></a>
        </header>
        
        <form:form commandName="actorInfo" action="${cp}/forgotPassword" method="post" htmlEscape="true" cssClass="actorForgotForm">
            <ul>
                <li class="description">
                    회원가입을 한 이메일 주소나 아이디를 입력해주시기 바랍니다.
                </li>
                <lI>
                    <form:input path="email" placeholder="이메일 주소나 아이디를 입력해주세요."/>
                    <form:errors path="email"/>
                </lI>
                <li class="actorForgotBtn">
                    <button type="submit">변경 이메일 전송</button>
                </li>
            </ul>
        </form:form>
        
    </article>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
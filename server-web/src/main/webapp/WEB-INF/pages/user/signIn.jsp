<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>

<section class="layoutFullSection">
    
    <article class="layoutFullArticle">
        
        <header class="actorHeader">
            <h1>Nornen<span class="colorOrange">js</span></h1>
            <h2>Volume Rendering system</h2>
        </header>
        
        <form id="user" action="${cp}/j_spring_security_check" method="post" class="actorForm">
            <ul>
                <li>
                    <input class="text" id="nornenjsUsername" name="nornenjsUsername" type="text" placeholder="아이디 또는 이메일"/>
                </li>
                <li>
                    <input class="text" id="nornenjsPassword" name="nornenjsPassword" type="password" placeholder="비밀번호"/>
                </li>
                <c:if test="${not empty param.login_error}">
                    <li class="error">
                        <c:out value="${SPRING_SECURITY_LAST_EXCEPTION.message}"/>
                    </li>
                </c:if>
                <li class="divide">
                    <div>
                        <input class="checkbox" type="checkbox" name="nornenjsRememberme" id="nornenjsRememberme"/>
                        <label for="nornenjsRememberme" class="colorGray size">자동 로그인</label>
                    </div>
                    <div>
                        <a href="#" class="colorGrayHover size">회원가입</a>
                        <a href="#" class="colorGrayHover size">비밀번호 찾기</a>
                    </div>
                </li>
                <li class="button">
                    <button type="submit">로그인</button>
                </li>
            </ul>
        </form>
        
    </article>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
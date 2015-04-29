<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>
<%@ include file="../../layout/nav.jspf" %>


<section class="layoutContentSectionWrap">

    <section class="layoutContentSection">
        
        <header>
            <h2>My Info</h2>
        </header>
        
        <sec:authentication property="principal.username" var="username"/>
        <form:form commandName="actorInfo" action="${cp}/myInfo/${username}" method="post" htmlEscape="true" cssClass="actorUpdateForm">
            <form:hidden path="updateDate"/>
            <ul>
                <li>
                    <span class="updateDate">[ 최신 수정 일자 : <c:out value="${actorInfo.updateDate}"/> ]</span>
                </li>
                <li>
                    <form:label path="actor.username">아이디</form:label>
                    <form:input path="actor.username" readonly="true" cssClass="readonly"/>
                    <form:errors path="actor.username"/>
                </li>
                <lI>
                    <form:label path="email">이메일</form:label>
                    <form:input path="email" readonly="true" cssClass="readonly"/>
                    <form:errors path="email" cssClass="error"/>
                </lI>
                <li>
                    <form:label path="lastName">성</form:label>
                    <form:input path="lastName" placeholder="성을 입력해주세요." maxlength="20"/>
                    <form:errors path="lastName" cssClass="error"/>
                </li>
                <li>
                    <form:label path="firstName">이름</form:label>
                    <form:input path="firstName" placeholder="이름을 입력해주세요" maxlength="20"/>
                    <form:errors path="firstName" cssClass="error"/>
                </li>
                <li class="actorUpdateBtn">
                    <button type="submit" class="orangeButton">정보수정</button>
                </li>
            </ul>
        </form:form>
    </section>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
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
                    <form:errors path="lastName" cssClass="error"/>
                    <form:input path="lastName" placeholder="성을 입력해주세요." maxlength="20"/>
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

        <header>
            <h2>Change Password</h2>
        </header>

        <form:form commandName="actor" action="${cp}/myInfo/${username}" method="put" htmlEscape="true" cssClass="actorUpdateForm">
            <form:hidden path="username"/>
            <ul>
                <li>
                    <form:label path="password">비밀번호</form:label>
                    <form:password path="password" cssClass="text" placeholder="현재 비밀번호를 입력해주세요." maxlength="20"/>
                    <form:errors path="password" cssClass="error"/>
                </li>
                <li>
                    <form:label path="changePassword">비밀번호변경</form:label>
                    <form:password path="changePassword" cssClass="text" placeholder="문자,숫자,특수문자로 구성된 8자리 이상 비밀번호를 입력해주세요." maxlength="20"/>
                    <form:errors path="changePassword" cssClass="error"/>
                </li>
                <li>
                    <label for="confirmPassword">비밀번호확인</label>
                    <input type="password" id="confirmPassword" name="confirmPassword" placeholder="동일한 비밀번호를 입력해주세요."/>
                </li>
                <li class="actorUpdateBtn">
                    <button type="submit" class="orangeButton">비밀번호수정</button>
                </li>
            </ul>
        </form:form>
        
    </section>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
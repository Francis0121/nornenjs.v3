<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../layout/header.jspf" %>
<%@ include file="../layout/nav.jspf" %>


<section class="layoutContentSectionWrap">

    <section class="layoutContentSection">

        <sec:authorize access="hasRole('ROLE_DOCTOR')">
            ROLE_DOCTOR
        </sec:authorize>
        
        <sec:authorize access="isAnonymous()">
            Anonymous
        </sec:authorize>
        
    </section>

</section>

<%@ include file="../layout/footer.jspf" %>
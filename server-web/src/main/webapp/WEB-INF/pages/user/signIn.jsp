<%@ include file="../../layout/header.jspf" %>

<section>
    <article>
        <form id="user" action="${cp}/j_spring_security_check" method="post">
            <ul>
                <li>
                    <input id="nornenjs_username" name="nornenjs_username" type="text" placeholder="ID"/>
                </li>
                <li>
                    <input id="nornenjs_password" name="nornenjs_password" type="password" placeholder="PW"/>
                </li>
                <c:if test="${not empty param.login_error}">
                    <li class="error">
                        <c:out value="${SPRING_SECURITY_LAST_EXCEPTION.message}"/>
                    </li>
                </c:if>
                <li>
                    <button type="submit">Sign In</button>
                </li>
            </ul>
        </form>
    </article>
</section>

<%@ include file="../../layout/footer.jspf" %>
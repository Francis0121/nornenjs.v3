<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>
<%@ include file="../../layout/nav.jspf" %>

<section class="layoutContentSectionWrap">

    <section class="layoutContentSection">

        <form:form commandName="volume" action="${cp}/volume/upload" method="post" htmlEscape="true" cssClass="volumeForm" enctype="multipart/form-data">
            <ul>
                <li>
                    <form:label path="width">가로</form:label>
                    <form:input path="width" cssClass="text" placeholder="볼륨 데이터의 가로 크기를 입력해주세요." maxlength="4" />
                    <form:errors path="width" cssClass="error"/>
                </li>
                <li>
                    <form:label path="height">세로</form:label>
                    <form:input path="height" cssClass="text" placeholder="볼륨 데이터의 가로 세로를 입력해주세요." maxlength="4"/>
                    <form:errors path="height" cssClass="error"/>
                </li>
                <li>
                    <form:label path="depth">깊이</form:label>
                    <form:input path="depth" cssClass="text" placeholder="볼륨 데이터의 가로 깊이를 입력해주세요." maxlength="4"/>
                    <form:errors path="depth" cssClass="error"/>
                </li>
                <li>
                    <form:label path="title">설명</form:label>
                    <form:textarea path="title" placeholder="볼륨 데이터에 대한 설명을 입력해주세요." maxlength="50"></form:textarea>
                    <form:errors path="title" cssClass="error"/>
                </li>
                <li>
                    <label for="dataUpload">볼륨 데이터</label>
                    <div id="volumeUploadBtn">
                        <c:if test="${data ne null}">
                            <span class="success">파일업로드 : <c:out value="${data.name}"/></span>
                        </c:if>
                    </div>
                    <input type="file" id="dataUpload" name="dataUpload"/>
                    <form:hidden path="volumeDataPn"/>
                    <form:errors path="volumeDataPn" cssClass="error"/>
                </li>
            </ul>

            <div class="volumeRenderingSampleWrap">
                <figure>
                    <img src="${cp}/data/thumbnail/${thumbnails[1] eq null ? -1 : thumbnails[1]}" id="thumbnailMPRx"/>
                    <figcaption>
                        MPR-X 영상
                    </figcaption>
                </figure>
                <figure>
                    <img src="${cp}/data/thumbnail/${thumbnails[2] eq null ? -1 : thumbnails[2]}" id="thumbnailMPRy"/>
                    <figcaption>
                        MPR-Y 영상
                    </figcaption>
                </figure>
                <figure>
                    <img src="${cp}/data/thumbnail/${thumbnails[3] eq null ? -1 : thumbnails[3]}" id="thumbnailMPRz"/>
                    <figcaption>
                        MPR-Z 영상
                    </figcaption>
                </figure>
                <figure>
                    <img src="${cp}/data/thumbnail/${thumbnails[0] eq null ? -1 : thumbnails[0]}" id="thumbnailMPRvolume"/>
                    <figcaption>
                        볼륨 영상
                    </figcaption>
                </figure>
            </div>

            <div class="volumeBtnWrap">
                <button type="submit" class="orangeButton volumeDataBtn">볼륨데이터 등록</button>
            </div>
        </form:form>
        
    </section>

</section>

<%@ include file="../../layout/footer.jspf" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>
<%@ include file="../../layout/nav.jspf" %>

<section class="layoutContentSectionWrap">

    <section class="layoutContentSection">

        <header>
            <h2>볼륨 데이터 업로드</h2>
        </header>

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
                    <div id="volumeUploadBtnWrap">
                        <input type="file" id="dataUpload" name="dataUpload"/>
                    </div>
                    <div id="volumeUploadTextWrap" class="volumeUploadTextWrap">
                        <p>볼륨 데이터를 업로드 하기 위해 볼륨에 대한</p>
                        <p><span class="emphasize">가로</span>, <span class="emphasize">세로</span>, <span class="emphasize">깊이</span> 수치를 입력해주시기 바랍니다.</p>
                    </div>
                    <form:hidden path="volumeDataPn"/>
                    <form:errors path="volumeDataPn" cssClass="error"/>

                    <div class="volumeRenderingSampleWrap" style="display: none;" id="volumeRenderingSampleWrap">
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

                </li>
                <li class="volumeBtnWrap">
                    <button type="submit" class="orangeButton volumeDataBtn">볼륨데이터 등록</button>
                </li>
            </ul>

        </form:form>
        
    </section>

    <div id="volumeLoadingWrap" class="volumeLoadingWrap volumeLoadingWrapHide">

        <div class="size">
            <div class="center">
                <img src="${cp}/resources/image/loading.gif"/><br/>
                <span class="text">Now data loading ... </span>
            </div>
        </div>

    </div>

</section>

<%@ include file="../../layout/footer.jspf" %>
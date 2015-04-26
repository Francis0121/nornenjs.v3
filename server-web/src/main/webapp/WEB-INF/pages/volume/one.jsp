<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>
<%@ include file="../../layout/nav.jspf" %>

<section class="layoutContentSectionWrap">

    <section class="layoutContentSection">
        
        <article class="volumeRenderingOne volumeRenderingMpr" id="volumeMprX">
            <div class="title">
                <span>MPR-X</span>
            </div>
            <img src="${cp}/resources/image/sample/01.jpg"/>
        </article>

        <article class="volumeRenderingOne volumeRenderingMpr" id="volumeMprY">
            <div class="title">
                <span>MPR-Y</span>
            </div>
            <img src="${cp}/resources/image/sample/01.jpg"/>
        </article>

        <article class="volumeRenderingOne volumeRenderingMpr" id="volumeMprZ">
            <div class="title">
                <span>MPR-Z</span>
            </div>
            <img src="${cp}/resources/image/sample/01.jpg"/>
        </article>
        
        <article class="volumeRenderingOne" id="volumeRendering">
            <div class="title">
                <button type="button" id="renderingSizeBtn" class="renderingExpandBtn" title="Size Change">Expand</button>
                <span>VOLUME</span>
            </div>
            <img src="${cp}/resources/image/sample/01.jpg"/>
        </article>
        
        <article class="volumeRenderingOTP">
            
        </article>
        
    </section>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
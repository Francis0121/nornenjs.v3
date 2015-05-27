<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>
<%@ include file="../../layout/nav.jspf" %>

<section class="layoutContentSectionWrap">

    <section class="layoutContentSection" style="margin-left: 40px;">
        
        <article class="volumeRenderingOne volumeRenderingMpr" id="volumeMprX">
            <div class="title">
                <span>MPR-X</span>
            </div>
            <canvas id="volumeMprXCanvas" class="">

            </canvas>

            <div class="sliderWrap">
                <button type="button" id="tinyMprXPlusBtn" class="mprPlus volumeRenderIcon">B+</button>
                <div class="sliderVerticalMPR" id="sliderVerticalMPRX"></div>
                <button type="button" id="tinyMprXMinusBtn" class="mprMinus volumeRenderIcon">B-</button>
            </div>
        </article>

        <article class="volumeRenderingOne volumeRenderingMpr" id="volumeMprY">
            <div class="title">
                <span>MPR-Y</span>
            </div>
            <canvas id="volumeMprYCanvas" class="socketCanvas">

            </canvas>
            <div class="sliderWrap">
                <button type="button" id="tinyMprYPlusBtn" class="mprPlus volumeRenderIcon">B+</button>
                <div class="sliderVerticalMPR" id="sliderVerticalMPRY"></div>
                <button type="button" id="tinyMprYMinusBtn" class="mprMinus volumeRenderIcon">B-</button>
            </div>
        </article>

        <article class="volumeRenderingOne volumeRenderingMpr" id="volumeMprZ">
            <div class="title">
                <span>MPR-Z</span>
            </div>
            <canvas id="volumeMprZCanvas" class="socketCanvas">

            </canvas>

            <div class="sliderWrap">
                <button type="button" id="tinyMprZPlusBtn" class="mprPlus volumeRenderIcon">B+</button>
                <div class="sliderVerticalMPR" id="sliderVerticalMPRZ"></div>
                <button type="button" id="tinyMprZMinusBtn" class="mprMinus volumeRenderIcon">B-</button>
            </div>
        </article>
        
        <article class="volumeRenderingOne" id="volumeRendering">

            <div class="title">
                <button type="button" id="renderingSizeBtn" class="renderingExpandBtn" title="Size Change">Expand</button>
                <span>VOLUME</span>

                <div class="activeBtnWrap">
                    <button type="button" id="volumeBtn" class="volumeRenderIcon volumeBtnActive">V</button>
                    <button type="button" id="mipBtn" class="volumeRenderIcon mipBtnNoneActive">M</button>
                    <button type="button" id="qualityBtn" class="volumeRenderIcon qualityBtnNoneActive">V</button>
                    <button type="button" id="otfBtn" class="volumeRenderIcon otfBtnNoneActive">M</button>
                    <button type="button" id="tinyScalePlusBtn" class="mprPlus volumeRenderIcon">S+</button>
                    <button type="button" id="tinyScaleMinusBtn" class="mprMinus volumeRenderIcon">S-</button>
                </div>

            </div>

            <canvas id="volumeRenderingCanvas" class="socketCanvas">

            </canvas>
            <div class="sliderWrap">
                <button type="button" id="tinyBrightnessPlusBtn" class="brightPlus volumeRenderIcon">B+</button>
                <div class="sliderVerticalBright" id="sliderVerticalBrightness"></div>
                <button type="button" id="tinyBrightnessMinusBtn" class="brightMinus volumeRenderIcon">B-</button>
            </div>
        </article>

        
        <div class="volumeRenderingOTF" id="volumeRenderingOTF" title="OTF Table">

            <div>
                <div id="leftTopTextWrap" class="otfNumberChangeWrap">
                    <a href="#" id="leftTopTextPlus" class="changeBtn">+</a>
                    <span id="leftTopText" class="number">80</span>
                    <a href="#" id="leftTopTextMinus" class="changeBtn">-</a>
                </div>

                <div id="leftBottomTextWrap" class="otfNumberChangeWrap">
                    <a href="#" id="leftBottomTextPlus" class="changeBtn">+</a>
                    <span id="leftBottomText">65</span>
                    <a href="#" id="leftBottomTextMinus" class="changeBtn">-</a>
                </div>
            </div>
            <svg height="150" width="275">
                <g>
                    <line x1="0" y1="40" x2="275" y2="40" style="stroke:rgb(203,203,203);stroke-width:3;" stroke-dasharray="5,5"/>
                    <line x1="0" y1="110" x2="275" y2="110" style="stroke:rgb(203,203,203);stroke-width:3;" stroke-dasharray="5,5"/>
                    <!-- polygon -->
                    <polygon points="90,40 110,40 130,110 75,110" style="fill:rgb(119,119,119);" id="otfPolygon"/>
                </g>

                <g>
                    <!-- Top -->
                    <line x1="90" y1="40" x2="110" y2="40" style="stroke:rgb(75,75,75);stroke-width:6; opacity: .9; cursor: move;" id="otfTopLine"/>
                    <!-- Left Line -->
                    <line x1="75" y1="110" x2="90" y2="40" style="stroke:rgb(75,75,75);stroke-width:6; opacity: .9;" id="otfLeftDashLine" class="otfDashLine"/>
                    <!-- Bottom -->
                    <line x1="0" y1="110" x2="265" y2="110" style="stroke:rgb(75,75,75);stroke-width:6; opacity: .9;" id="otfBottomLine"/>
                    <!-- Right Line -->
                    <line x1="110" y1="40" x2="130" y2="110" style="stroke:rgb(75,75,75);stroke-width:6; opacity: .9; " id="otfRightDashLine" class="otfDashLine"/>

                    <line x1="10" y1="20" x2="10" y2="123" style="stroke:rgb(75,75,75);stroke-width:6; opacity: .9; " id="otfRightDashLine" class="otfDashLine"/>

                    <!-- Left Top Circle-->
                    <circle cx="90" cy="40" r="6" fill="rgb(0,0,0)" id="otfLeftTopCircle" class="otfCircle"/>
                    <!-- Left Bottom Circle -->
                    <circle cx="75" cy="110" r="6" fill="rgb(0,0,0)" id="otfLeftBottomCircle" class="otfCircle"/>

                    <!-- Right Top Circle-->
                    <circle cx="110" cy="40" r="6" fill="rgb(0,0,0)" id="otfRightTopCircle" class="otfCircle"/>
                    <!-- Right Bottom Circle -->
                    <circle cx="130" cy="110" r="6" fill="rgb(0,0,0)" id="otfRightBottomCircle" class="otfCircle"/>
                </g>
            </svg>
            <div>
                <div id="rightTopTextWrap" class="otfNumberChangeWrap">
                    <a href="#" id="rightTopTextPlus" class="changeBtn">+</a>
                    <span id="rightTopText">100</span>
                    <a href="#" id="rightTopTextMinus" class="changeBtn">-</a>
                </div>

                <div id="rightBottomTextWrap" class="otfNumberChangeWrap">
                    <a href="#" id="rightBottomTextPlus" class="changeBtn">+</a>
                    <span id="rightBottomText">120</span>
                    <a href="#" id="rightBottomTextMinus" class="changeBtn">-</a>
                </div>

            </div>

        </div>
        
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


<script type="text/javascript" src="http://112.108.40.164:5000/socket.io/socket.io.js"></script>
<script type="text/javascript">
    var relayAddr = "http://112.108.40.164:5000";
    var init = {
        savePath : '<c:out value="${data.savePath}"/>',
        width : <c:out value="${volume.width}"/>,
        height : <c:out value="${volume.height}"/>,
        depth : <c:out value="${volume.depth}"/>
    };
    var renderImgUrl = '${cp}/data/thumbnail/${thumbnails[0] eq null ? -1 : thumbnails[0]}';
    var mprXImgUrl = '${cp}/data/thumbnail/${thumbnails[1] eq null ? -1 : thumbnails[1]}';
    var mprYImgUrl = '${cp}/data/thumbnail/${thumbnails[2] eq null ? -1 : thumbnails[2]}';
    var mprZImgUrl = '${cp}/data/thumbnail/${thumbnails[3] eq null ? -1 : thumbnails[3]}';
</script>
<script type="text/javascript" src="${cp}/resources/javascript/socket.js"></script>

<%@ include file="../../layout/footer.jspf" %>
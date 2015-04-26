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
        
        <article class="volumeRenderingOTF">
            
            <script>
                var isTopCircle = false;
                var beforeX;
                
                function onMouseDown(evt) {
                    var circle = evt.target;
                    isTopCircle = true;
                    beforeX = evt.pageX;
                }

                function onMouseMove(evt) {
                    var circle = evt.target;
                    if(isTopCircle){
                        console.log(isTopCircle);
                        if(evt.pageX > beforeX){ // + 
                            circle.setAttribute('cx', circle.getAttribute('cx')+10);
                        } else{ // -
                            circle.setAttribute('cx', circle.getAttribute('cx')-10);
                        }
                    }
                }
                
                function onMouseUp(evt){
                    var circle = evt.target;
                    isTopCircle = false;
                }
                
            </script>
            
            <svg height="210" width="500">
                <!-- Top -->
                <line x1="100" y1="20" x2="500" y2="20" style="stroke:rgb(243,157,65);stroke-width:3" id="otfTopLine"/>
                <!-- Line -->
                <line x1="50" y1="100" x2="100" y2="20" style="stroke:rgb(243,157,65);stroke-width:10" id="otfDashLine" class="otfDashLine"/>
                <!-- Bottom -->
                <line x1="20" y1="100" x2="500" y2="100" style="stroke:rgb(243,157,65);stroke-width:3" id="otfBottomLine"/>
                <!-- Top Circle-->
                <circle cx="100" cy="20" r="10" fill="rgb(224,72,54)" id="otfTopCircle" class="otfCircle"/>
                <!-- Bottom Circle -->
                <circle cx="50" cy="100" r="10" fill="rgb(224,72,54)" id="otfBottomCircle" class="otfCircle"/>
            </svg>

        </article>
        
    </section>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
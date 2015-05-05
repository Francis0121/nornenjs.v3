<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>
<%@ include file="../../layout/nav.jspf" %>

<script src="http://112.108.40.166:5000/socket.io/socket.io.js"></script>

<script>
    document.oncontextmenu = function(e){
        var evt = new Object({ keyCode:93 });
        if(event.preventDefault != undefined)
            event.preventDefault();
        if(event.stopPropagation != undefined)
            event.stopPropagation();
    }

    var relay = io.connect('http://112.108.40.166:5000',{ forceNew : true, reconnection : false });

    var count = 0;

    /**
     * send connect message to server
     */
    relay.emit('getInfo', count++);

    /**
     * emit connect - response message
     */
    relay.on('getInfoClient', function(info){
        //console.log(info, typeof info);
        if(!info.conn){
            console.log('Connection User is full');
            return;
        }else{
            relay.disconnect();
            bindSocket(info);
        }
    });

    /**
     * Other client disconnect
     */
    relay.on('connSocket', function(){
        console.log('Connection Socket');
        relay.emit('getInfo');
    });

    var socket;

    var bindSocket = function(info){

        socket = io.connect('http://'+info.ipAddress+':'+info.port);

        console.log(info);

        var init = {
            savePath : '<c:out value="${data.savePath}"/>',
            width : <c:out value="${volume.width}"/>,
            height : <c:out value="${volume.height}"/>,
            depth : <c:out value="${volume.depth}"/>
        };

        socket.emit('join', info.deviceNumber);
        socket.emit('init', init);

        socket.on('loadCudaMemory', function(){
            socket.emit('webPng');
        });

        /**
         * Socket stream data
         */
        socket.on('stream', function(imageBlob){

            var blob = new Blob( [ imageBlob ], { type: 'image/jpeg' } );
            var url = (window.URL || window.webkitURL).createObjectURL(blob);
            var canvas = document.getElementById('volumeRenderingCanvas');
            var ctx = canvas.getContext('2d');

            var img = new Image(512, 512);
            img.onload = function(){
                ctx.drawImage(img, 0, 0, 512, 512, 0, 0, canvas.clientWidth, canvas.clientWidth);
            };
            img.src = url;
        });

        // ~ Touch

        var canvas = document.getElementById('volumeRenderingCanvas');

        var left = {
            isOn : false,
            beforeX : 0,
            beforeY : 0,
            count : 0
        };

        var right = {
            isOn : false,
            beforeX : 0,
            beforeY : 0,
            count : 0
        };

        var rotationOption = {
            rotationX : 0,
            rotationY : 0,
            isPng : false
        };

        var moveOption = {
            positionX : 0,
            positionY : 0,
            isPng : false
        };

        var scaleOption = {
            positionZ : 3.0
        };

        var brightOption = {
            brightness : 2.0
        };


        canvas.addEventListener('mousedown', function(event){
            event.preventDefault();

            switch(event.button){
                case 0:
                    left.isOn = true;
                    left.beforeX = event.pageX;
                    left.beforeY = event.pageY;
                    left.count = 0;
                    break;
                case 1:

                    break;
                case 2:
                    right.isOn = true;
                    right.beforeX = event.pageX;
                    right.beforeY = event.pageY;
                    right.count = 0;
                    break;
            }

        });

        canvas.addEventListener('mousemove', function(event){
            event.preventDefault();
            left.count++;
            right.count++;

            switch(event.button){
                case 0:
                    if(left.isOn && left.count%3 == 0){
                        rotationOption.rotationX += (event.pageX - left.beforeX)/3.0;
                        rotationOption.rotationY += (event.pageY - left.beforeY)/3.0;
                        rotationOption.isPng = false;
                        left.beforeX = event.pageX;
                        left.beforeY = event.pageY;

                        socket.emit('leftMouse', rotationOption);
                    }
                    break;
                case 1:

                    break;
                case 2:
                    if(right.isOn && right.count%3 == 0){
                        moveOption.positionX += (event.pageX - right.beforeX)/50.0;
                        moveOption.positionY -= (event.pageY - right.beforeY)/50.0;
                        moveOption.isPng = false;

                        right.beforeX = event.pageX;
                        right.beforeY = event.pageY;

                        socket.emit('rightMouse', moveOption);
                    }
                    break;
            }


        });

        canvas.addEventListener('mouseup', function(event){
            event.preventDefault();
            switch(event.button){
                case 0:
                    left.isOn = false;
                    rotationOption.isPng = true;
                    socket.emit('leftMouse', rotationOption);
                    break;
                case 1:

                    break;
                case 2:
                    right.isOn = false;
                    moveOption.isPng = true;
                    socket.emit('rightMouse', moveOption);
                    break;
            }
        });

        var wheelTimeout = undefined;
        var wheel = function (event){
            scaleOption.isPng = false;
            scaleOption.positionZ += -(event.wheelDelta/1200);
            socket.emit('wheelScale', scaleOption);

            if(wheelTimeout == undefined) {
                wheelTimeout = setTimeout(wheelTimeFunc, 1000);
            }else{
                clearTimeout(wheelTimeout);
                scaleOption.isPng = true;
                wheelTimeout = setTimeout(wheelTimeFunc, 1000);
            }
        };

        var wheelTimeFunc = function(){
            scaleOption.isPng = true;
            socket.emit('wheelScale', scaleOption);
            wheelTimeout = undefined;
        };

        canvas.addEventListener('mousewheel', wheel, false);
        canvas.addEventListener('DOMMouseScroll', wheel, false);

        // ~ Btn Event
        document.getElementById('tinyScalePlusBtn').addEventListener('click', function(){
            scaleOption.positionZ -= 0.1;
            socket.emit('sizeBtn', scaleOption);
        });

        document.getElementById('tinyScaleMinusBtn').addEventListener('click', function(){
            scaleOption.positionZ += 0.1;
            socket.emit('sizeBtn', scaleOption);
        });

        document.getElementById('tinyBrightnessPlusBtn').addEventListener('click', function(){
            brightOption.brightness += 0.1;
            socket.emit('brightBtn', brightOption);
        });

        document.getElementById('tinyBrightnessMinusBtn').addEventListener('click', function(){
            brightOption.brightness -= 0.1;
            socket.emit('brightBtn', brightOption);
        });

        window.onresize = function() {
            socket.emit('webPng');
        };
    };

</script>

<section class="layoutContentSectionWrap">

    <section class="layoutContentSection">
        
        <article class="volumeRenderingOne volumeRenderingMpr" id="volumeMprX">
            <div class="title">
                <span>MPR-X</span>
            </div>
            <img src="${cp}/data/thumbnail/${thumbnails[1] eq null ? -1 : thumbnails[1]}"/>
        </article>

        <article class="volumeRenderingOne volumeRenderingMpr" id="volumeMprY">
            <div class="title">
                <span>MPR-Y</span>
            </div>
            <img src="${cp}/data/thumbnail/${thumbnails[2] eq null ? -1 : thumbnails[2]}"/>
        </article>

        <article class="volumeRenderingOne volumeRenderingMpr" id="volumeMprZ">
            <div class="title">
                <span>MPR-Z</span>
            </div>
            <img src="${cp}/data/thumbnail/${thumbnails[3] eq null ? -1 : thumbnails[3]}"/>
        </article>
        
        <article class="volumeRenderingOne" id="volumeRendering">
            <div class="title">
                <button type="button" id="renderingSizeBtn" class="renderingExpandBtn" title="Size Change">Expand</button>
                <span>VOLUME</span>
            </div>
            <div class="event">
                <button type="button" id="tinyBrightnessPlusBtn">B+</button>
                <button type="button" id="tinyBrightnessMinusBtn">B-</button>
                <button type="button" id="tinyScalePlusBtn">S+</button>
                <button type="butto$('.volumeRenderingOTF>svg').attr('width', width);n" id="tinyScaleMinusBtn">S-</button>
            </div>
            <canvas width="300" height="300" id="volumeRenderingCanvas">

            </canvas>
            <%--<img src="${cp}/data/thumbnail/${thumbnails[0] eq null ? -1 : thumbnails[0]}"/>--%>
        </article>

        
        <article class="volumeRenderingOTF">

            <svg height="250" width="500">

                <!-- Top -->
                <line x1="240" y1="50" x2="300" y2="50" style="stroke:rgb(243,157,65);stroke-width:3" id="otfTopLine"/>
                <!-- Left Line -->
                <line x1="195" y1="150" x2="240" y2="50" style="stroke:rgb(243,157,65);stroke-width:10" id="otfLeftDashLine" class="otfDashLine"/>
                <!-- Bottom -->
                <line x1="0" y1="150" x2="768" y2="150" style="stroke:rgb(243,157,65);stroke-width:3" id="otfBottomLine"/>
                <!-- Right Line -->
                <line x1="300" y1="50" x2="360" y2="150" style="stroke:rgb(243,157,65);stroke-width:10" id="otfRightDashLine" class="otfDashLine"/>

                <!-- Left Top Circle-->
                <circle cx="240" cy="50" r="10" fill="rgb(224,72,54)" id="otfLeftTopCircle" class="otfCircle"/>
                <!-- Left Bottom Circle -->
                <circle cx="195" cy="150" r="10" fill="rgb(224,72,54)" id="otfLeftBottomCircle" class="otfCircle"/>

                <!-- Right Top Circle-->
                <circle cx="300" cy="50" r="10" fill="rgb(224,72,54)" id="otfRightTopCircle" class="otfCircle"/>
                <!-- Right Bottom Circle -->
                <circle cx="360" cy="150" r="10" fill="rgb(224,72,54)" id="otfRightBottomCircle" class="otfCircle"/>
            </svg>

        </article>
        
    </section>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
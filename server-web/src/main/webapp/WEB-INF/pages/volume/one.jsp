<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>
<%@ include file="../../layout/nav.jspf" %>

<script src="http://112.108.40.19:5000/socket.io/socket.io.js"></script>

<script>

    var relay = io.connect('http://112.108.40.19:5000',{ forceNew : true, reconnection : false });

    var count = 0;
    var fps = 0;
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

    var bindSocket = function(info){

        var socket = io.connect('http://'+info.ipAddress+':'+info.port);

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

            fps++;
        });

        var fpsInterval = function(){
            //console.log('Fps ['+fps+']')
            fps = 0;
        }

        setInterval(fpsInterval, 1000);


        // ~ Touch

        var canvas = document.getElementById('volumeRenderingCanvas');

        var touch = {
            isOn : false,
            beforeX : 0,
            beforeY : 0
        };

        var mouse = {
            isOn : false,
            beforeX : 0,
            beforeY : 0,
            count : 0
        };

        var option = {
            rotationX : 0,
            rotationY : 0
        };

        canvas.addEventListener('touchstart', function(event){
            event.preventDefault();
            var touches = event.changedTouches;

            if(touches.length == 1){
                touch.isOn = true;
                touch.beforeX = touches[0].pageX;
                touch.beforeY = touches[0].pageY;
            }
        });

        canvas.addEventListener('touchmove', function(event){
            event.preventDefault();
            var touches = event.changedTouches;

            if(touch.isOn){
                option.rotationX += (touches[0].pageX - touch.beforeX)/10.0;
                option.rotationY += (touches[0].pageY - touch.beforeY)/10.0;

                touch.beforeX = touches[0].pageX;
                touch.beforeY = touches[0].pageY;

                socket.emit('touch', option);
            }
        });

        canvas.addEventListener('touchend', function(event){
            event.preventDefault();
            touch.isOn = false;
        });

        canvas.addEventListener('touchcancel', function(event) {
            event.preventDefault();
            touch.isOn = false;
        });

        canvas.addEventListener('touchleave', function(event){
            event.preventDefault();
            touch.isOn = false;
        });


        canvas.addEventListener('mousedown', function(event){
            event.preventDefault();

            mouse.isOn = true;
            mouse.beforeX = event.pageX;
            mouse.beforeY = event.pageY;
            mouse.count = 0;
        });

        canvas.addEventListener('mousemove', function(event){
            event.preventDefault();
            mouse.count++;
            if(mouse.isOn && mouse.count%3 == 0){
                option.rotationX += (event.pageX - mouse.beforeX)/3.0;
                option.rotationY += (event.pageY - mouse.beforeY)/3.0;

                mouse.beforeX = event.pageX;
                mouse.beforeY = event.pageY;

                socket.emit('leftMouse', option);
            }
        });

        canvas.addEventListener('mouseup', function(event){
            event.preventDefault();
            mouse.isOn = false;
        });


        var infinityFlag = false;
        var infinityInterval = null;

        var infinityFunction = function(){

            option.rotationX += 1;
            option.rotationY += 1;

            socket.emit('leftMouse', option);

        };

        $('#infinityRotateEvent').on('click', function(){

            if(infinityFlag){
                clearInterval(infinityInterval);
                infinityFlag = false;
            }else{
                infinityInterval= setInterval(infinityFunction, 1000/30);
                infinityFlag = true;
            }

        });

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
            <canvas width="300" height="300" id="volumeRenderingCanvas">

            </canvas>
            <%--<img src="${cp}/data/thumbnail/${thumbnails[0] eq null ? -1 : thumbnails[0]}"/>--%>
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
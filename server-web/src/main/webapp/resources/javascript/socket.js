/**
 * F5, CTRL + E 새로 고침 막는 코드
 */
$(document).bind('keydown keyup', function(e) {
    if(e.which == 116) {
        return false;
    }
    if(e.which == 82 && e.ctrlKey) {
        return false;
    }
});

/**
 * 마우스 오른쪽 클릭 메뉴 막음
 */
document.oncontextmenu = function(e){
    var evt = new Object({ keyCode:93 });
    if(event.preventDefault != undefined)
        event.preventDefault();
    if(event.stopPropagation != undefined)
        event.stopPropagation();
}


var relay = io.connect(relayAddr,{ forceNew : true, reconnection : false });

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
        volumeRenderLoading(true, '사용자가 너무 많아 접근할 수 없습니다.<br/>잠시만 기다려주세요...');
        console.log('Connection User is full');
    }else{
        relay.disconnect();
        bindSocket(info);
        volumeRenderLoading(true, '데이터를 불러오고 있습니다.<br/>잠시만 기다려주세요 ...');
    }
});

/**
 * Other client disconnect
 */
relay.on('connSocket', function(){
    console.log('Connection Socket');
    relay.emit('getInfo');
});

// ~ Mpr video streaming
var socket;
var renderingType = 1;
var quality = 2;

var bindSocket = function(info){

    socket = io.connect('http://'+info.ipAddress+':'+info.port);

    var initializeServer = true;

    socket.emit('join', info.deviceNumber);
    socket.emit('init', init);

    socket.on('loadCudaMemory', function(){
        socket.emit('webPng', { type : 0, renderingType : renderingType, quality : quality} );
    });

    /**var
     * Socket stream data
     */
    socket.on('stream', function(image){
        if(initializeServer){
            setTimeout(volumeRenderLoading, 1000, false, '');
            initializeServer = false;
        }
        var imageBlob = image.data;
        var type = image.type;

        var blob = new Blob( [ imageBlob ], { type: 'image/jpeg' } );
        var url = (window.URL || window.webkitURL).createObjectURL(blob);
        var canvas;
        switch(type){
            case 0:
                canvas = document.getElementById('volumeRenderingCanvas');
                renderImgUrl = url;
                break;
            case 1:
                canvas = document.getElementById('volumeMprXCanvas');
                mprXImgUrl = url;
                break;
            case 2:
                canvas = document.getElementById('volumeMprYCanvas');
                mprYImgUrl = url;
                break;
            case 3:
                canvas = document.getElementById('volumeMprZCanvas');
                mprZImgUrl = url;
                break;
        }
        var ctx = canvas.getContext('2d');

        var img = new Image(image.width, image.height);
        img.onload = function(){
            ctx.drawImage(img, 0, 0, image.width, image.height, 0, 0, canvas.clientWidth, canvas.clientWidth);
        };
        img.src = url;
    });

    // ~ Touch

    var renderingCanvas = document.getElementById('volumeRenderingCanvas');
    var renderingCtx = renderingCanvas.getContext('2d');
    var renderingImg = new Image(512, 512);
    renderingImg.onload = function(){
        renderingCtx.drawImage(renderingImg, 0, 0, 512, 512, 0, 0, renderingCanvas.clientWidth, renderingCanvas.clientWidth);
    };
    renderingImg.src = renderImgUrl;

    var left = {
        isOn : false,
        beforeX : 0,
        beforeY : 0,
        count : 0,
        type : 0,
        quality : quality
    };

    var right = {
        isOn : false,
        beforeX : 0,
        beforeY : 0,
        count : 0,
        type : 0,
        quality : quality
    };

    var rotationOption = {
        rotationX : 0,
        rotationY : 0,
        isPng : false,
        type : 0,
        quality : quality
    };

    var moveOption = {
        positionX : 0,
        positionY : 0,
        isPng : false,
        type : 0,
        quality : quality
    };

    var scaleOption = {
        positionZ : 3.0,
        type : 0,
        quality : quality
    };

    var brightOption = {
        isPng : false,
        brightness : 2.0,
        type : 0,
        quality : quality
    };


    renderingCanvas.addEventListener('mousedown', function(event){
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

    renderingCanvas.addEventListener('mousemove', function(event){
        event.preventDefault();
        left.count++;
        right.count++;

        switch(event.button){
            case 0:
                if(left.isOn && left.count%3 == 0){
                    rotationOption.rotationX += (event.pageX - left.beforeX)/3.0;
                    rotationOption.rotationY += (event.pageY - left.beforeY)/3.0;
                    rotationOption.isPng = false;
                    rotationOption.renderingType = renderingType;
                    rotationOption.quality = quality;

                    left.beforeX = event.pageX;
                    left.beforeY = event.pageY;
                    socket.emit('leftMouse', rotationOption);
                }
                break;
            case 1:

                break;
            case 2:
                if(right.isOn && right.count%3 == 0){
                    moveOption.positionX += (event.pageX - right.beforeX)/150.0;
                    moveOption.positionY -= (event.pageY - right.beforeY)/150.0;
                    moveOption.isPng = false;
                    moveOption.renderingType = renderingType;
                    moveOption.quality = quality;

                    right.beforeX = event.pageX;
                    right.beforeY = event.pageY;

                    socket.emit('rightMouse', moveOption);
                }
                break;
        }


    });

    renderingCanvas.addEventListener('mouseup', function(event){
        event.preventDefault();
        switch(event.button){
            case 0:
                left.isOn = false;
                rotationOption.isPng = true;
                rotationOption.renderingType = renderingType;
                rotationOption.quality = quality;
                socket.emit('leftMouse', rotationOption);
                break;
            case 1:

                break;
            case 2:
                right.isOn = false;
                moveOption.isPng = true;
                moveOption.renderingType = renderingType;
                moveOption.quality = quality;
                socket.emit('rightMouse', moveOption);
                break;
        }
    });

    var wheelTimeout = undefined;
    var wheel = function (event){
        console.log(scaleOption.positionZ);
        if(scaleOption.positionZ > 6.0){
            scaleOption.positionZ = 6.0;
            return;
        }

        if(scaleOption.positionZ < 0.0){
            scaleOption.positionZ = 0.0;
            return;
        }

        scaleOption.isPng = false;
        scaleOption.positionZ += -(event.wheelDelta/1200);
        scaleOption.renderingType = renderingType;
        scaleOption.quality = quality;
        socket.emit('wheelScale', scaleOption);

        if(wheelTimeout == undefined) {
            wheelTimeout = setTimeout(wheelTimeFunc, 200);
        }else{
            clearTimeout(wheelTimeout);
            scaleOption.isPng = true;
            wheelTimeout = setTimeout(wheelTimeFunc, 200);
        }
    };

    var wheelTimeFunc = function(){
        scaleOption.isPng = true;
        scaleOption.renderingType = renderingType;
        scaleOption.quality = quality;
        socket.emit('wheelScale', scaleOption);
        wheelTimeout = undefined;
    };

    renderingCanvas.addEventListener('mousewheel', wheel, false);
    renderingCanvas.addEventListener('DOMMouseScroll', wheel, false);

    // ~ Btn Event
    document.getElementById('tinyScalePlusBtn').addEventListener('click', function(){
        if(scaleOption.positionZ <= 0.0){
            scaleOption.positionZ = 0.0;
            return;
        }
        scaleOption.positionZ -= 0.1;
        scaleOption.renderingType = renderingType;
        scaleOption.quality = quality;
        socket.emit('sizeBtn', scaleOption);
    });

    document.getElementById('tinyScaleMinusBtn').addEventListener('click', function(){
        if(scaleOption.positionZ >= 6.0){
            scaleOption.positionZ = 6.0;
            return;
        }
        scaleOption.positionZ += 0.1;
        scaleOption.renderingType = renderingType;
        scaleOption.quality = quality;
        socket.emit('sizeBtn', scaleOption);
    });

    document.getElementById('tinyBrightnessPlusBtn').addEventListener('click', function(){
        if(brightOption.brightness >= 12.0){
            brightOption.brightness = 12.0;
            return;
        }
        brightOption.isPng = true;
        brightOption.brightness += 0.1;
        brightOption.renderingType = renderingType;
        brightOption.quality = quality;
        $('#sliderVerticalBrightness').slider('value', brightOption.brightness*100);
        socket.emit('brightBtn', brightOption);
    });

    document.getElementById('tinyBrightnessMinusBtn').addEventListener('click', function(){
        if(brightOption.brightness <= 0.0){
            brightOption.brightness = 0.0;
            return;
        }
        brightOption.isPng = true;
        brightOption.brightness -= 0.1;
        brightOption.renderingType = renderingType;
        brightOption.quality = quality;
        $('#sliderVerticalBrightness').slider('value', brightOption.brightness*100);
        socket.emit('brightBtn', brightOption);
    });

    window.onresize = function() {
        socket.emit('webPng', { type : 0, renderingType : renderingType, quality : quality});
    };

    var transferScaleXOption = {
        transferScaleX : 0.5,
        type : 1,
        isPng : false
    };

    document.getElementById('tinyMprXMinusBtn').addEventListener('click', function(){
        if(transferScaleXOption.transferScaleX <= 0.0){
            transferScaleXOption.transferScaleX = 0.0;
            return;
        }
        transferScaleXOption.isPng = true;
        transferScaleXOption.transferScaleX -= 0.01;
        $('#sliderVerticalMPRX').slider('value', transferScaleXOption.transferScaleX*100);
        socket.emit('transferScaleXEvent', transferScaleXOption);
    });

    document.getElementById('tinyMprXPlusBtn').addEventListener('click', function(){
        if(transferScaleXOption.transferScaleX >= 1.0){
            transferScaleXOption.transferScaleX = 1.0;
            return;
        }
        transferScaleXOption.isPng = true;
        transferScaleXOption.transferScaleX += 0.01;
        $('#sliderVerticalMPRX').slider('value', transferScaleXOption.transferScaleX*100);
        socket.emit('transferScaleXEvent', transferScaleXOption);
    });


    var transferScaleYOption = {
        transferScaleY : 0.5,
        type : 2,
        isPng : false
    };

    document.getElementById('tinyMprYMinusBtn').addEventListener('click', function(){
        if(transferScaleYOption.transferScaleY <= 0.0){
            transferScaleYOption.transferScaleY = 0.0;
            return;
        }
        transferScaleYOption.isPng = true;
        transferScaleYOption.transferScaleY -= 0.01;
        $('#sliderVerticalMPRY').slider('value', transferScaleYOption.transferScaleY*100);
        socket.emit('transferScaleYEvent', transferScaleYOption);
    });

    document.getElementById('tinyMprYPlusBtn').addEventListener('click', function(){
        if(transferScaleYOption.transferScaleY >= 1.0){
            transferScaleYOption.transferScaleY = 1.0;
            return;
        }
        transferScaleYOption.isPng = true;
        transferScaleYOption.transferScaleY += 0.01;
        $('#sliderVerticalMPRY').slider('value', transferScaleYOption.transferScaleY*100);
        socket.emit('transferScaleYEvent', transferScaleYOption);
    });

    var transferScaleZOption = {
        transferScaleZ : 0.5,
        type : 3,
        isPng : false
    };

    document.getElementById('tinyMprZMinusBtn').addEventListener('click', function(){
        if(transferScaleZOption.transferScaleZ <= 0.0){
            transferScaleZOption.transferScaleZ = 0.0;
            return;
        }
        transferScaleZOption.isPng = true;
        transferScaleZOption.transferScaleZ -= 0.01;
        $('#sliderVerticalMPRZ').slider('value', transferScaleZOption.transferScaleZ*100);
        socket.emit('transferScaleZEvent', transferScaleZOption);
    });

    document.getElementById('tinyMprZPlusBtn').addEventListener('click', function(){
        if(transferScaleZOption.transferScaleZ >= 1.0){
            transferScaleZOption.transferScaleZ = 1.0;
            return;
        }
        transferScaleZOption.isPng = true;
        transferScaleZOption.transferScaleZ += 0.01;
        $('#sliderVerticalMPRZ').slider('value', transferScaleZOption.transferScaleZ*100);
        socket.emit('transferScaleZEvent', transferScaleZOption);
    });

    document.getElementById('volumeBtn').addEventListener('click', function(){
        renderingType = 1;
        $('#mipBtn').addClass('mipBtnNoneActive').removeClass('mipBtnActive');
        $('#volumeBtn').addClass('volumeBtnActive').removeClass('volumeBtnNoneActive');
        socket.emit('webPng', { type : 0, renderingType : renderingType, quality : quality });
        layoutFunction.typeChangeEventListener();

        $('#otfBtn').show();
        $('#qualityBtn').show();
    });

    document.getElementById('mipBtn').addEventListener('click', function(){
        renderingType = 2;
        $('#mipBtn').addClass('mipBtnActive').removeClass('mipBtnNoneActive');
        $('#volumeBtn').addClass('volumeBtnNoneActive').removeClass('volumeBtnActive');
        socket.emit('webPng', { type : 0, renderingType : renderingType, quality : quality});
        layoutFunction.typeChangeEventListener();

        $('#otfBtn').hide();
        $('#qualityBtn').hide();

        if($('#otfBtn').hasClass('otfBtnActive')){
            $('#otfBtn').addClass('otfBtnNoneActive').removeClass('otfBtnActive');
            $('#volumeRenderingOTF').dialog('close');
        }
    });

    document.getElementById('otfBtn').addEventListener('click', function(){
        if($('#otfBtn').hasClass('otfBtnActive')){
            $('#otfBtn').addClass('otfBtnNoneActive').removeClass('otfBtnActive');
            $('#volumeRenderingOTF').dialog('close');
        }else {
            $('#otfBtn').addClass('otfBtnActive').removeClass('otfBtnNoneActive');
            $('#volumeRenderingOTF').dialog('open');
            layoutFunction.otfDialogEventListener();
        }
    });

    document.getElementById('qualityBtn').addEventListener('click', function(){
        if($('#qualityBtn').hasClass('qualityBtnActive')){
            $('#qualityBtn').addClass('qualityBtnNoneActive').removeClass('qualityBtnActive');
            quality = 2;
        }else {
            $('#qualityBtn').addClass('qualityBtnActive').removeClass('qualityBtnNoneActive');
            quality = 1;
        }
        socket.emit('webPng', { type : 0, renderingType : renderingType, quality : quality});
    });


    $(function() {
        $( '#sliderVerticalMPRX' ).slider({
            orientation: 'vertical',
            range: 'min',
            min: 0,
            max: 100,
            value: 50,
            slide: function( event, ui ) {
                transferScaleXOption.isPng = false;
                transferScaleXOption.transferScaleX = ui.value/100;
                socket.emit('transferScaleXEvent', transferScaleXOption);
            },
            stop : function(event, ui) {
                transferScaleXOption.isPng = true;
                transferScaleXOption.transferScaleX = ui.value/100;
                socket.emit('transferScaleXEvent', transferScaleXOption);
                transferScaleXOption.isPng = false;
            }
        });

        $( '#sliderVerticalMPRY' ).slider({
            orientation: 'vertical',
            range: 'min',
            min: 0,
            max: 100,
            value: 50,
            slide: function( event, ui ) {
                transferScaleYOption.isPng = false;
                transferScaleYOption.transferScaleY = ui.value/100;
                socket.emit('transferScaleYEvent', transferScaleYOption);
            },
            stop : function(event, ui) {
                transferScaleYOption.isPng = true;
                transferScaleYOption.transferScaleY = ui.value/100;
                socket.emit('transferScaleYEvent', transferScaleYOption);
                transferScaleYOption.isPng = false;
            }
        });

        $( '#sliderVerticalMPRZ' ).slider({
            orientation: 'vertical',
            range: 'min',
            min: 0,
            max: 100,
            value: 50,
            slide: function( event, ui ) {
                transferScaleZOption.isPng = false;
                transferScaleZOption.transferScaleZ = ui.value/100;
                socket.emit('transferScaleZEvent', transferScaleZOption);
            },
            stop : function(event, ui) {
                transferScaleZOption.isPng = true;
                transferScaleZOption.transferScaleZ = ui.value/100;
                socket.emit('transferScaleZEvent', transferScaleZOption);
                transferScaleZOption.isPng = false;
            }
        });

        $( '#sliderVerticalBrightness' ).slider({
            orientation: 'vertical',
            range: 'min',
            min: 0,
            max: 1200,
            value: 200,
            slide: function( event, ui ) {
                console.log( ui.value );
                brightOption.isPng = false;
                brightOption.brightness = ui.value/100;
                brightOption.renderingType = renderingType;
                brightOption.quality = quality;
                socket.emit('brightBtn', brightOption);
            },
            stop : function( event, ui ){
                brightOption.isPng = true;
                brightOption.brightness = ui.value/100;
                brightOption.renderingType = renderingType;
                brightOption.quality = quality;
                socket.emit('brightBtn', brightOption);
            }
        });

        $('#volumeRenderingOTF').dialog({
            autoOpen: false,
            resizable : false,
            width : 350,
            position: { at: "left bottom"},
            close: function( event, ui ) {
                if($('#otfBtn').hasClass('otfBtnActive')){
                    $('#otfBtn').addClass('otfBtnNoneActive').removeClass('otfBtnActive');
                    $('#volumeRenderingOTF').dialog('close');
                }
            }
        });
    });

};



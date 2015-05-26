/**
 * Copyright Francis kim.
 */
var EVENT_MESSAGE = require('./message');
var ENUMS = require('../enums');
var logger = require('../logger');
var Encoding = require('../cuda/encoding').Encoding;

/**
 * Web Event Handler
 *
 * @param cudaRenderMap
 *  적제된 쿠다 모듈이 있는 Hash map Object
 *  TODO 적제되어 있는 모듈에 대한것을 파라미터로 넘겨주는 상황이 별로 좋지 않아 보임*
 * @constructor
 */
var Web = function(cudaRenderMap){
    if(typeof cudaRenderMap !== 'object'){
        throw new Error('Web event handler require client HashMap Object');
    }

    this.encoding = new Encoding();
    this.cudaRenderMap = cudaRenderMap;
    this.socket = null;
};

/**
 * Add socket event handler
 * @param socket
 *  socket object
 */
Web.prototype.addSocketEventListener = function(socket){
    if(typeof socket !== 'object'){
        throw new Error('Web event handler need socket client object');
    }
    
    this.socket = socket;
    this.pngEventListener();
    this.leftMouseEventListener();
    this.rightMouseEventListener();
    this.wheelEventListener();
    this.sizeBtnEventListener();
    this.brightBtnEventListener();
    this.otfEventListener();
    this.transferScaleEventListener();
};

/**
 * Last encoding image is type png
 * @param socket
 */
Web.prototype.pngEventListener = function(){
    var $this = this, 
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.PNG, function(option){
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        cudaRender.type = option.renderingType;
        cudaRender.positionZ = cudaRender.recordPositionZ;
        cudaRender.quality = option.quality;
        $this.encoding.png(cudaRender, socket, option.type);
    });
};

/**
 * LeftMouse Event Handler
 * @param socket
 */
Web.prototype.leftMouseEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.LEFT_CLICK, function(rotationOption){
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        cudaRender.type = rotationOption.renderingType;
        cudaRender.rotationX = rotationOption.rotationX;
        cudaRender.rotationY = rotationOption.rotationY;
        cudaRender.positionZ = cudaRender.recordPositionZ;
        cudaRender.quality = rotationOption.quality;

        if(rotationOption.isPng) {
            $this.encoding.png(cudaRender, socket, rotationOption.type);
        }else{
            $this.encoding.jpeg(cudaRender, socket, rotationOption.type);
        }
    });

};

Web.prototype.rightMouseEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.RIGHT_CLICK, function(moveOption){
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        cudaRender.type = moveOption.renderingType;
        cudaRender.positionX = moveOption.positionX;
        cudaRender.positionY = moveOption.positionY;
        cudaRender.positionZ = cudaRender.recordPositionZ;
        cudaRender.quality = moveOption.quality;

        if(moveOption.isPng) {
            $this.encoding.png(cudaRender, socket, moveOption.type);
        }else{
            $this.encoding.jpeg(cudaRender, socket, moveOption.type);
        }
    });
};

Web.prototype.wheelEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.WHEEL_SCALE, function(scaleOption){
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        cudaRender.type = scaleOption.renderingType;
        cudaRender.positionZ = scaleOption.positionZ;
        cudaRender.recordPositionZ = scaleOption.positionZ;
        cudaRender.quality = scaleOption.quality;

        if(scaleOption.isPng) {
            $this.encoding.png(cudaRender, socket, scaleOption.type);
        }else{
            $this.encoding.jpeg(cudaRender, socket, scaleOption.type);
        }
    });
};

Web.prototype.sizeBtnEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.SIZE_EVENT, function(scaleOption){
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        cudaRender.type = scaleOption.renderingType;
        cudaRender.positionZ = scaleOption.positionZ;
        cudaRender.recordPositionZ = scaleOption.positionZ;
        cudaRender.quality = scaleOption.quality;

        $this.encoding.png(cudaRender, socket, scaleOption.type);
    });

};

Web.prototype.brightBtnEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.BRIGHT_EVENT, function(brightOption){
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        cudaRender.type = brightOption.renderingType;
        cudaRender.brightness = brightOption.brightness;
        cudaRender.positionZ = cudaRender.recordPositionZ;
        cudaRender.quality = brightOption.quality;

        if(brightOption.isPng) {
            $this.encoding.png(cudaRender, socket, brightOption.type);
        }else{
            $this.encoding.jpeg(cudaRender, socket, brightOption.type);
        }
    });
};

Web.prototype.otfEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.OTF_EVENT, function(otfOption){
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        cudaRender.type = ENUMS.RENDERING_TYPE.VOLUME;
        cudaRender.transferStart = otfOption.transferStart;
        cudaRender.transferMiddle1 = otfOption.transferMiddle1;
        cudaRender.transferMiddle2 = otfOption.transferMiddle2;
        cudaRender.transferEnd = otfOption.transferEnd;
        cudaRender.transferFlag = otfOption.transferFlag
        cudaRender.positionZ = cudaRender.recordPositionZ;
        cudaRender.quality = otfOption.quality;

        if(otfOption.isPng) {
            $this.encoding.png(cudaRender, socket, otfOption.type);
        }else{
            $this.encoding.jpeg(cudaRender, socket, otfOption.type);
        }
    });

};

Web.prototype.transferScaleEventListener = function(){

    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.TRANSFER_SCALE_X_EVENT, function(transferScaleOptionX) {
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        cudaRender.type = ENUMS.RENDERING_TYPE.MPR;
        cudaRender.mprType = ENUMS.MPR_TYPE.X;
        cudaRender.positionZ = 3.0;

        cudaRender.transferScaleX = transferScaleOptionX.transferScaleX;
        cudaRender.transferScaleY = 0;
        cudaRender.transferScaleZ = 0;
        console.log('transferScaleOptionX', transferScaleOptionX);

        if(transferScaleOptionX.isPng) {
            $this.encoding.png(cudaRender, socket, transferScaleOptionX.type);
        }else{
            $this.encoding.jpeg(cudaRender, socket, transferScaleOptionX.type);
        }
    });

    socket.on(EVENT_MESSAGE.WEB.TRANSFER_SCALE_Y_EVENT, function(transferScaleOptionY) {
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        cudaRender.type = ENUMS.RENDERING_TYPE.MPR;
        cudaRender.mprType = ENUMS.MPR_TYPE.Y;
        cudaRender.positionZ = 3.0;

        cudaRender.transferScaleX = 0;
        cudaRender.transferScaleY = transferScaleOptionY.transferScaleY;
        cudaRender.transferScaleZ = 0;
        console.log('transferScaleOptionY', transferScaleOptionY);

        if(transferScaleOptionY.isPng) {
            $this.encoding.png(cudaRender, socket, transferScaleOptionY.type);
        }else{
            $this.encoding.jpeg(cudaRender, socket, transferScaleOptionY.type);
        }
    });

    socket.on(EVENT_MESSAGE.WEB.TRANSFER_SCALE_Z_EVENT, function(transferScaleOptionZ) {
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        cudaRender.type = ENUMS.RENDERING_TYPE.MPR;
        cudaRender.mprType = ENUMS.MPR_TYPE.Z;
        cudaRender.positionZ = 3.0;

        cudaRender.transferScaleX = 0;
        cudaRender.transferScaleY = 0;
        cudaRender.transferScaleZ = transferScaleOptionZ.transferScaleZ;
        console.log('transferScaleOptionZ', transferScaleOptionZ);


        if(transferScaleOptionZ.isPng) {
            $this.encoding.png(cudaRender, socket, transferScaleOptionZ.type);
        }else{
            $this.encoding.jpeg(cudaRender, socket, transferScaleOptionZ.type);
        }
    });

};


module.exports.Web = Web;
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
        $this.encoding.png(cudaRender, socket);
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

        cudaRender.rotationX = rotationOption.rotationX;
        cudaRender.rotationY = rotationOption.rotationY;

        if(rotationOption.isPng) {
            $this.encoding.png(cudaRender, socket);
        }else{
            $this.encoding.jpeg(cudaRender, socket);
        }
    });

};

Web.prototype.rightMouseEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.RIGHT_CLICK, function(moveOption){
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        cudaRender.rotationX = moveOption.rotationX;
        cudaRender.rotationY = moveOption.rotationY;

        if(moveOption.isPng) {
            $this.encoding.png(cudaRender, socket);
        }else{
            $this.encoding.jpeg(cudaRender, socket);
        }
    });
};

Web.prototype.wheelEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.WHEEL_SCALE, function(scaleOption){
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        cudaRender.positionZ = scaleOption.positionZ;
        if(scaleOption.isPng) {
            $this.encoding.png(cudaRender, socket);
        }else{
            $this.encoding.jpeg(cudaRender, socket);
        }
    });
};

Web.prototype.sizeBtnEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.SIZE_EVENT, function(scaleOption){
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        cudaRender.positionZ = scaleOption.positionZ;
        $this.encoding.png(cudaRender, socket);
    });

};

Web.prototype.brightBtnEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.WEB.BRIGHT_EVENT, function(brightOption){
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        cudaRender.brightness = brightOption.brightness;
        $this.encoding.png(cudaRender, socket);
    });

};

module.exports.Web = Web;
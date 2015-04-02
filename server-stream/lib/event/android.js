/**
 * Copyright Francis kim.
 */
var EVENT_MESSAGE = require('./message');
var ENUMS = require('../enums');
var logger = require('../logger');
var Encoding = require('../cuda/encoding').Encoding;

/**
 * Android Event Handler 
 *
 * @param cudaRenderMap
 *  적제된 쿠다 모듈이 있는 Hash map Object
 *  TODO 적제되어 있는 모듈에 대한것을 파라미터로 넘겨주는 상황이 별로 좋지 않아 보임
 * @constructor
 */
var Android = function(cudaRenderMap){
    if(typeof cudaRenderMap !== 'object'){
        throw new Error('Android event handler require client HashMap Object');
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
Android.prototype.addSocketEventListener = function(socket){
    if(typeof socket !== 'object'){
        throw new Error('Android event handler need socket client object');
    }

    this.socket = socket;
    this.touchEventListener();
    this.pngEventListener();
};

/**
 * Last encoding image is type png
 * @param socket
 */
Android.prototype.pngEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.ANDROID.PNG, function(option){
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        $this.encoding.png(cudaRender, socket);
    });
};

/**
 * Touch Event Handler
 * @param socket
 */
Android.prototype.touchEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.ANDROID.TOUCH, function(option){
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        cudaRender.rotationX = option.rotationX;
        cudaRender.rotationY = option.rotationY;

        $this.encoding.jpeg(cudaRender, socket);
    });

};

module.exports.Android = Android;
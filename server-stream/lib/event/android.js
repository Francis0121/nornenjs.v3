/**
 * Copyright Francis kim.
 */
var ENUMS = require('../enums');
var logger = require('../logger');
var Encoding = require('../cuda/encoding').Encoding;

/**
 * Android Event Handler 
 *
 * @param cudaRenderMap
 *  적제된 쿠다 모듈이 있는 Hash map Object
 * @constructor
 */
var Android = function(cudaRenderMap){
    if(typeof cudaRenderMap !== 'object'){
        throw new Error('Android Event Handler require client HashMap Object');
    }

    this.encoding = new Encoding();
    this.cudaRenderMap = cudaRenderMap;
};

/**
 * Add socket event handler 
 * @param socket
 *  socket object
 */
Android.prototype.addSocketEventListener = function(socket){
    this.touchEventListener(socket);
};

/**
 * Touch Event Handler
 * @param socket
 */
Android.prototype.touchEventListener = function(socket){

    var $this = this;
    socket.on('touch', function(option){
        
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        
        cudaRender.rotationX = option.rotationX;
        cudaRender.rotationY = option.rotationY;

        $this.encoding.jpeg(cudaRender, socket);
    });
    
};

module.exports.Android = Android;
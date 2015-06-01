/**
 * Created by pi on 15. 5. 31.
 */
var EVENT_MESSAGE = require('./message');
var ENUMS = require('../enums');
var logger = require('../logger');
var Encoding = require('../cuda/encoding').Encoding;


/**
 * Tizen Event Handler
 *
 * @param cudaRenderMap
 *  적제된 쿠다 모듈이 있는 Hash map Object
 *  TODO 적제되어 있는 모듈에 대한것을 파라미터로 넘겨주는 상황이 별로 좋지 않아 보임*
 * @constructor
 */
var Tizen = function(cudaRenderMap){
    if(typeof cudaRenderMap !== 'object'){
        throw new Error('Tizen event handler require client HashMap Object');
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
Tizen.prototype.addSocketEventListener = function(socket){
    if(typeof socket !== 'object'){
        throw new Error('Tizen event handler need socket client object');
    }

    this.socket = socket;
    this.qualityEventListener();
    this.jpegEventListener();
    this.rotationTouchEventListener();
    this.zoomTouchEventListener();
};

Tizen.prototype.qualityEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.TIZEN.QUALITY, function(){
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        $this.encoding.tizenQuality(cudaRender, socket);
    });
};

/**
 * Last encoding image is type
 */
Tizen.prototype.jpegEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.TIZEN.REQUEST, function(){
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        $this.encoding.tizenJpeg(cudaRender, socket);
    });
};


/**
 * Tizen Roation Touch Event Listner
 */
Tizen.prototype.rotationTouchEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.TIZEN.ROTATION, function(strOption){
        var option = JSON.parse(strOption);
        logger.debug('TIZEN '+ option);
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        cudaRender.rotationX = option.rotationX;
        cudaRender.rotationY = option.rotationY;

        $this.encoding.tizenJpeg(cudaRender, socket);
    });
};


/**
 * Zoom touch event listener
 */
Tizen.prototype.zoomTouchEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.TIZEN.ZOOM, function(strOption){
        var option = JSON.parse(strOption);
        logger.debug('TIZEN '+ option);
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        cudaRender.positionZ = option.positionZ;

        $this.encoding.tizenJpeg(cudaRender, socket);
    });
};

module.exports.Tizen = Tizen;
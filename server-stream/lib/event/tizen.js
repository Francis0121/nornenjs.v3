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
    this.brightnessEventListener();
    this.otfEventListener();
};

/**
 * Tizen quality image
 */
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
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        cudaRender.positionZ = option.positionZ;

        $this.encoding.tizenJpeg(cudaRender, socket);
    });
};

/**
 * Brightness Event Listener
 */
Tizen.prototype.brightnessEventListener = function(){
    var $this = this,
        socket = this.socket;

    socket.on(EVENT_MESSAGE.TIZEN.BRIGHTNESS, function(strOption){
        var option = JSON.parse(strOption);
        var cudaRender = $this.cudaRenderMap.get(socket.id);
        cudaRender.brightness = parseFloat(option.brightness);
        $this.encoding.tizenJpeg(cudaRender, socket);
    });
};


/**
 * Brightness Event Listener
 */
Tizen.prototype.otfEventListener = function(){
    var $this = this,
        socket = this.socket;

    var START = 65, MID1 = 80, MID2 = 100, END = 120;

    socket.on(EVENT_MESSAGE.TIZEN.OTF, function(strOption){
        var option = JSON.parse(strOption);
        var cudaRender = $this.cudaRenderMap.get(socket.id);

        var diff = parseInt(option.otf) - START;
        cudaRender.transferStart = START + diff;
        cudaRender.transferMiddle1 = MID1 + diff;
        cudaRender.transferMiddle2 = MID2 + diff;
        cudaRender.transferEnd = END + diff;
        cudaRender.transferFlag = option.transferFlag;
        logger.debug(cudaRender.transferStart, cudaRender.transferMiddle1, cudaRender.transferMiddle2, cudaRender.transferEnd);

        if(option.transferFlag == 1) {
            $this.encoding.tizenJpeg(cudaRender, socket);
        }else if(option.transferFlag == 2) {
            $this.encoding.tizenQuality(cudaRender, socket);
        }
    });
};

module.exports.Tizen = Tizen;
/**
 * Copyright Francis kim.
 */
var ENUMS = require('./enums');
var logger = require('./logger');
var sqlite = require('./sql/default');

var path = require('path');
var HashMap = require('hashmap').HashMap;
var socketIo = require('socket.io');

var CudaRender = require('./cuda/render').CudaRender;
var cu = require('./cuda/load');
var cuCtx = new cu.Ctx(0, cu.Device(0));
var Encoding = require('./cuda/encoding').Encoding;

/**
 * Nornejs server create 
 * @param server
 *  HttpCreateServer 
 * @constructor
 *  SET max conection client, cuda ptx path, cuda data path;
 */
var NornenjsServer = function(server){
    this.encoding = new Encoding();
    this.MAX_CONNECTION_CLIENT = 10;

    this.CUDA_PTX_PATH = path.join(__dirname, '../src-cuda/volume.ptx');
    this.CUDA_DATA_PATH = path.join(__dirname, './data/');
    this.CUDA_RENDER_MAP = new HashMap();

    this.server = server;
    this.io = socketIo.listen(this.server);
};

/**
 * Nornensjs server create
 */
NornenjsServer.prototype.connect = function(){
    this.socketIoConnect();
};

/**
 * Socket Io First Connect
 */
NornenjsServer.prototype.socketIoConnect = function(){

    var $this = this;
    var clientQueue = [];
    var streamClientCount = 0;

    this.io.sockets.on('connection', function(socket){

        /**
         * Connection User
         */
        socket.on('connectMessage', function(){

            var clientId = socket.id;

            var message = {
                error : '',
                success : false,
                clientId : clientId
            };

            if(streamClientCount < $this.MAX_CONNECTION_CLIENT){
                streamClientCount++;
                message.success = true;
            }else{
                clientQueue.push(clientId);
                message.error = 'Visitor limit';
            }

            logger.debug('[Connect] Stream user count [' + streamClientCount + ']');
            logger.debug('Send message client id [' + clientId + ']');

            socket.emit('connectMessage', message);
        });

        /**
         * Disconnection User - Wait queue client connect request
         */
        socket.on('disconnect', function () {
            
            var clientId = socket.id;
            
            if(clientQueue.indexOf(clientId) == -1){
                streamClientCount--;
                logger.debug('[Disconnect] Stream user count [' + streamClientCount + ']');
                logger.debug('Disconnect socket client id ['+ clientId +']');
            }

            var connectClientId = clientQueue.shift();

            if(connectClientId != undefined){
                logger.debug('Connect socket client id [' + connectClientId + ']');
                socket.broadcast.to(connectClientId).emit('otherClientDisconnect');
            }

        });


        /**
         * Init cudaRenderObject
         */
        socket.on('init', function(){

            var volumePn = 1;
            var clientId = socket.id;

            sqlite.db.get(sqlite.sql.volume.selectVolumeOne, { $pn: volumePn }, function(err, volume){

                logger.debug('[Stream] volume object ', volume);

                if(volume == undefined) {
                    logger.error('[Stream] Fail select volume data');
                    // TODO Announce fail lode volume data to client
                    return;
                }else {
                    logger.debug('[Stream] Register CUDA module ');

                    var cudaRender = new CudaRender(
                                            ENUMS.RENDERING_TYPE.VOLUME, $this.CUDA_DATA_PATH + volume.save_name,
                                            volume.width, volume.height, volume.depth,
                                            cuCtx, cu.moduleLoad($this.CUDA_PTX_PATH));
                    cudaRender.init();

                    $this.CUDA_RENDER_MAP.set(clientId, cudaRender);
                    $this.encoding.jpeg(cudaRender, socket);
                }
            });
        });

        /**
         * Event
         */
        socket.on('event', function(option){
            var clientId = socket.id;
            
            var cudaRender = $this.CUDA_RENDER_MAP.get(clientId);
            cudaRender.rotationX = option.rotationX;
            cudaRender.rotationY = option.rotationY;

            $this.encoding.jpeg(cudaRender, socket);
        });

    });

};

module.exports.NornenjsServer = NornenjsServer;

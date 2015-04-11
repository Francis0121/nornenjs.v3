/**
 * Copyright Francis kim.
 */

// ~ Import module (Me)
var ENUMS = require('./enums');
var logger = require('./logger');
var sqlite = require('./sql/default');
var util = require('./util');

var CudaRender = require('./cuda/render').CudaRender;
var cu = require('./cuda/load');
var cuDevice = cu.Device(0);
var cuCtx = new cu.Ctx(0, cuDevice);

var Android = require('./event/android').Android;
var Web = require('./event/web').Web;

// ~ Import module (Nodejs)
var path = require('path');
var HashMap = require('hashmap').HashMap;
var socketIo = require('socket.io');
var socketIoClient = require('socket.io-client');
var exec = require('child_process').exec;
var redis = require('redis');

// ~ Redis key
var keys = {
    HOSTLIST : 'HostList'
};

/**
 * Nornejs server create 
 * @param server
 *  HttpCreateServer 
 * @constructor
 *  SET max conection client, cuda ptx path, cuda data path;
 */
var NornenjsServer = function(server, isMaster, masterIpAddres){
    this.MAX_CONNECTION_CLIENT = 10;

    this.CUDA_PTX_PATH = path.join(__dirname, '../src-cuda/volume.ptx');
    this.CUDA_DATA_PATH = path.join(__dirname, './data/');

    this.server = server;
    this.io = socketIo.listen(this.server);
    this.cudaRenderMap = new HashMap();
    
    //TODO 해당되는 클라이언트가 무엇인지에 따라서 사용되는 socket event Handler를 다르게 한다.
    this.android = new Android(this.cudaRenderMap);
    this.web = new Web(this.cudaRenderMap);

    // Exec Redis Server create
    if(typeof isMaster !== 'boolean'){
        throw new Error('IsRoot type is "Boolean" type');
    }

    this.REDIS_PATH = '/home/pi/redis-3.0.0/src/redis-server';
    this.REDIS_PORT = 6379;
    this.ipAddress = null;
    this.redisProcess = null;

    if(isMaster){
        // ~ Master Relay server. Exec redis server and connect redis.
        this.redisProcess = exec(this.REDIS_PATH, function (error) {
            if (error !== null) {
                throw new Error(error);
            }
        });

        this.ipAddress = util.getIpAddress();

        var client = redis.createClient(this.REDIS_PORT, this.ipAddress, { } );

        client.flushall(function (error, reply){
            logger.debug('Flushall', reply);
        });

        this.addDevice();

        // ~ Test server IP add
        //client = redis.createClient(this.REDIS_PORT, this.ipAddress, { } );
        //client.hset(keys.HOSTLIST, '112.108.40.14_0', '0', function(){
        //    logger.debug(keys.HOSTLIST+' add device 112.108.40.14_0');
        //    client.quit();
        //});

    }else{
        // ~ Slave server. Connect master server redis.
        if(typeof masterIpAddres !== 'string'){
            throw new Error('IpAddress type is "String" type')
        }

        this.ipAddress = masterIpAddres;
        this.addDevice();
    }
};

// ~ Device information

/**
 *
 */
NornenjsServer.prototype.addDevice = function(callback){

    var client = redis.createClient(this.REDIS_PORT, this.ipAddress, { } );

    for(var i=0; i<cu.deviceCount; i++){
        var key = this.ipAddress+'_'+i;
        client.hset(keys.HOSTLIST, key, '0', function(err, reply){
            // ~ TODO error callback
            logger.debug(keys.HOSTLIST+' add device ' + key + ' Reply ' + reply);
            client.quit();

            if(callback === 'function') callback();
        });
    }

};

NornenjsServer.prototype.removeDevice = function(callback){

    var client = redis.createClient(this.REDIS_PORT, this.ipAddress, { } );

    for(var i=0; i<cu.deviceCount; i++){
        var key = this.ipAddress+'_'+i;
        client.hdel(keys.HOSTLIST, key, function(err, reply){
            // ~ TODO error callback
            logger.debug(keyS.HOSTLIST+' remove device ' + key + ' Reply ' + reply);
            client.quit();

            if(callback === 'function') callback();
        });
    }

};


/**
 * Nornensjs server create
 */
NornenjsServer.prototype.connect = function(){
    this.socketIoConnect();
    this.distributed();
};

/**
 * Distributed User
 */
NornenjsServer.prototype.distributed = function() {

    var client = redis.createClient(this.REDIS_PORT, this.ipAddress, { } );

    client.hgetall(keys.HOSTLIST, function (error, list) {
        if(error != null){
            throw new Error('Hash get all function error', error);
        }
        var ipHost,
            minClient = 10;

        for (key in list) {
            logger.debug('key - ', key, ': value -', list[key]);
            if(list[key] == 0){
                minClient = 0;
                ipHost = key.split('_');
                break;
            }

            if(minClient > list[key]){
                minClient = list[key];
                ipHost = key.split('_');
            }
        }
        client.quit();

        logger.debug(ipHost, minClient);
    });
};

/**
 * Socket Io First Connect
 */
NornenjsServer.prototype.socketIoConnect = function(){

    var $this = this;
    var clientQueue = [];
    var streamClientCount = 0;

    this.io.sockets.on('connection', function(socket){
        
        $this.android.addSocketEventListener(socket);
        $this.web.addSocketEventListener(socket);
        
        /**
         * Connection User
         * TODO 사용자 관리는 프록시 서버가 되는 부분으로 이전될 예정
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
                    
                    $this.cudaRenderMap.set(clientId, cudaRender);
                    socket.emit('loadCudaMemory');
                }
            });
        });

    });

};

/**
 * NornenjsServer close event. (Important Need server is closed)
 * @param callback
 */
NornenjsServer.prototype.close = function(callback){
    var $this = this;

    if(typeof $this.redisProcess !== 'object') {

        // ~ Slave Server
        logger.debug('Nornenjs server closed.');
        $this.removeDevice(callback);

    }else {
        // ~ Relay Server Remove all redis key
        var client = redis.createClient(this.REDIS_PORT, this.ipAddress, {});

        client.hgetall(keys.HOSTLIST, function (err, list) {
            var i = 0, size = Object.keys(list).length;
            for (key in list) {
                logger.debug('Delete redis key - ', key, ': value -', list[key]);
                client.hdel(keys.HOSTLIST, key, function (err, reply) {
                    i++;
                    if (i == size) {
                        client.quit();

                        logger.debug('Redis server kill.');
                        logger.debug('Nornenjs server closed.');

                        $this.redisProcess.kill();
                        callback();
                    }
                });
            }
        });
    }
};

module.exports.NornenjsServer = NornenjsServer;

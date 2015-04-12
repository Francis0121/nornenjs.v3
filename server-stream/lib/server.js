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
    this.MAX_CONNECTION_CLIENT = 5;

    this.CUDA_PTX_PATH = path.join(__dirname, '../src-cuda/volume.ptx');
    this.CUDA_DATA_PATH = path.join(__dirname, './data/');

    this.io = socketIo.listen(server);
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
    this.redisProcess = undefined;
    this.isMaster = isMaster;

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
            client.quit();
        });

        this.addDevice();

    }else{
        // ~ Slave server. Connect master server redis.
        if(typeof masterIpAddres !== 'string'){
            throw new Error('IpAddress type is "String" type')
        }

        this.ipAddress = masterIpAddres;
        this.addDevice();
    }
};

/**
 * Add graphic device
 *
 * @param callback
 */
NornenjsServer.prototype.addDevice = function(callback){

    var ipAddress = util.getIpAddress();
    var client = redis.createClient(this.REDIS_PORT, this.ipAddress, { } );

    var launch = function(client, key, isLast, callback){
        client.hset(keys.HOSTLIST, key, '0', function(err, reply){
            logger.debug(keys.HOSTLIST+' add device ' + key + ' Reply ' + reply);
            if(isLast){
                client.quit();
                if(typeof callback === 'function') callback();
            }
        });
    };

    for(var i=0; i<cu.deviceCount; i++){
        var key = ipAddress+'_'+i;
        launch(client, key, i+1 === cu.deviceCount ? true : false, callback);
    }

};

/**
 * Remove graphic device
 *
 * @param callback
 */
NornenjsServer.prototype.removeDevice = function(callback){

    var ipAddress = util.getIpAddress();
    var client = redis.createClient(this.REDIS_PORT, this.ipAddress, { } );

    var launch = function(client, key, isLast, callback){
        client.hdel(keys.HOSTLIST, key, function(err, reply){
            logger.debug(keys.HOSTLIST+' remove device ' + key + ' Reply ' + reply);
            if(isLast){
                client.quit();
                if(typeof callback === 'function') callback();
            }
        });
    };

    for(var i=0; i<cu.deviceCount; i++){
        var key = ipAddress+'_'+i;
        launch(client, key, i+1 === cu.deviceCount ? true : false, callback);
    }
};

/**
 * Increase or decrease Device count
 */
NornenjsServer.prototype.updateDevice = function(key, type, callback){

    var client = redis.createClient(this.REDIS_PORT, this.ipAddress, { } );

    client.hincrby(keys.HOSTLIST, key, type, function(err, reply){
        logger.debug(keys.HOSTLIST+ ' hash key update ' + type + ' Reply ' + reply);
        client.quit();

        if(typeof callback === 'function') callback();
    });
};

/**
 * Minimum device key Or No device count
 */
NornenjsServer.prototype.getDeviceKey = function(callback){

    var $this = this;
    var client = redis.createClient(this.REDIS_PORT, this.ipAddress, { } );
    var min = $this.MAX_CONNECTION_CLIENT+1,
        select = undefined;

    client.hgetall(keys.HOSTLIST, function (err, list) {

        for (key in list) {
            var val = list[key];
            if(min > val && val < $this.MAX_CONNECTION_CLIENT){
                select = key, min = val;
                if(min === 0) break;
            }
        }

        client.quit();
        if(typeof callback === 'function') callback(select);
    });
};

/**
 * User remove publish client to server
 */
NornenjsServer.prototype.publish = function(){

    var client = redis.createClient(this.REDIS_PORT, this.ipAddress, {});
    client.publish('streamOut', util.getIpAddress());
    client.quit();

};

/**
 * Nornensjs server create
 */
NornenjsServer.prototype.connect = function(){
    if(this.isMaster) {
        this.socketIoRelayServer();
    }else{
        this.socketIoSlaveServer();
    }
    this.socketIoCuda();
};


/**
 * Distributed User
 */
var clientQueue = [];
NornenjsServer.prototype.distributed = function(socket) {

    var $this = this;

    this.getDeviceKey(function(select){

        var socketId = socket.id;
        var info = {
                ipAddress : null,
                deviceNumber : null,
                port : 5000,
                conn : true
            };

        if(typeof select !== 'string'){
            clientQueue.push(socketId);
            logger.debug('Connection Device Full');
            logger.debug('Relay server list', clientQueue);
            info.conn = false;
        }else{
            $this.updateDevice(select, ENUMS.REDIS_UPDATE_TYPE.INCREASE);
            var split = select.split('_');
            info.ipAddress = split[0];
            info.deviceNumber = split[1];
            info.conn = true;
            logger.debug('Selected', split);
        }

        socket.emit('getInfoClient', info);
    });

};


/**
 * Socket Io First Connect
 */
var oneTime = true;
NornenjsServer.prototype.socketIoRelayServer = function(){

    var $this = this;
    var relayClient = [];
    var deviceMap = new HashMap();

    var subscribe = redis.createClient(this.REDIS_PORT, this.ipAddress, {});
    subscribe.subscribe('streamOut');

    $this.io.sockets.on('connection', function(socket){
        if(oneTime) {
            oneTime = false;
            subscribe.on('message', function (channel, message) {

                var connSocketId = undefined;
                while(typeof connSocketId === 'undefined'){
                    connSocketId = clientQueue.shift();
                    if(typeof socket.server.eio.clients[connSocketId] === 'undefined'){
                        connSocketId = undefined;
                    }
                }

                logger.debug(connSocketId, typeof connSocketId);
                socket.broadcast.to(connSocketId).emit('connSocket');

            });
        }

        /**
         * Connection Relay Server
         */
        socket.on('getInfo', function(object){
            logger.debug('Connect relay server', socket.id, object);
            $this.distributed(socket);
            relayClient.push(socket.id);
        });

        /**
         * Connection Stream Server
         */
        socket.on('join', function(deviceNumber){
            logger.debug('Connect stream server');
            deviceMap.set(socket.id, deviceNumber);
        });

        /**
         * Disconnection User - Wait queue client connect request
         */
        socket.on('disconnect', function () {

            var index = relayClient.indexOf(socket.id);
            if( index !== -1){
                relayClient = relayClient.splice(index, 1);
                logger.debug('Disconnect relay server');
                logger.debug('RelayClient List', relayClient);
            }else{
                var deviceNumber = deviceMap.get(socket.id);
                deviceMap.remove(socket.id);
                logger.debug('Disconnect stream server', deviceNumber);

                if(typeof deviceNumber !== 'undefined') {
                    // Update Redis Info
                    $this.updateDevice(util.getIpAddress() + '_' + deviceNumber, ENUMS.REDIS_UPDATE_TYPE.DECREASE);
                    // Publish client to server
                    $this.publish();
                }
            }
        });
    });
};

NornenjsServer.prototype.socketIoSlaveServer = function(){

    var $this = this;
    var deviceMap = new HashMap();

    $this.io.sockets.on('connection', function(socket){

        /**
         * Connection Stream Server
         */
        socket.on('join', function(deviceNumber){
            logger.debug('Connect stream server');
            deviceMap.set(socket.id, deviceNumber);
        });

        socket.on('disconnect', function () {
            var deviceNumber = deviceMap.get(socket.id);
            deviceMap.remove(socket.id);
            logger.debug('Disconnect stream server', deviceNumber);

            if(typeof deviceNumber !== 'undefined') {
                // Update Redis Info
                $this.updateDevice(util.getIpAddress() + '_' + deviceNumber, ENUMS.REDIS_UPDATE_TYPE.DECREASE);
                // Publish client to server
                $this.publish();
            }
        });
    });
};

/**
 * Cuda Init
 */
NornenjsServer.prototype.socketIoCuda = function(){

    var $this = this;

    $this.io.sockets.on('connection', function(socket){

        $this.android.addSocketEventListener(socket);

        $this.web.addSocketEventListener(socket);
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

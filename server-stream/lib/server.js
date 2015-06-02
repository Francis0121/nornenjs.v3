/**
 * Copyright Francis kim.
 */

// ~ Import module (Me)
var ENUMS = require('./enums');
var logger = require('./logger');
var util = require('./util');

var CudaRender = require('./cuda/render').CudaRender;
var cu = require('./cuda/load');

var Android = require('./event/android').Android;
var Web = require('./event/web').Web;
var Tizen = require('./event/tizen').Tizen;

// ~ Import module (Nodejs)
var path = require('path');
var HashMap = require('hashmap').HashMap;
var socketIo = require('socket.io');
var exec = require('child_process').exec;
var redis = require('redis');

// Not running sudo mount
//exec("sudo mount 112.108.40.14:/storage /storage", function(error, stdout, stderr){
//});

/**
 * Create constructor
 *
 * @param server
 *  Http server
 * @param httpPort
 *  Socket.io port
 * @param isMaster
 *  true : relay server master, false : slave server
 * @param masterIpAddres
 *  it`s only need slave server
 * @constructor
 */



var NornenjsServer = function(server, isMaster, masterIpAddres){
    this.MAX_CONNECTION_CLIENT = 4;

    this.CUDA_PTX_PATH = path.join(__dirname, '../src-cuda/volume.ptx');


    this.io = socketIo.listen(server);
    this.cudaRenderMap = new HashMap();

    this.android = new Android(this.cudaRenderMap);
    this.web = new Web(this.cudaRenderMap);
    this.tizen = new Tizen(this.cudaRenderMap);

    // Exec Redis Server create
    if(typeof isMaster !== 'boolean'){
        throw new Error('IsRoot type is "Boolean" type');
    }

    this.REDIS_PATH = '/home/pi/redis-3.0.0/src/redis-server';
    this.REDIS_PORT = 6379;
    this.ipAddress = null;
    this.redisProcess = undefined;
    this.isMaster = isMaster;
    this.cuCtxs = [];

    if(isMaster){
        // ~ Master Relay server. Exec redis server and connect redis.
        // ~
        var $this = this;
        this.redisProcess = exec(this.REDIS_PATH, function (error) {
            if (error != null) {
                exec('killall redis-server', function(killallError){
                    if(killallError != null) {
                        throw new Error(killallError);
                    }else{
                        $this.redisProcess = exec($this.REDIS_PATH, function(reError){
                            if(reError != null){
                                throw new Error(reError);
                            }
                        });
                    }
                });
            }
        });

        this.ipAddress = util.getIpAddress();

        var client = redis.createClient(this.REDIS_PORT, this.ipAddress, { } );
        client.flushall(function (error, reply){
            logger.info('Redis "Flushall command"', reply);
            client.quit();

        });

        this.addDevice();

        this.subscribe = redis.createClient(this.REDIS_PORT, this.ipAddress, {});
        this.subscribe.subscribe('streamOut');

    }else{
        // ~ Slave server. Connect master server redis.
        if(typeof masterIpAddres !== 'string'){
            throw new Error('IpAddress type is "String" type')
        }

        this.ipAddress = masterIpAddres;
        this.addDevice();
    }
};

// ~ Redis key
var keys = {
    HOSTLIST : 'HostList'
};

/**
 * Add graphic device
 *
 * @param callback
 */
NornenjsServer.prototype.addDevice = function(callback) {

    var launch = function (key, ipAddress, port, isLast, callback) {
        var client = redis.createClient(port, ipAddress, {});
        client.hset(keys.HOSTLIST, key, '0', function (err, reply) {
            logger.info('[Redis] ADD DEVICE ' + keys.HOSTLIST + ' add device ' + key + ' Reply ' + reply);
            client.quit();
            if (isLast) {
                if (typeof callback === 'function') callback();
            }
        });
    };

    var ipAddress = util.getIpAddress();
    for (var i = 0; i < cu.deviceCount; i++) {
        var key = ipAddress + '_' + i;
        launch(key, this.ipAddress, this.REDIS_PORT, i + 1 === cu.deviceCount ? true : false, callback);
    }

    for (var i = 0; i < cu.deviceCount; i++) {
        logger.info('[Init] Cuda context initialize in constructor DeviceNumber', i);
        this.cuCtxs.push(new cu.Ctx(0, cu.Device(i)));
    }
};

/**
 * Remove graphic device
 *
 * @param callback
 */
NornenjsServer.prototype.removeDevice = function(callback){

    var launch = function(key, ipAddress, port, isLast, callback){
        var client = redis.createClient(port, ipAddress, { } );
        client.hdel(keys.HOSTLIST, key, function(err, reply){
            logger.info('[Redis] REMOVE DEVICE '+keys.HOSTLIST+' remove device ' + key + ' Reply ' + reply);
            client.quit();
            if(isLast){
                if(typeof callback === 'function') callback();
            }
        });
    };

    var ipAddress = util.getIpAddress();
    for(var i=0; i<cu.deviceCount; i++){
        var key = ipAddress+'_'+i;
        launch(key, this.ipAddress, this.REDIS_PORT, i+1 === cu.deviceCount ? true : false, callback);
    }
};

/**
 * Increase or decrease Device count
 */
NornenjsServer.prototype.updateDevice = function(key, type, callback){

    var client = redis.createClient(this.REDIS_PORT, this.ipAddress, { } );

    client.hincrby(keys.HOSTLIST, key, type, function(err, reply){
        logger.info('[Redis] INCREMENT ' + keys.HOSTLIST+ ' hash key update ' + type + ' Reply ' + reply);
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
        client.quit();
        for (key in list) {
            var val = list[key];
            if(min > val && val < $this.MAX_CONNECTION_CLIENT){
                select = key, min = val;
                if(min === 0) break;
            }
        }
        if(typeof callback === 'function') callback(select, min);
    });
};

/**
 * User remove publish client to server
 */
NornenjsServer.prototype.publish = function(){

    logger.info('[REDIS]               PUBLISH START');
    var client = redis.createClient(this.REDIS_PORT, this.ipAddress, {});
    client.publish('streamOut', util.getIpAddress());
    client.quit();
    logger.info('[REDIS]               PUBLISH END');

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
NornenjsServer.prototype.distributed = function(socket, callback) {

    var $this = this;

    this.getDeviceKey(function(select, deviceCount){

        var socketId = socket.id;
        var info = {
                ipAddress : null,
                deviceNumber : null,
                deviceCount : deviceCount,
                port : 5000,
                conn : true
            };

        if(typeof select !== 'string'){
            clientQueue.push(socketId);
            logger.info('[Redis] CONNECT DEVICE FULL');
            logger.info('[Redis] CLIENT QUEUE LIST', clientQueue);
            info.conn = false;
        }else{
            $this.updateDevice(select, ENUMS.REDIS_UPDATE_TYPE.INCREASE);
            var split = select.split('_');
            info.ipAddress = split[0];
            info.deviceNumber = split[1];
            info.conn = true;
            logger.info('[Redis] SELECTED ', split);
        }

        socket.emit('getInfoClient', info);
        if(typeof callback === 'function') callback();
    });

};


/**
 * Socket Io First Connect
 */
var relayServer = [];
var deviceMap = new HashMap();

NornenjsServer.prototype.socketIoRelayServer = function(){

    var isRegisterSubscribe = true;
    var $this = this;

    $this.io.sockets.on('connection', function(socket){
        if(isRegisterSubscribe) {
            isRegisterSubscribe = false;
            $this.subscribe.on('message', function (channel, message) {

                var connSocketId = undefined;
                while(typeof connSocketId === 'undefined' && clientQueue.length !== 0){
                    connSocketId = clientQueue.shift();

                    if(typeof socket.server.eio.clients[connSocketId] === 'undefined'){
                        connSocketId = undefined;
                    }

                }

                if(typeof connSocketId !== 'undefined') {
                    logger.info('[Subscribe]            ', connSocketId, typeof connSocketId, '\n');
                    socket.broadcast.to(connSocketId).emit('connSocket');
                }

            });
        }

        /**
         * Connection Relay Server
         */
        socket.on('getInfo', function(count){
            logger.info('Count', count);
            var launch = function(){
                if(relayServer.indexOf(socket.id) === -1)
                    relayServer.push(socket.id);
            };
            logger.info('[Socket] CONNECT RELAY Server', socket.id);
            $this.distributed(socket, launch());
        });

        /**
         * Connection Stream Server
         */
        socket.on('join', function(deviceNumber){
            logger.info('[Socket]           CONNECT STREAM server\n');
            deviceMap.set(socket.id, deviceNumber);
        });

        /**
         * Disconnection User - Wait queue client connect request
         */
        socket.on('disconnect', function () {

            var index = relayServer.indexOf(socket.id);
            logger.info('RELAY SERVER !!!!', index);
            if( index !== -1){
                relayServer.splice(index, 1);
                logger.info('[Socket] DISCONNECT RELAY Server KEY ['+socket.id+']', index);
                logger.info('[Socket]        RELAY SERVER CONNECT List    -', relayServer.length == 0 ? 'NONE' : relayServer);
            }else{
                var deviceNumber = deviceMap.get(socket.id);

                var cudaRender = $this.cudaRenderMap.get(socket.id);
                //logger.debug('[Socket] DISCONNECT stream server', socket.id,cudaRender);
                if(cudaRender != null && cudaRender != undefined) {
                    logger.info('Confirm --------');
                    cudaRender.destroy();
                }

                deviceMap.remove(socket.id);

                logger.info('[Socket] DISCONNECT stream server', deviceNumber);
                if(typeof deviceNumber !== 'undefined') {
                    var launch = function(){
                        // Publish client to server
                        $this.publish();
                    };
                    // Update Redis Info
                    $this.updateDevice(util.getIpAddress() + '_' + deviceNumber, ENUMS.REDIS_UPDATE_TYPE.DECREASE, launch);
                }
            }
        });
    });
};


NornenjsServer.prototype.socketIoSlaveServer = function(){

    var $this = this;

    $this.io.sockets.on('connection', function(socket){

        /**
         * Connection Stream Server
         */
        socket.on('join', function(deviceNumber){
            logger.info('Connect stream server');
            deviceMap.set(socket.id, deviceNumber);
        });

        socket.on('disconnect', function () {
            var deviceNumber = deviceMap.get(socket.id);
            deviceMap.remove(socket.id);
            logger.info('Disconnect stream server', deviceNumber);

            if(typeof deviceNumber !== 'undefined') {
                var launch = function(){
                    // Publish client to server
                    $this.publish();
                };

                // Update Redis Info
                $this.updateDevice(util.getIpAddress() + '_' + deviceNumber, ENUMS.REDIS_UPDATE_TYPE.DECREASE, launch);

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

        $this.tizen.addSocketEventListener(socket);

        /**
         * Init cudaRenderObject
         */
        socket.on('init', function(init){
            logger.info('[Socket] '+JSON.stringify(init));
            var clientId = socket.id;
            var deviceCount = deviceMap.get(socket.id);
            logger.debug('[Stream] Register CUDA module ');

            var cudaRender = new CudaRender(
                ENUMS.RENDERING_TYPE.VOLUME, init.savePath,
                init.width, init.height, init.depth,
                $this.cuCtxs[deviceCount], cu.moduleLoad($this.CUDA_PTX_PATH));

            logger.info('[Stream]           Device Number', deviceCount);
            cudaRender.init();

            $this.cudaRenderMap.set(clientId, cudaRender);

            socket.emit('loadCudaMemory');
        });

        socket.on('tizenInit', function(){
            var clientId = socket.id;
            var deviceCount = deviceMap.get(socket.id);
            logger.debug('[Tizen Stream] Register CUDA module ');

            var cudaRender = new CudaRender(
                ENUMS.RENDERING_TYPE.VOLUME, '/storage/data/6ca3600c-fc0c-4af7-b804-7d94fb4db779',
                256, 256, 225,
                $this.cuCtxs[deviceCount], cu.moduleLoad($this.CUDA_PTX_PATH));

            logger.info('[Tizen Stream]           Device Number', deviceCount);
            cudaRender.init();

            $this.cudaRenderMap.set(clientId, cudaRender);

            socket.emit('loadCudaMemory');
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
        logger.info('Nornenjs server closed.');
        $this.removeDevice(callback);

    }else {
        // ~ Relay Server Remove all redis key
        this.subscribe.quit();
        logger.info('Redis subscribe client quit');

        var client = redis.createClient(this.REDIS_PORT, this.ipAddress, {});

        client.hgetall(keys.HOSTLIST, function (err, list) {
            var i = 0, size = Object.keys(list).length;
            for (key in list) {
                logger.info('Delete redis key - ', key, ': value -', list[key]);
                client.hdel(keys.HOSTLIST, key, function (err, reply) {
                    i++;
                    if (i == size) {
                        client.quit();

                        logger.info('Redis server kill.');
                        logger.info('Nornenjs server closed.');

                        $this.redisProcess.kill();
                        callback();
                    }
                });
            }
        });
    }
};

module.exports.NornenjsServer = NornenjsServer;

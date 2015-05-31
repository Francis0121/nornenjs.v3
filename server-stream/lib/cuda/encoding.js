/**
 * Copyright Francis kim.
 */
var logger = require('../logger');
var Jpeg = require('../node-jpeg/build/Release/jpeg').Jpeg;
var Png = require('png').Png;
var fs = require('fs');
/**
 * Encoding Jpeg and png Image
 * @constructor
 */
var Encoding = function(){
    
};

Encoding.prototype.thumbnail = function(cudaRender, savePath){
    var hrStart = process.hrtime();

    cudaRender.start();
    var hrCuda = process.hrtime(hrStart);
    logger.debug('Make start finish frame png compress execution time (hr) : %dms', hrCuda[1]/1000000);

    var png = new Png(cudaRender.d_outputBuffer, 512, 512, 'rgba');
    var buf = png.encodeSync();

    fs.writeFileSync(savePath, buf, 'binary');

    var hrEnd = process.hrtime(hrStart);
    logger.debug('Make start finish frame png compress execution time (hr) : %dms', hrEnd[1]/1000000);
};

/**
 * Make Android Jpeg Encoding image And streaming client
 *
 * @param cudaRender
 *  cudaRender object
 * @param socket
 *  socket.io object
 */
Encoding.prototype.Androidjpeg = function(cudaRender, socket){
    var hrStart = process.hrtime();
    cudaRender.imageWidth = 256;
    cudaRender.imageHeight = 256;
    cudaRender.start();
    var hrCuda = process.hrtime(hrStart);
    logger.debug('Make start finish frame jpeg compress execution time (hr) : %dms', hrCuda[1]/1000000);

    var jpeg = new Jpeg(cudaRender.d_outputBuffer, cudaRender.imageWidth, cudaRender.imageHeight, 'bgra');
    socket.emit('stream',  { data : jpeg.turboencodeSync(), width : cudaRender.imageWidth, height : cudaRender.imageWidth});
    cudaRender.end();

    var hrEnd = process.hrtime(hrStart);
    logger.debug('Make start finish frame jpeg compress execution time (hr) : %dms', hrEnd[1]/1000000);
};

/**
 * Make Android Png Encoding image And streaming client
 * @param cudaRender
 *  cudaRender object
 * @param socket
 *  socket.io object
 */
Encoding.prototype.Androidpng = function(cudaRender, socket){
    var hrStart = process.hrtime();
    cudaRender.imageWidth = 512;
    cudaRender.imageHeight = 512;
    cudaRender.start();
    var hrCuda = process.hrtime(hrStart);
    logger.debug('Make start finish frame png compress execution time (hr) : %dms', hrCuda[1]/1000000);

    var png = new Png(cudaRender.d_outputBuffer,  cudaRender.imageWidth, cudaRender.imageHeight, 'bgra');
    //socket.emit('stream', png.encodeSync());
    socket.emit('stream',  { data : png.encodeSync(), width : cudaRender.imageWidth, height : cudaRender.imageWidth});
    cudaRender.end();

    var hrEnd = process.hrtime(hrStart);
    logger.debug('Make start finish frame png compress execution time (hr) : %dms', hrEnd[1]/1000000);
};

/**
 * Make Jpeg Encoding image And streaming client
 * 
 * @param cudaRender
 *  cudaRender object 
 * @param socket
 *  socket.io object
 *  
 */
Encoding.prototype.jpeg = function(cudaRender, socket, type){
    var hrStart = process.hrtime();
    cudaRender.imageWidth = 256;
    cudaRender.imageHeight = 256;
    cudaRender.start();
    var hrCuda = process.hrtime(hrStart);
    logger.debug('Make start finish frame jpeg compress execution time (hr) : %dms', hrCuda[1]/1000000);

    var jpeg = new Jpeg(cudaRender.d_outputBuffer, cudaRender.imageWidth, cudaRender.imageHeight , 'rgba');
    socket.emit('stream', { data : jpeg.turboencodeSync(), type : type, width : cudaRender.imageWidth, height : cudaRender.imageWidth});
    cudaRender.end();

    var hrEnd = process.hrtime(hrStart);
    logger.debug('Make start finish frame jpeg compress execution time (hr) : %dms', hrEnd[1]/1000000);
};

/**
 * Make Png Encoding image And streaming client
 * @param cudaRender
 *  cudaRender object
 * @param socket
 *  socket.io object
 */
Encoding.prototype.png = function(cudaRender, socket, type){
    var hrStart = process.hrtime();
    cudaRender.imageWidth = 512;
    cudaRender.imageHeight = 512;
    cudaRender.start();
    var hrCuda = process.hrtime(hrStart);
    logger.debug('Make start finish frame png compress execution time (hr) : %dms', hrCuda[1]/1000000);

    var png = new Png(cudaRender.d_outputBuffer, cudaRender.imageWidth, cudaRender.imageHeight, 'rgba');
    socket.emit('stream', { data : png.encodeSync(), type : type, width : cudaRender.imageWidth, height : cudaRender.imageWidth } );
    cudaRender.end();

    var hrEnd = process.hrtime(hrStart);
    logger.debug('Make start finish frame png compress execution time (hr) : %dms', hrEnd[1]/1000000);
};

Encoding.prototype.tizenJpeg = function(cudaRender, socket){
    var hrStart = process.hrtime();

    cudaRender.imageWidth = 512;
    cudaRender.imageHeight = 512;
    cudaRender.brightness = 4.0;
    cudaRender.start();

    var hrCuda = process.hrtime(hrStart);
    logger.debug('Tizen Make start finish frame jpeg compress execution time (hr) : %dms', hrCuda[1]/1000000);

    var jpeg = new Jpeg(cudaRender.d_outputBuffer, cudaRender.imageWidth, cudaRender.imageHeight , 'rgba');
    var buffer = jpeg.turboencodeSync();
    socket.emit('tizenJpeg', {
                                stream :{
                                    size : buffer.length,
                                    buffer : buffer
                                }
                            });
    cudaRender.end();

    var hrEnd = process.hrtime(hrStart);
    logger.debug('Tizen Make start finish frame jpeg compress execution time (hr) : %dms', hrEnd[1]/1000000);
};

module.exports.Encoding = Encoding;
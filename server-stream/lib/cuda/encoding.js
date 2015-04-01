/**
 * Copyright Francis kim.
 */
var logger = require('../logger');
var Jpeg = require('jpeg').Jpeg;
var Png = require('png').Png;

/**
 * Encoding Jpeg and png Image
 * @constructor
 */
var Encoding = function(){
    
};

/**
 * Make Jpeg Encoding image And streaming client
 * 
 * @param cudaRender
 *  cudaRender object 
 * @param socket
 *  socket.io object
 */
Encoding.prototype.jpeg = function(cudaRender, socket){
    var hrStart = process.hrtime();

    cudaRender.start();
    var hrCuda = process.hrtime(hrStart);
    logger.debug('Make start finish frame jpeg compress execution time (hr) : %dms', hrCuda[1]/1000000);

    var jpeg = new Jpeg(cudaRender.d_outputBuffer, 512, 512, 'rgba');
    socket.emit('stream', jpeg.encodeSync());
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
Encoding.prototype.png = function(cudaRender, socket){
    var hrStart = process.hrtime();

    cudaRender.start();
    var hrCuda = process.hrtime(hrStart);
    logger.debug('Make start finish frame png compress execution time (hr) : %dms', hrCuda[1]/1000000);

    var png = new Png(cudaRender.d_outputBuffer, 512, 512, 'rgba');
    socket.emit('stream', png.encodeSync());
    cudaRender.end();

    var hrEnd = process.hrtime(hrStart);
    logger.debug('Make start finish frame png compress execution time (hr) : %dms', hrEnd[1]/1000000);
};

module.exports.Encoding = Encoding;
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
 * Make Jpeg Encoding image And streaming client
 * 
 * @param cudaRender
 *  cudaRender object 
 * @param socket
 *  socket.io object
 */
Encoding.prototype.jpeg = function(cudaRender, socket, type){
    var hrStart = process.hrtime();

    cudaRender.start();
    var hrCuda = process.hrtime(hrStart);
    logger.debug('Make start finish frame jpeg compress execution time (hr) : %dms', hrCuda[1]/1000000);

    var jpeg = new Jpeg(cudaRender.d_outputBuffer, 512, 512, 'rgba');
    socket.emit('stream', { data : jpeg.turboencodeSync(), type : type});
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

    cudaRender.start();
    var hrCuda = process.hrtime(hrStart);
    logger.debug('Make start finish frame png compress execution time (hr) : %dms', hrCuda[1]/1000000);

    var png = new Png(cudaRender.d_outputBuffer, 512, 512, 'rgba');
    socket.emit('stream', { data : png.encodeSync(), type : type} );
    cudaRender.end();

    var hrEnd = process.hrtime(hrStart);
    logger.debug('Make start finish frame png compress execution time (hr) : %dms', hrEnd[1]/1000000);
};

module.exports.Encoding = Encoding;
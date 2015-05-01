var Buffer = require('buffer').Buffer;

var fs  = require('fs');
var sys = require('sys');
var Jpeg = require('/usr/local/lib/node_modules/jpeg').Jpeg;

var jpeg = new Jpeg(d_outputBuffer, 512, 512, 'rgba');
var jpeg_img = jpeg.encodeSync().toString('binary');
fs.writeFileSync('./jpeg.jpeg', jpeg_img, 'binary');



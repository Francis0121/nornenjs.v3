var logger = require('./../logger');
var ENUMS = require('./../enums');
var Buffer = require('buffer').Buffer;
var cu = require('./load');
var fs = require('fs');
var constants = require('constants');
var mat4 = require('./mat/mat4');
var vec3 = require('./mat/vec3');

function iDivUp(a, b) {

    return (a % b != 0) ? (a >> 5 + 1) : (a >> 5);
}

(function (factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define([], factory);
    } else if (typeof exports === 'object') {
        // Node js environment
        exports.CudaRender = factory();
    } else {
        // Browser globals (this is window)
        this.CudaRender = factory();
    }
}(function () {

    function CudaRender(type, textureFilePath, volume_width, volume_height, volume_depth, cuCtx, cuModule) {
        this.type = type;
        this.textureFilePath = textureFilePath;
        this.volumewidth = volume_width;
        this.volumeheight = volume_height;
        this.volumedepth = volume_depth;
        this.cuCtx = cuCtx;
        this.cuModule = cuModule;
    }

    CudaRender.prototype = {
        constructor:CudaRender,

        recordPositionZ : 3.0,
        //Option
        imageWidth : 512,
        imageHeight : 512,
        density : 0.05,
        brightness : 2.0,
        transferOffset : 0.0,
        transferScaleX : 0.0,
        transferScaleY : 0.0,
        transferScaleZ : 0.0,
        positionX: 0.0,
        positionY: 0.0,
        positionZ: 3.0,
        rotationX: 0,
        rotationY: 0,
        transferStart: 65,
        transferMiddle1: 80,
        transferMiddle2: 100,
        transferEnd: 120,
        transferSize : 256,
        transferFlag : 2,
        mprType:1,
        quality:1,

        d_output : undefined,
        d_tf2Dtable : undefined,
        d_tf2DtableBuffer : undefined,
        d_invViewMatrix : undefined,
        c_invViewMatrix : undefined,
        d_outputBuffer : undefined,

        h_volume : undefined,
        d_volume : undefined,
        d_blockvolume : undefined,
        d_blockBUffer : undefined,

        blockSizeX : 32,
        blockSizeY : 32,

        blockVolumeX : undefined,
        blockVolumeY : undefined,
        blockVolumeZ : undefined,

        init : function(){
            // ~ VolumeLoad & VolumeTexture Binding
            var error = this.cuModule.memVolumeTextureAlloc(this.textureFilePath, this.volumewidth,this.volumeheight, this.volumedepth);
            this.makeBlockvolume();

        },
        makeBlockvolume : function(){

            this.h_volume = fs.readFileSync(this.textureFilePath);
            this.d_volume = cu.memAlloc(this.volumewidth * this.volumeheight * this.volumedepth);

            var error = this.d_volume.copyHtoD(this.h_volume);

            this.blockVolumeX= Math.ceil(this.volumewidth/4);
            this.blockVolumeY= Math.ceil(this.volumeheight/4);
            this.blockVolumeZ= Math.ceil(this.volumedepth/4);

            this.d_blockvolume = cu.memAlloc(this.blockVolumeX * this.blockVolumeY * this.blockVolumeZ);
            var error = this.d_blockvolume.memSet(this.blockVolumeX * this.blockVolumeY * this.blockVolumeZ);
            var _cuModule = this.cuModule;
            var cuFunction = _cuModule.getFunction('block_volume');

            var error = cu.launch(
                cuFunction, [iDivUp(this.blockVolumeX,32), iDivUp(this.blockVolumeY, 32), 1], [32, 32, 1],
                [
                    {
                        type: 'DevicePtr',
                        value: this.d_volume.devicePtr
                    },{
                        type: 'DevicePtr',
                        value: this.d_blockvolume.devicePtr
                    },{
                        type: 'Uint32',
                        value: this.volumewidth
                    },{
                        type: 'Uint32',
                        value: this.volumeheight
                    },{
                        type: 'Uint32',
                        value: this.volumedepth
                    },{
                        type: 'Uint32',
                        value: this.blockVolumeX
                    },{
                        type: 'Uint32',
                        value: this.blockVolumeY
                    },{
                        type: 'Uint32',
                        value: this.blockVolumeZ
                    }
                ]
            );
            this.d_blockBUffer = new Buffer(this.blockVolumeX * this.blockVolumeY * this.blockVolumeZ );
            this.d_blockvolume.copyDtoH(this.d_blockBUffer, false);

            _cuModule.memBlockTextureAlloc(this.d_blockBUffer, this.blockVolumeX , this.blockVolumeY, this.blockVolumeZ);

            this.d_volume.free();
            this.d_blockvolume.free();

            delete this.d_blockBUffer;
            delete this.h_volume;
        },
        make2Dtable :function(){

            this.d_tf2Dtable = cu.memAlloc(this.transferSize * this.transferSize * 4 * 4);
            //logger.debug('[INFO_CUDA] d_tf2Dtable.memAlloc', this.d_output);
            var error = this.d_tf2Dtable.memSet(this.transferSize * this.transferSize * 4 * 4);
            //logger.debug('[INFO_CUDA_TF] d_tf2Dtable.memSet', error);

            var _cuModule = this.cuModule;
            var cuFunction = _cuModule.getFunction('TF2d_kernel');
            logger.debug('[INFO_CUDA_TF] cuFunction', cuFunction);

            var error = cu.launch(
                cuFunction, [iDivUp(this.transferSize,this.blockSizeX), iDivUp(this.transferSize, this.blockSizeY), 1], [this.blockSizeX, this.blockSizeY, 1],
                [
                    {
                        type: 'DevicePtr',
                        value: this.d_tf2Dtable.devicePtr
                    },{
                        type: 'Uint32',
                        value: this.transferSize
                    }
                ]
            );
            logger.debug('[INFO_CUDA_TF] cu.launch', error);

        },
        start : function(){

            if(this.transferFlag == 1 || this.transferFlag == 2){
                if(this.transferFlag == 2){
                    this.transferFlag = 0;
                }
                // OTF TextureBinding
                logger.debug('[INFO_CUDA] TEXTURE memory COPY');
                var error = this.cuModule.memOTFTextureAlloc(this.transferStart, this.transferMiddle1,this.transferMiddle2, this.transferEnd);
                //logger.debug('[INFO_CUDA] _cuModule.memOTFTextureAlloc', error);
                this.make2Dtable();
             }

            // ~ 3D volume array
            this.d_output = cu.memAlloc(this.imageWidth * this.imageHeight * 4);
            logger.debug('[INFO_CUDA] cu.memAlloc', this.d_output);
            var error = this.d_output.memSet(this.imageWidth * this.imageHeight * 4);
            logger.debug('[INFO_CUDA] d_output.memSet', error);

            // ~ View Vector
            this.makeViewVector();

            // ~ rendering
            this.render();

        },
        makeViewVector : function(){
            var vec;
            var model_matrix = mat4.create();
            if(this.type == ENUMS.RENDERING_TYPE.MPR ) {
                if (this.mprType == ENUMS.MPR_TYPE.X) {
                    vec = vec3.fromValues(-1.0, 0.0, 0.0);
                    mat4.rotate(model_matrix, model_matrix, ( (270.0) * 3.14159265 / 180.0), vec);

                    vec = vec3.fromValues(0.0, 1.0, 0.0);
                    mat4.rotate(model_matrix, model_matrix, ( (- 90) * 3.14159265 / 180.0), vec);

                    vec = vec3.fromValues(0, 0, this.positionZ);
                    mat4.translate(model_matrix, model_matrix, vec);

                }else if(this.mprType == ENUMS.MPR_TYPE.Y){
                    vec = vec3.fromValues(-1.0, 0.0, 0.0);
                    mat4.rotate(model_matrix, model_matrix, ( (270.0 ) * 3.14159265 / 180.0), vec);

                    vec = vec3.fromValues(0.0, 1.0, 0.0);
                    mat4.rotate(model_matrix, model_matrix, ( (0.0 ) * 3.14159265 / 180.0), vec);

                    vec = vec3.fromValues(0, 0, this.positionZ);
                    mat4.translate(model_matrix, model_matrix, vec);

                }else if(this.mprType == ENUMS.MPR_TYPE.Z) {
                    vec = vec3.fromValues(-1.0, 0.0, 0.0);
                    mat4.rotate(model_matrix, model_matrix, ( (180 ) * 3.14159265 / 180.0), vec);

                    vec = vec3.fromValues(0.0, 1.0, 0.0);
                    mat4.rotate(model_matrix, model_matrix, ( (0.0 ) * 3.14159265 / 180.0), vec);

                    vec = vec3.fromValues(0, 0, this.positionZ);
                    mat4.translate(model_matrix, model_matrix, vec);
                }
            }else{
                vec = vec3.fromValues(-1.0, 0.0, 0.0);
                // TODO Temp rotation Value
                mat4.rotate(model_matrix, model_matrix, ( (270.0 + (this.rotationY * -1)) * 3.14159265 / 180.0), vec);

                vec = vec3.fromValues(0.0, 1.0, 0.0);
                mat4.rotate(model_matrix, model_matrix,( (0.0 + (this.rotationX*-1)) * 3.14159265 / 180.0), vec);

                vec = vec3.fromValues(-this.positionX, this.positionY, this.positionZ);
                mat4.translate(model_matrix, model_matrix,vec)
            }
            /*view vector*/
            this.c_invViewMatrix = new Buffer(12*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[0], 0*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[4], 1*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[8], 2*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[12], 3*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[1], 4*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[5], 5*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[9], 6*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[13], 7*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[2], 8*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[6], 9*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[10], 10*4);
            this.c_invViewMatrix.writeFloatLE( model_matrix[14], 11*4);

            this.d_invViewMatrix = cu.memAlloc(12*4);
            var error = this.d_invViewMatrix.copyHtoD(this.c_invViewMatrix);
            logger.debug('[INFO_CUDA] d_invViewMatrix.copyHtoD', error);
        },

        render : function(){
            var _cuModule = this.cuModule;

            // ~ Rendering
            var cuFunction = undefined;
            if(this.type == ENUMS.RENDERING_TYPE.VOLUME){
                cuFunction = _cuModule.getFunction('render_kernel_volume');
            }else if(this.type == ENUMS.RENDERING_TYPE.MIP){
                cuFunction = _cuModule.getFunction('render_kernel_MIP');
            }else if(this.type == ENUMS.RENDERING_TYPE.MPR){
                cuFunction = _cuModule.getFunction('render_kernel_MPR');
            }else{
                logger.debug('type not exist');
                // ~ do default
                cuFunction = _cuModule.getFunction('render_kernel_volume');
            }
            logger.debug('[INFO_CUDA] cuFunction', cuFunction);

            //cuLaunchKernel
            var error = cu.launch(
                cuFunction, [iDivUp(this.imageWidth,this.blockSizeX), iDivUp(this.imageHeight, this.blockSizeY), 1], [this.blockSizeX, this.blockSizeY, 1],
                [
                    {
                        type: 'DevicePtr',
                        value: this.d_output.devicePtr
                    },{
                        type: 'DevicePtr',
                        value: this.d_invViewMatrix.devicePtr
                    },{
                        type: 'DevicePtr',
                        value: this.d_tf2Dtable.devicePtr
                    },{
                        type: 'Uint32',
                        value: this.imageWidth
                    },{
                        type: 'Uint32',
                        value: this.imageHeight
                    },{
                        type: 'Float32',
                        value: this.brightness
                    },{
                        type: 'Float32',
                        value: this.transferScaleX
                    },{
                        type: 'Float32',
                        value: this.transferScaleY
                    },{
                        type: 'Float32',
                        value: this.transferScaleZ
                    }, {
                        type: 'Uint32',
                        value: this.quality
                    },{
                        type: 'Uint32',
                        value: this.transferStart
                    }
                ]
            );
            logger.debug('[INFO_CUDA] cu.launch', error);

            // cuMemcpyDtoH
            this.d_outputBuffer = new Buffer(this.imageWidth * this.imageHeight * 4 );
            this.d_output.copyDtoH(this.d_outputBuffer, false);
        },

        end : function() {
            this.d_output.free();
            if (this.transferFlag == 1) {
                this.d_tf2Dtable.free();
                logger.debug('[INFO_CUDA] TEXTURE memory FREE');
            } else {
                logger.debug('[INFO_CUDA] TEXTURE memory NOT_FREE');
            }
            delete this.c_invViewMatrix;
            this.d_invViewMatrix.free();
            //this.cuModule.destroyOtfTexRef(); //1차 tf 메모리 관리
        },

        destroy : function() {
            var _cuModule = this.cuModule;
             _cuModule.destroyTexRef();
        }

    };

    return CudaRender;

}));
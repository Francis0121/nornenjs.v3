/**
 * Created by pi on 15. 4. 2.
 */

var EVENT_MESSAGE = {

    ANDROID : {
        ROTATION : 'rotation',
        TRANSLATION : 'translation',
        PINCHZOOM : 'pinchZoom',
        PNG : 'androidPng',
        MPR : 'mpr',
        OTF : 'OTF',
        BRIGHT : 'Brightness'
    },
    
    WEB : {
        LEFT_CLICK : 'leftMouse',
        RIGHT_CLICK : 'rightMouse',
        WHEEL_SCALE : 'wheelScale',
        BRIGHT_EVENT : 'brightBtn',
        SIZE_EVENT : 'sizeBtn',
        PNG : 'webPng',
        OTF_EVENT : 'otfEvent',
        TRANSFER_SCALE_X_EVENT : 'transferScaleXEvent',
        TRANSFER_SCALE_Y_EVENT : 'transferScaleYEvent',
        TRANSFER_SCALE_Z_EVENT : 'transferScaleZEvent'
    },

    TIZEN : {
        REQUEST : 'tizenRequest'
    }
};

module.exports = EVENT_MESSAGE;
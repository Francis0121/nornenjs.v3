function LayoutFunction(){
    this.isSignInFull = this.isSignInFull();
    this.isJoinFull = this.isJoinFull();
    this.isForgotFull  = this.isForgotFull();
    this.isNaigation = this.isNavigation();
    this.isRenderingPage = this.isRenderingPage();
    this.isRenderingBig = true;
};

LayoutFunction.prototype.isSignInFull = function(){
    return $('.layoutSignInArticle').html() != undefined;
};

LayoutFunction.prototype.signInResize = function(){
    var windowHeight = $(window).height();
    var articleHeight = $('.layoutSignInArticle').height();
    var margin = (windowHeight - articleHeight)/2;
    $('.layoutSignInArticle').css({
        'margin-top' : margin-30,
        'margin-bottom' : margin+30
    });
};

LayoutFunction.prototype.isJoinFull = function(){
    return $('.layoutJoinArticle').html() != undefined;
};

LayoutFunction.prototype.joinResize = function(){
    var windowHeight = $(window).height();
    var articleHeight = $('.layoutJoinArticle').height();
    var margin = (windowHeight - articleHeight)/2;
    $('.layoutJoinArticle').css({
        'margin-top' : margin-30,
        'margin-bottom' : margin+30
    });
};

LayoutFunction.prototype.isForgotFull = function(){
    return $('.layoutForgotArticle').html() != undefined;
};

LayoutFunction.prototype.forgotResize = function(){
    var windowHeight = $(window).height();
    var articleHeight = $('.layoutForgotArticle').height();
    var margin = (windowHeight - articleHeight)/2;
    $('.layoutForgotArticle').css({
        'margin-top' : margin-30,
        'margin-bottom' : margin+30
    });
};

LayoutFunction.prototype.isNavigation = function(){
    return $('#navigation').html() != undefined;
};

LayoutFunction.prototype.menuEventListener = function(){
    $('.menu>.title>a').off('click').on('click', function(){
        var menu = $(this).parents('ul.menu'),
            submenu = menu.find('ul.submenu');
        submenu.toggle('blind', { }, 500 );
    });
};

LayoutFunction.prototype.isRenderingPage = function(){
    return $('.volumeRenderingOne').html() != undefined;
};

LayoutFunction.prototype.renderingPageResize = function(){
    $('.layoutContentSection').css({
        'overflow-y' : 'hidden'
    });
    
    if(this.isRenderingBig) {
        var windowHeight = $(window).height(),
            height = windowHeight / 2 - 3;

        var windowWidth = $(window).width(),
            navWidth = $('#navigation').width(),
            width = (windowWidth - navWidth) / 2 - 3;

        $('.volumeRenderingOne').width(width).height(height);
        $('.volumeRenderingOne>canvas').attr('width', height).attr('height', height);

        var mprXCanvas = document.getElementById('volumeMprXCanvas');
        var mprXCtx = mprXCanvas.getContext('2d');
        var mprXImg = new Image(512, 512);
        mprXImg.onload = function(){
            mprXCtx.drawImage(mprXImg, 0, 0, 512, 512, 0, 0, mprXCanvas.clientWidth, mprXCanvas.clientWidth);
        };
        mprXImg.src = mprXImgUrl;

        var mprYCanvas = document.getElementById('volumeMprYCanvas');
        var mprYCtx = mprYCanvas.getContext('2d');
        var mprYImg = new Image(512, 512);
        mprYImg.onload = function(){
            mprYCtx.drawImage(mprYImg, 0, 0, 512, 512, 0, 0, mprYCanvas.clientWidth, mprYCanvas.clientWidth);
        };
        mprYImg.src = mprYImgUrl;

        var mprZCanvas = document.getElementById('volumeMprZCanvas');
        var mprZCtx = mprZCanvas.getContext('2d');
        var mprZImg = new Image(512, 512);
        mprZImg.onload = function(){
            mprZCtx.drawImage(mprZImg, 0, 0, 512, 512, 0, 0, mprZCanvas.clientWidth, mprZCanvas.clientWidth);
        };
        mprZImg.src = mprZImgUrl;
    }else{
        var windowHeight = $(window).height(),
            height = windowHeight - 250;

        var windowWidth = $(window).width(),
            navWidth = $('#navigation').width(),
            width = windowWidth - navWidth - 4;

        $('#volumeRendering').width(width).height(height);
        $('#volumeRendering>canvas').attr('width', height).attr('height', height);

        $('.volumeRenderingOTF').width(width-20);
        $('.volumeRenderingOTF>svg').attr('width', width-20);
    }
};

LayoutFunction.prototype.typeChangeEventListener = function(){
    if(this.isRenderingBig){
        return;
    }

    if(renderingType == 1) {
        var windowHeight = $(window).height(),
            height = windowHeight - 250;

        var windowWidth = $(window).width(),
            navWidth = $('#navigation').width(),
            width = windowWidth - navWidth - 4;

        $('.volumeRenderingMpr').hide();
        $('.volumeRenderingOTF').show().width(width - 20);
        $('.volumeRenderingOTF>svg').attr('width', width - 20);

        $('#volumeRendering').removeClass('volumeRenderingOne').addClass('volumeRenderingBig').width(width).height(height);
        $('#volumeRendering>canvas').attr('width', height).attr('height', height);
        $('#renderingSizeBtn').removeClass('renderingExpandBtn').addClass('renderingReduceBtn');
    }else{
        var windowHeight = $(window).height();

        var windowWidth = $(window).width(),
            navWidth = $('#navigation').width(),
            width = windowWidth - navWidth - 4;

        $('.volumeRenderingMpr').hide();
        $('.volumeRenderingOTF').hide();

        $('#volumeRendering').removeClass('volumeRenderingOne').addClass('volumeRenderingBig').width(width).height(windowHeight);
        $('#volumeRendering>canvas').attr('width', windowHeight).attr('height', windowHeight);
        $('#renderingSizeBtn').removeClass('renderingExpandBtn').addClass('renderingReduceBtn');
    }
};

LayoutFunction.prototype.expandEventListener = function(){
    if(this.isRenderingBig) {
        this.isRenderingBig = false;

        if(renderingType == 1) {
            var windowHeight = $(window).height(),
                height = windowHeight - 250;

            var windowWidth = $(window).width(),
                navWidth = $('#navigation').width(),
                width = windowWidth - navWidth - 4;

            $('.volumeRenderingMpr').hide();
            $('.volumeRenderingOTF').show().width(width - 20);
            $('.volumeRenderingOTF>svg').attr('width', width - 20);

            $('#volumeRendering').removeClass('volumeRenderingOne').addClass('volumeRenderingBig').width(width).height(height);
            $('#volumeRendering>canvas').attr('width', height).attr('height', height);
            $('#renderingSizeBtn').removeClass('renderingExpandBtn').addClass('renderingReduceBtn');
        }else{
            var windowHeight = $(window).height();

            var windowWidth = $(window).width(),
                navWidth = $('#navigation').width(),
                width = windowWidth - navWidth - 4;

            $('.volumeRenderingMpr').hide();
            $('.volumeRenderingOTF').hide();

            $('#volumeRendering').removeClass('volumeRenderingOne').addClass('volumeRenderingBig').width(width).height(windowHeight);
            $('#volumeRendering>canvas').attr('width', windowHeight).attr('height', windowHeight);
            $('#renderingSizeBtn').removeClass('renderingExpandBtn').addClass('renderingReduceBtn');
        }

    }else{
        this.isRenderingBig = true;
        var windowHeight = $(window).height(),
            height = windowHeight/2-3;

        var windowWidth = $(window).width(),
            navWidth = $('#navigation').width(),
            width = (windowWidth - navWidth)/2 - 3;
        
        $('.volumeRenderingMpr').show();
        $('.volumeRenderingOTF').hide();
        $('#volumeRendering').addClass('volumeRenderingOne').removeClass('volumeRenderingBig');
        $('#renderingSizeBtn').addClass('renderingExpandBtn').removeClass('renderingReduceBtn');
        
        $('.volumeRenderingOne').width(width).height(height);
        $('.volumeRenderingOne>canvas').attr('width', height).attr('height', height);

        var mprXCanvas = document.getElementById('volumeMprXCanvas');
        var mprXCtx = mprXCanvas.getContext('2d');
        var mprXImg = new Image(512, 512);
        mprXImg.onload = function(){
            mprXCtx.drawImage(mprXImg, 0, 0, 512, 512, 0, 0, mprXCanvas.clientWidth, mprXCanvas.clientWidth);
        };
        mprXImg.src = mprXImgUrl;

        var mprYCanvas = document.getElementById('volumeMprYCanvas');
        var mprYCtx = mprYCanvas.getContext('2d');
        var mprYImg = new Image(512, 512);
        mprYImg.onload = function(){
            mprYCtx.drawImage(mprYImg, 0, 0, 512, 512, 0, 0, mprYCanvas.clientWidth, mprYCanvas.clientWidth);
        };
        mprYImg.src = mprYImgUrl;

        var mprZCanvas = document.getElementById('volumeMprZCanvas');
        var mprZCtx = mprZCanvas.getContext('2d');
        var mprZImg = new Image(512, 512);
        mprZImg.onload = function(){
            mprZCtx.drawImage(mprZImg, 0, 0, 512, 512, 0, 0, mprZCanvas.clientWidth, mprZCanvas.clientWidth);
        };
        mprZImg.src = mprZImgUrl;
    }
    socket.emit('webPng', {type : 0, renderingType : renderingType});
};

var layoutFunction = new LayoutFunction();

$(function(){

    $('#navigationBtn').on('click', function(){
        if($('#navigation').hasClass('navigation')){
            $('#navigation').addClass('navigationSmall').removeClass('navigation');
            $('.layoutContentSection').css({ 'margin-left': '40px'});
        }else{
            $('#navigation').addClass('navigation').removeClass('navigationSmall');
            $('.layoutContentSection').css({ 'margin-left': '200px'});
        }
        layoutFunction.renderingPageResize();
        if(socket != null){
            socket.emit('webPng', {type : 0, renderingType : renderingType});
        }
    });
    
    if(layoutFunction.isSignInFull) {
        layoutFunction.signInResize();
        $(window).resize(function () {
            layoutFunction.signInResize();
        });
    }

    if(layoutFunction.isJoinFull) {
        layoutFunction.joinResize();
        $(window).resize(function () {
            layoutFunction.joinResize();
        });
    }
    
    if(layoutFunction.isForgotFull){
        layoutFunction.forgotResize();
        $(window).resize(function(){
            layoutFunction.forgotResize();
        });
    }
    
    if(layoutFunction.isNaigation){
        layoutFunction.menuEventListener();
    }
    
    if(layoutFunction.isRenderingPage){
        layoutFunction.renderingPageResize();
        $(window).resize(function(){
            layoutFunction.renderingPageResize();
        });
    }
    

    $('#volumeFilter #from').datepicker({
        changeMonth: true,
        numberOfMonths: 1,
        dateFormat: 'yy-mm-dd',
        onClose: function( selectedDate ) {
            $('#volumeFilter #to').datepicker('option', 'minDate', selectedDate);
        },
        onSelect : function(){
        }
    });

    $('#volumeFilter #to').datepicker({
        changeMonth: true,
        numberOfMonths: 1,
        dateFormat : 'yy-mm-dd',
        onClose: function( selectedDate ) {
            $('#volumeFilter #from').datepicker('option', 'maxDate', selectedDate );
        },
        onSelect : function(){
        }
    });

    $('.layoutContentSection').scroll( function() {
        var section = $('.layoutContentSection');
        if(section[0].scrollHeight - section.scrollTop() == section.outerHeight()) {
            goToNextPages();
        }
    });

    // ~ STEP 01 볼륨을 업로드한다.
    $('#dataUpload').uploadify({
        'buttonText' : '파일선택',
        'buttonClass' : 'volumeDataBtn',
        'fileTypeDesc' : 'Volume Data',
        'fileTypeExts' : '*.den; *.data',
        'swf': contextPath + '/resources/javascript/uploadify.swf',
        'uploader' : contextPath+'/data/upload',
        'onUploadStart' : function(){
            //console.log('On Upload Start');
        },
        'onUploadSuccess' : function(file, data, response){
            //console.log('On Upload Success');
            if(response){
                // ~ STEP 02 섬네일 생성 요청을 한다.
                var data = JSON.parse(data)
                $('#volume #volumeDataPn').val(data.pn);
                $('#volumeUploadBtn').html('<span class="success">파일업로드 : '+data.name+'</span>');

                var url = contextPath + '/data/thumbnail';
                var json = {
                    volumeDataPn : data.pn,
                    width : $('#width').val(),
                    height : $('#height').val(),
                    depth : $('#depth').val()
                }

                $.postJSON(url, json, function(result){
                    // ~ STEP 03 Thumbnail 이 모두 생성되었는지 폴링한다. 비교값은 result 랑 연동
                    var pollingFunc = function() {
                        var polling = contextPath + '/data/polling/' + data.pn;
                        console.log('polling')
                        $.getJSON(polling, function (list) {
                            if(list.length != result.thumbnailOptionList.length){
                                setTimeout(pollingFunc, 1000);
                            }else{
                                console.log(list);
                                $('#thumbnailMPRx').attr('src', contextPath+'/data/thumbnail/'+list[1]);
                                $('#thumbnailMPRy').attr('src', contextPath+'/data/thumbnail/'+list[2]);
                                $('#thumbnailMPRz').attr('src', contextPath+'/data/thumbnail/'+list[3]);
                                $('#thumbnailMPRvolume').attr('src', contextPath+'/data/thumbnail/'+list[0]);
                            }
                        });
                    }
                    setTimeout(pollingFunc, 1000);
                });
            }
        }
    });
    
    $('#renderingSizeBtn').off('click').on('click', function(){
        layoutFunction.expandEventListener();
    });


    // ~ Left OTF EVENT

    var beforeX;

    var isLeftTopCircle = false;
    var isLeftBottomCircle = false;
    var isLeftDashLine = false;

    var isRightTopCircle = false;
    var isRightBottomCircle = false;
    var isRightDashLine = false;

    var isOtfTopLine = false;

    var otfOption = {
        transferStart : 65,
        transferMiddle1 : 80,
        transferMiddle2 : 100,
        transferEnd : 120,
        transferFlag : 0,
        isPng : false,
        type : 0
    };

    $(window).on('mousemove', function(event){

        if(isLeftTopCircle){
            if(event.pageX >= beforeX){ // +
                var cx = Number($('#otfLeftTopCircle').attr('cx'))+1;
                if(cx > $('#otfRightTopCircle').attr('cx')){
                    return;
                }
                $('#otfLeftTopCircle').attr('cx', cx);
                $('#otfTopLine').attr('x1', cx);
                $('#otfLeftDashLine').attr('x2', cx);

                otfOption.transferMiddle1 = Math.round(cx/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;

                socket.emit('otfEvent', otfOption);
            } else{ // -
                var cx = Number($('#otfLeftTopCircle').attr('cx'))-1;
                if(cx < $('#otfLeftBottomCircle').attr('cx')){
                    return;
                }
                $('#otfLeftTopCircle').attr('cx', cx);
                $('#otfTopLine').attr('x1', cx);
                $('#otfLeftDashLine').attr('x2', cx);

                otfOption.transferMiddle1 = Math.round(cx/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;

                socket.emit('otfEvent', otfOption);
            }
        }

        if(isLeftBottomCircle){
            if(event.pageX >= beforeX){ // +
                var cx = Number($('#otfLeftBottomCircle').attr('cx'))+1;
                if(cx > $('#otfLeftTopCircle').attr('cx')){
                    return;
                }

                $('#otfLeftBottomCircle').attr('cx', cx);
                $('#otfLeftDashLine').attr('x1', cx);

                otfOption.transferStart = Math.round(cx/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            } else{ // -
                var cx = Number($('#otfLeftBottomCircle').attr('cx'))-1;
                if(cx < 0){
                    return;
                }

                $('#otfLeftBottomCircle').attr('cx', cx);
                $('#otfLeftDashLine').attr('x1', cx);

                otfOption.transferStart = Math.round(cx/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            }
        }

        if(isLeftDashLine){
            if(event.pageX >= beforeX){ // +
                var cxTop = Number($('#otfLeftTopCircle').attr('cx'))+1;
                var cxBottom = Number($('#otfLeftBottomCircle').attr('cx'))+1
                if(cxTop > $('#otfRightTopCircle').attr('cx')){
                    return;
                }

                $('#otfTopLine').attr('x1', cxTop);
                $('#otfLeftTopCircle').attr('cx', cxTop);
                $('#otfLeftBottomCircle').attr('cx', cxBottom);

                $('#otfLeftDashLine').attr('x1', cxBottom);otfTopLine
                $('#otfLeftDashLine').attr('x2', cxTop);

                otfOption.transferMiddle1 = Math.round(cxTop/3);
                otfOption.transferStart = Math.round(cxBottom/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            } else{ // -
                var cxTop = Number($('#otfLeftTopCircle').attr('cx'))-1;
                var cxBottom = Number($('#otfLeftBottomCircle').attr('cx'))-1;
                if(cxBottom < 0){
                    return;
                }

                $('#otfTopLine').attr('x1', cxTop);
                $('#otfLeftTopCircle').attr('cx', cxTop);
                $('#otfLeftBottomCircle').attr('cx', cxBottom);

                $('#otfLeftDashLine').attr('x1', cxBottom);
                $('#otfLeftDashLine').attr('x2', cxTop);

                otfOption.transferMiddle1 = Math.round(cxTop/3);
                otfOption.transferStart = Math.round(cxBottom/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            }
        }

        if(isRightTopCircle){
            if(event.pageX >= beforeX){ // +
                var cx = Number($('#otfRightTopCircle').attr('cx'))+1;
                if(cx > $('#otfRightBottomCircle').attr('cx')){
                    return;
                }
                $('#otfRightTopCircle').attr('cx', cx);
                $('#otfTopLine').attr('x2', cx);
                $('#otfRightDashLine').attr('x1', cx);

                otfOption.transferMiddle2 = Math.round(cx/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            } else{ // -
                var cx = Number($('#otfRightTopCircle').attr('cx'))-1;
                if(cx < $('#otfLeftTopCircle').attr('cx')){
                    return;
                }
                $('#otfRightTopCircle').attr('cx', cx);
                $('#otfTopLine').attr('x2', cx);
                $('#otfRightDashLine').attr('x1', cx);

                otfOption.transferMiddle2 = Math.round(cx/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            }
        }

        if(isRightBottomCircle){
            if(event.pageX >= beforeX){ // +
                var cx = Number($('#otfRightBottomCircle').attr('cx'))+1;
                if(cx > 768){
                    return;
                }

                $('#otfRightBottomCircle').attr('cx', cx);
                $('#otfRightDashLine').attr('x2', cx);

                otfOption.transferEnd = Math.round(cx/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            } else{ // -
                var cx = Number($('#otfRightBottomCircle').attr('cx'))-1;
                if(cx < $('#otfRightTopCircle').attr('cx')){
                    return;
                }
                $('#otfRightBottomCircle').attr('cx', cx);
                $('#otfRightDashLine').attr('x2', cx);

                otfOption.transferEnd = Math.round(cx/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            }
        }

        if(isRightDashLine){
            if(event.pageX >= beforeX){ // +
                var cxTop = Number($('#otfRightTopCircle').attr('cx'))+1;
                var cxBottom = Number($('#otfRightBottomCircle').attr('cx'))+1;
                if(cxBottom > 768){
                    return;
                }

                $('#otfTopLine').attr('x2', cxTop);
                $('#otfRightTopCircle').attr('cx', cxTop);
                $('#otfRightBottomCircle').attr('cx', cxBottom);

                $('#otfRightDashLine').attr('x2', cxBottom);
                $('#otfRightDashLine').attr('x1', cxTop);

                otfOption.transferMiddle2 = Math.round(cxTop/3);
                otfOption.transferEnd = Math.round(cxBottom/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            } else{ // -
                var cxTop = Number($('#otfRightTopCircle').attr('cx'))-1;
                var cxBottom = Number($('#otfRightBottomCircle').attr('cx'))-1;
                if(cxTop < $('#otfLeftTopCircle').attr('cx')){
                    return;
                }

                $('#otfTopLine').attr('x2', cxTop);
                $('#otfRightTopCircle').attr('cx', cxTop);
                $('#otfRightBottomCircle').attr('cx', cxBottom);

                $('#otfRightDashLine').attr('x2', cxBottom);
                $('#otfRightDashLine').attr('x1', cxTop);
                otfOption.transferMiddle2 = Math.round(cxTop/3);
                otfOption.transferEnd = Math.round(cxBottom/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            }
        }

        if(isOtfTopLine){
            if(event.pageX >= beforeX){ // +
                var cxLeftTop = Number($('#otfLeftTopCircle').attr('cx'))+2;
                var cxRightTop = Number($('#otfRightTopCircle').attr('cx'))+2;
                var cxLeftBottom = Number($('#otfLeftBottomCircle').attr('cx'))+2;
                var cxRightBottom = Number($('#otfRightBottomCircle').attr('cx'))+2;
                if(cxRightBottom > 768){
                    return;
                }

                $('#otfTopLine').attr('x1', cxLeftTop);
                $('#otfTopLine').attr('x2', cxRightTop);

                $('#otfLeftTopCircle').attr('cx', cxLeftTop);
                $('#otfRightTopCircle').attr('cx', cxRightTop);
                $('#otfLeftBottomCircle').attr('cx', cxLeftBottom);
                $('#otfRightBottomCircle').attr('cx', cxRightBottom);

                $('#otfLeftDashLine').attr('x1', cxLeftBottom)
                $('#otfLeftDashLine').attr('x2', cxLeftTop);

                $('#otfRightDashLine').attr('x2', cxRightBottom);
                $('#otfRightDashLine').attr('x1', cxRightTop);

                otfOption.transferStart = Math.round(cxLeftBottom/3);
                otfOption.transferMiddle1 = Math.round(cxLeftTop/3);
                otfOption.transferMiddle2 = Math.round(cxRightTop/3);
                otfOption.transferEnd = Math.round(cxRightBottom/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            } else{ // -
                var cxLeftTop = Number($('#otfLeftTopCircle').attr('cx'))-2;
                var cxRightTop = Number($('#otfRightTopCircle').attr('cx'))-2;
                var cxLeftBottom = Number($('#otfLeftBottomCircle').attr('cx'))-2;
                var cxRightBottom = Number($('#otfRightBottomCircle').attr('cx'))-2;
                if(cxLeftBottom < 0){
                    return;
                }

                $('#otfTopLine').attr('x1', cxLeftTop);
                $('#otfTopLine').attr('x2', cxRightTop);

                $('#otfLeftTopCircle').attr('cx', cxLeftTop);
                $('#otfRightTopCircle').attr('cx', cxRightTop);
                $('#otfLeftBottomCircle').attr('cx', cxLeftBottom);
                $('#otfRightBottomCircle').attr('cx', cxRightBottom);

                $('#otfLeftDashLine').attr('x1', cxLeftBottom)
                $('#otfLeftDashLine').attr('x2', cxLeftTop);

                $('#otfRightDashLine').attr('x2', cxRightBottom);
                $('#otfRightDashLine').attr('x1', cxRightTop);

                otfOption.transferStart = Math.round(cxLeftBottom/3);
                otfOption.transferMiddle1 = Math.round(cxLeftTop/3);
                otfOption.transferMiddle2 = Math.round(cxRightTop/3);
                otfOption.transferEnd = Math.round(cxRightBottom/3);
                otfOption.transferFlag = 1;
                otfOption.isPng = false;
                socket.emit('otfEvent', otfOption);
            }
        }

        beforeX = event.pageX;
    });


    $(window).on('mouseup', function(event){
        if(isLeftTopCircle || isLeftBottomCircle || isLeftDashLine ||
            isRightTopCircle || isRightBottomCircle || isRightDashLine ||
            isOtfTopLine){

            isLeftTopCircle = false;
            isLeftBottomCircle = false;
            isLeftDashLine = false;

            isRightTopCircle = false;
            isRightBottomCircle = false;
            isRightDashLine = false;

            isOtfTopLine = false;

            otfOption.transferFlag = 2;
            otfOption.isPng = true;
            socket.emit('otfEvent', otfOption);
        }
    });


    $('#otfLeftTopCircle').on('mousedown',function(event){
        isLeftTopCircle = true;
        beforeX = event.pageX;
    });

    $('#otfLeftBottomCircle').on('mousedown',function(event){
        isLeftBottomCircle = true;
        otfOption.transferStart = Math.round(cx/3);
        beforeX = event.pageX;
    });

    $('#otfLeftDashLine').on('mousedown',function(event){
        isLeftDashLine = true;
        beforeX = event.pageX;
    });

    // ~ RightEvent

    $('#otfRightTopCircle').on('mousedown',function(event){
        isRightTopCircle = true;
        beforeX = event.pageX;
    });


    $('#otfRightBottomCircle').on('mousedown',function(event){
        isRightBottomCircle = true;
        beforeX = event.pageX;
    });


    $('#otfRightDashLine').on('mousedown',function(event){
        isRightDashLine = true;
        beforeX = event.pageX;
    });

    // ~ TopLine

    $('#otfTopLine').on('mousedown', function(event){
        isOtfTopLine = true;
        beforeX = event.pageX;
    });

    

});
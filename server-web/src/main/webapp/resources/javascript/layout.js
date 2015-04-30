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
    return $('.navigation').html() != undefined;
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
            navWidth = $('.navigation').width(),
            width = (windowWidth - navWidth) / 2 - 3;

        $('.volumeRenderingOne').width(width).height(height);
        $('.volumeRenderingOne>img').width(height).height(height);
    }else{
        var windowHeight = $(window).height(),
            height = windowHeight - 250;

        var windowWidth = $(window).width(),
            navWidth = $('.navigation').width(),
            width = windowWidth - navWidth - 4;

        $('#volumeRendering').width(width).height(height);
        $('#volumeRendering>img').width(height).height(height);
        $('.volumeRenderingOTF').width(width);
    }
};

LayoutFunction.prototype.expandEventListener = function(){
    if(this.isRenderingBig) {
        this.isRenderingBig = false;
        var windowHeight = $(window).height(),
            height = windowHeight - 250;

        var windowWidth = $(window).width(),
            navWidth = $('.navigation').width(),
            width = windowWidth - navWidth - 4;

        $('.volumeRenderingMpr').hide();
        $('.volumeRenderingOTF').show().width(width);
        
        $('#volumeRendering').removeClass('volumeRenderingOne').addClass('volumeRenderingBig').width(width).height(height);
        $('#volumeRendering>img').width(height).height(height);
        $('#renderingSizeBtn').removeClass('renderingExpandBtn').addClass('renderingReduceBtn');

    }else{
        this.isRenderingBig = true;
        var windowHeight = $(window).height(),
            height = windowHeight/2-3;

        var windowWidth = $(window).width(),
            navWidth = $('.navigation').width(),
            width = (windowWidth - navWidth)/2 - 3;
        
        $('.volumeRenderingMpr').show();
        $('.volumeRenderingOTF').hide();
        $('#volumeRendering').addClass('volumeRenderingOne').removeClass('volumeRenderingBig');
        $('#renderingSizeBtn').addClass('renderingExpandBtn').removeClass('renderingReduceBtn');
        
        $('.volumeRenderingOne').width(width).height(height);
        $('.volumeRenderingOne>img').width(height).height(height);
    }
};

$(function(){
    var layoutFunction = new LayoutFunction();
    
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

    var volumeListSlider = $('.volumeListSlider').bxSlider({
        'preloadImages' : 'all',
        'pager' : false
    });

    $('.volumeListSlider').css({
        'max-height' : $('.volumeListSlider>li>img').height(),
        'border-top' : 0
    });

    $('.layoutContentSection').scroll( function() {
        var section = $('.layoutContentSection');
        if(section[0].scrollHeight - section.scrollTop() == section.outerHeight()) {
            for(var i=0; i<4; i++) {
                var one = $('<li>').addClass('one').append($('.volumeListArticle>.list>.one').html());
                $('.volumeListArticle>ul').append(one);
            }
        }
    });

    $('#dataUpload').uploadify({
        'buttonText' : '파일선택',
        'buttonClass' : 'volumeDataBtn',
        'fileTypeDesc' : 'Volume Data',
        'fileTypeExts' : '*.den; *.data',
        'swf': contextPath + '/resources/javascript/uploadify.swf',
        'uploader' : contextPath+'/data/upload',
        'onUploadStart' : function(){
            console.log('On Upload Start');
        },
        'onUploadSuccess' : function(file, data, response){
            console.log('On Upload Success');
            if(response){
                var data = JSON.parse(data)
                $('#volume #volumeDataPn').val(data.pn);
                $('#volumeUploadBtn').html('<span class="success">파일업로드 : '+data.name+'</span>');
                console.log(JSON.stringify(file));
            }
        }
    });
    
    $('#renderingSizeBtn').off('click').on('click', function(){
        layoutFunction.expandEventListener();
    });

    var isTopCircle = false;
    var isBottomCircle = false;
    var isDashLine = false;
    var beforeX;
    
    $('#otfTopCircle').on('mousedown',function(event){
        isTopCircle = true;
        beforeX = event.pageX;
    });
    
    $('#otfTopCircle').on('mousemove',function(event){
        var circle = event.target;
        if(isTopCircle){
            if(event.pageX >= beforeX){ // +
                var cx = Number(circle.getAttribute('cx'))+2;
                circle.setAttribute('cx', cx);
                $('#otfTopLine').attr('x1', cx);
                $('#otfDashLine').attr('x2', cx);
            } else{ // -
                var cx = Number(circle.getAttribute('cx'))-2;
                circle.setAttribute('cx', cx);
                $('#otfTopLine').attr('x1', cx);
                $('#otfDashLine').attr('x2', cx);
            }
        }
        beforeX = event.pageX;
    });

    $('#otfBottomCircle').on('mousedown',function(event){
        isBottomCircle = true;
        beforeX = event.pageX;
    });
    
    $('#otfBottomCircle').on('mousemove',function(event){
        var circle = event.target;
        if(isBottomCircle){
            if(event.pageX >= beforeX){ // +
                var cx = Number(circle.getAttribute('cx'))+2;
                circle.setAttribute('cx', cx);
                $('#otfDashLine').attr('x1', cx);
            } else{ // -
                var cx = Number(circle.getAttribute('cx'))-2;
                circle.setAttribute('cx', cx);
                $('#otfDashLine').attr('x1', cx);
            }
        }
        beforeX = event.pageX;
    });

    $('#otfDashLine').on('mousedown',function(event){
        isDashLine = true;
        beforeX = event.pageX;
    });

    $('#otfDashLine').on('mousemove',function(event){

        if(isDashLine){
            if(event.pageX >= beforeX){ // +
                var cxTop = Number($('#otfTopCircle').attr('cx'))+2;
                var cxBottom = Number($('#otfBottomCircle').attr('cx'))+2

                $('#otfTopLine').attr('x1', cxTop);
                $('#otfTopCircle').attr('cx', cxTop);
                $('#otfBottomCircle').attr('cx', cxBottom);
                
                $('#otfDashLine').attr('x1', cxBottom);
                $('#otfDashLine').attr('x2', cxTop);
            } else{ // -
                var cxTop = Number($('#otfTopCircle').attr('cx'))-2;
                var cxBottom = Number($('#otfBottomCircle').attr('cx'))-2

                $('#otfTopLine').attr('x1', cxTop);
                $('#otfTopCircle').attr('cx', cxTop);
                $('#otfBottomCircle').attr('cx', cxBottom);

                $('#otfDashLine').attr('x1', cxBottom);
                $('#otfDashLine').attr('x2', cxTop);
            }
        }
        beforeX = event.pageX;
    });
    
    $(window).on('mouseup', function(event){
        isTopCircle = false;
        isBottomCircle = false;
        isDashLine = false;
    })
    
});
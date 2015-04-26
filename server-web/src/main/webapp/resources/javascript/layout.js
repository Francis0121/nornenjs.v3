function LayoutFunction(){
    this.isSignInFull = this.isSignInFull();
    this.isJoinFull = this.isJoinFull();
    this.isForgotFull  = this.isForgotFull();
    this.isNaigation = this.isNavigation();
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
    
});
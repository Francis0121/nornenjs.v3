function LayoutFunction(){
    this.isSignInFull = this.isSignInFull();
    this.isJoinFull = this.isJoinFull();
};

LayoutFunction.prototype.isSignInFull = function(){
    return $('.layoutSignInArticle').html() != undefined;
};

LayoutFunction.prototype.signInResize = function(){
    var windowHeight = $(window).height();
    $('.layoutFullSection').height(windowHeight);

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
    $('.layoutFullSection').height(windowHeight);

    var articleHeight = $('.layoutJoinArticle').height();
    var margin = (windowHeight - articleHeight)/2;
    $('.layoutJoinArticle').css({
        'margin-top' : margin-30,
        'margin-bottom' : margin+30
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
});
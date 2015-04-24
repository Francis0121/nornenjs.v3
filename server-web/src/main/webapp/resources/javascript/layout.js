function LayoutFunction(){
    this.isSignInFull = this.isSignInFull();
    this.isJoinInFull = this.isJoinFull();
};

LayoutFunction.prototype.isSignInFull = function(){
    return $('.layoutSignInArticle').html() != undefined;
};

LayoutFunction.prototype.signInResize = function(){
    if(!this.isSignInFull){
        return;
    }
    var windowHeight = $(window).height();
    $('.layoutFullSection').height(windowHeight);

    var articleHeight = $('.layoutSignInArticle').height();
    var margin = (windowHeight - articleHeight)/2;
    $('.layoutSignInArticle').css({
        'margin-top' : margin-30,
        'margin-bottom' : margin+30
    });

    $(window).resize(function () {
        layoutFunction.signInResize();
    });
};

LayoutFunction.prototype.isJoinFull = function(){
    return $('.layoutJoinArticle').html() != undefined;
};

LayoutFunction.prototype.joinResize = function(){
    if(!this.isJoinInFull){
        return;
    }
    
    var windowHeight = $(window).height();
    $('.layoutFullSection').height(windowHeight);

    var articleHeight = $('.layoutJoinArticle').height();
    var margin = (windowHeight - articleHeight)/2;
    $('.layoutJoinArticle').css({
        'margin-top' : margin-30,
        'margin-bottom' : margin+30
    });

    $(window).resize(function () {
        layoutFunction.joinResize();
    });
};

$(function(){
    var layoutFunction = new LayoutFunction();
    layoutFunction.signInResize();
    layoutFunction.joinResize();
});
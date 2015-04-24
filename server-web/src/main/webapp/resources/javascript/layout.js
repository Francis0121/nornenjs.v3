function LayoutFunction(){
    this.isLayoutFull = this.isFull();
};

LayoutFunction.prototype.isFull = function(){
    return $('.layoutFullSection').html() != undefined;
};

LayoutFunction.prototype.resize = function(){
    var windowHeight = $(window).height();
    $('.layoutFullSection').height(windowHeight);

    var articleHeight = $('.layoutFullArticle').height();
    var margin = (windowHeight - articleHeight)/2;
    $('.layoutFullArticle').css({
        'padding-top' : margin-30,
        'padding-bottom' : margin+30
    });
};

$(function(){
    var layoutFunction = new LayoutFunction();
    layoutFunction.resize();
});
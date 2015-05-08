<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ include file="../../layout/header.jspf" %>
<%@ include file="../../layout/nav.jspf" %>
<c:set var="pagination" value="${volumeFilter.pagination }"/>
<script type="text/javascript">
    /* <![CDATA[ */
    var currentPage = <c:out value='${pagination.currentPage}'/>;
    var numPages = <c:out value='${pagination.numPages}'/>;

    function goToNextPages() {
        if(currentPage == numPages){
            return;
        }
        goToPage(Math.min(numPages, currentPage + 1));
    }

    function goToPage(page) {

        var form = document.forms['volumeFilter'];
        console.log('Form', form);
        console.log('Page',page);

        var url = contextPath + '/volume/list/json'
        var volumeFilter = {
            page : page,
            username : form.username.value,
            title : form.title.value,
            from : form.from.value,
            to : form.to.value
        };

        $.postJSON(url, volumeFilter, function(data){
            var volumes = data.volumes;

            currentPage+=1;

            var list = $('.volumeListArticle>.list');
            var html = '';
            for(var i=0; i<volumes.length; i++){
                var volume = volumes[i];
                html+='<li class="one" data-pn="'+volume.pn+'">';
                html+='<figure>';
                html+=' <ul class="volumeListSlider">';
                for(var j=0; j<volume.thumbnailPnList.length; j++){
                    var thumbnailPn = volume.thumbnailPnList[j];
                    html+=' <li><img src="'+contextPath+'/data/thumbnail/'+thumbnailPn+'"/></li>';
                }
                html+=' </ul>';

                html+=' <figcaption>';
                html+='     <a href="'+contextPath+'/volume/'+volume.pn+'" class="name">'+volume.title+'</a> <a href="'+contextPath+'/volume/page/'+volume.pn+'" class="volumeUpdateBtn">수정</a><br/>';
                html+='     <span class="number">'+volume.width+' x '+volume.height+' x '+volume.depth+'</span> <span class="date">'+volume.inputDate+'</span>';
                html+=' </figcaption>';
                html+='</figure>';
                html+='</li>';
            }
            list.append(html);

            volumeListSlider.reloadSlider();
//            volumeListSlider = $('.volumeListSlider').bxSlider({
//                'pager' : false
//            });
//
//            $('.volumeListSlider').css({
//                'max-height' : $('.volumeListSlider>li>img').height(),
//                'border-top' : 0
//            });
        });

    }
    /* ]]> */
</script>

<section class="layoutContentSectionWrap">
    
    <section class="layoutContentSection">

        <form:form commandName="volumeFilter" htmlEscape="true" cssClass="volumeFilterForm">
            <form:hidden path="username"/>
            <ul>
                <li class="searchIcon">
                    <img src="${cp}/resources/image/icon/search.png"/>
                </li>
                <li>
                    <form:input path="title" cssClass="search" placeholder="검색"/>
                </li>
                <li class="dateIcon">
                    <img src="${cp}/resources/image/icon/calendar.png"/>
                </li>
                <li>
                    <form:input path="from" readonly="true" cssClass="date"/>
                </li>
                <li>
                    <form:input path="to" readonly="true" cssClass="date"/>
                </li>
            </ul>
            
        </form:form>
        
        <article class="volumeListArticle">
            
            <ul class="list">
                <c:forEach items="${volumes}" var="volume" varStatus="loop">
                <li class="one" data-pn="<c:out value="${volume.pn}"/>">
                    <figure>
                        <ul class="volumeListSlider">
                            <c:forEach items="${volume.thumbnailPnList}" var="thumbnailPn" varStatus="loop">
                                <li><img src="${cp}/data/thumbnail/${thumbnailPn}" /></li>
                            </c:forEach>
                        </ul>
                        
                        <figcaption>
                            <a href="${cp}/volume/${volume.pn}" class="name"><c:out value="${volume.title}"/></a> <a href="${cp}/volume/page/${volume.pn}" class="volumeUpdateBtn">수정</a><br/>
                            <span class="number"><c:out value="${volume.width}"/> x <c:out value="${volume.height}"/> x <c:out value="${volume.depth}"/></span> <span class="date"><c:out value="${volume.inputDate}"/></span>
                        </figcaption>
                    </figure>
                </li>
                </c:forEach>
            </ul>
            
        </article>
            
    </section>
    
</section>

<%@ include file="../../layout/footer.jspf" %>
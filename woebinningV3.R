library(smbinning)
library(InformationValue)
library(plotrix)
#连续变量分析函数
#method=c("depth","optimal","width")
get_segment=function(data,x,y,method,segnum){
  data=as.data.frame(data)
  if(!x%in%colnames(data)){
    print("变量名输入错误，请检查！")
    return(NULL)
  }else{
    data[,x]=as.numeric(data[,x])
  }
  mi=min(data[,x],na.rm=T)
  ma=max(data[,x],na.rm=T)
  if(method=="depth"){
    if(segnum<=1|segnum>=11){
      print("分段数量不在1~10之间")
      return(NULL)
    }else{
      segment=seq(0,1,1/segnum)
      RET=as.numeric(quantile(data[,x],probs=segment,na.rm=T))
      RET=RET[-c(1,length(RET))]
    }
  }else if(method=="optimal"){
    sm=smbinning(df=data,y=y,x=x,p=0.05)
    if(is.character(sm)){
      print(paste0("变量",x," 没有最优分割点"))
      return(NULL)
    }else{
      RET=sm$cuts
    }
  }else{
    if(segnum<=1|segnum>=11){
      print("分段数量不在1~10之间")
      return(NULL)
    }else{
      RET=seq(mi,ma,length.out=segnum+1)
      RET=RET[-c(1,length(RET))]
    }
  }
  return(RET)
}
get_woetable=function(data,x,y,segment){
  woetable=NULL
  colnum=ncol(data)
  colindex=which(colnames(data)==x)
  if(colindex<0|colindex>colnum){
    return(woetable)
  }
  allgoodnum=nrow(data[which(data[,y]==0),])
  if(allgoodnum==0){
    return(woetable)
  }
  allbadnum=nrow(data[which(data[,y]==1),])
  if(allbadnum==0){
    return(woetable)
  }
  allnum=allgoodnum+allbadnum
  allgoodrate=allgoodnum/allnum
  allbadrate=allbadnum/allnum
  segnum=length(segment)
  for(index in 1:(segnum+2)){
    if(index==1){
      CntGood=nrow(data[which(data[,y]==0 & data[,colindex]<=segment[index]),])
      CntBad=nrow(data[which(data[,y]==1 & data[,colindex]<=segment[index]),])
      CntCumGood=nrow(data[which(data[,y]==0 & data[,colindex]<=segment[index]),])
      CntCumBad=nrow(data[which(data[,y]==1 & data[,colindex]<=segment[index]),])
    }else if(index<segnum+1){
      CntGood=nrow(data[which(data[,y]==0 & data[,colindex]>segment[index-1] & data[,colindex]<=segment[index]),])
      CntBad=nrow(data[which(data[,y]==1 & data[,colindex]>segment[index-1] & data[,colindex]<=segment[index]),])
      CntCumGood=nrow(data[which(data[,y]==0 & data[colindex]<=segment[index]),])
      CntCumBad=nrow(data[which(data[,y]==1 & data[colindex]<=segment[index]),])
    }else if(index==segnum+1){
      CntGood=nrow(data[which(data[,y]==0 & data[,colindex]>segment[index-1]),])
      CntBad=nrow(data[which(data[,y]==1 & data[,colindex]>segment[index-1]),])
      CntCumGood=nrow(data[which(data[,y]==0 & !is.na(data[colindex])),])
      CntCumBad=nrow(data[which(data[,y]==1 & !is.na(data[colindex])),])
    }else{
      CntGood=nrow(data[which(data[,y]==0 & is.na(data[,colindex])),])
      CntBad=nrow(data[which(data[,y]==1 & is.na(data[,colindex])),])
      CntCumGood=allgoodnum
      CntCumBad=allbadnum
    }
    CntGood=ifelse(CntGood>0,CntGood,1) #若某一区间好样本数为0，则设为1
    CntBad=ifelse(CntBad>0,CntBad,1) #若某一区间坏样本数为0，则设为1
    CntCumGood=ifelse(CntCumGood>0,CntCumGood,1) #若某一区间累计好样本数为0，则设为1
    CntCumBad=ifelse(CntCumBad>0,CntCumBad,1) #若某一区间累计坏样本数为0，则设为1
    CntRec=CntGood+CntBad
    CntCumRec=CntCumGood+CntCumBad
    GoodDist=round(CntGood/allgoodnum,4)
    BadDist=round(CntBad/allbadnum,4)
    GoodRate=round(CntGood/CntRec,4)
    BadRate=round(CntBad/CntRec,4)
    Odds=round(BadRate/GoodRate,4)
    LnOdds=round(log(Odds),4)
    WOE=round(log(BadDist/GoodDist),4)
    IV=round((BadDist-GoodDist)*log(BadDist/GoodDist),4)
    if(index==1){
      woetable=rbind(woetable,c(index,-Inf,segment[index],CntRec,CntGood,CntBad,CntCumRec,CntCumGood,CntCumBad,GoodDist,BadDist,GoodRate,BadRate,Odds,LnOdds,WOE,IV))
    }else if(index<segnum+1){
      woetable=rbind(woetable,c(index,segment[index-1],segment[index],CntRec,CntGood,CntBad,CntCumRec,CntCumGood,CntCumBad,GoodDist,BadDist,GoodRate,BadRate,Odds,LnOdds,WOE,IV))
    }else if(index==segnum+1){
      woetable=rbind(woetable,c(index,segment[index-1],Inf,CntRec,CntGood,CntBad,CntCumRec,CntCumGood,CntCumBad,GoodDist,BadDist,GoodRate,BadRate,Odds,LnOdds,WOE,IV))
    }else{
      woetable=rbind(woetable,c(index,NA,NA,CntRec,CntGood,CntBad,CntCumRec,CntCumGood,CntCumBad,GoodDist,BadDist,GoodRate,BadRate,Odds,LnOdds,WOE,IV))
    }
  }
  woetable=data.frame(woetable)
  colnames(woetable)=c("index","LB","UB","CntRec","CntGood","CntBad","CntCumRec","CntCumGood","CntCumBad","GoodDist","BadDist","GoodRate","BadRate","Odds","LnOdds","WOE","IV")
  return(woetable)
}
woe_for_continue=function(var,woetable){
  n_row=nrow(woetable)
  cutpoint=c(woetable$LB[1:n_row-1],Inf)
  label=cut(var,cutpoint,labels=1:(n_row-1))
  label=as.numeric(as.vector(label))
  label[is.na(label)]=woetable[n_row,"index"]
  var_woe=woetable[label,"WOE"]
  return(var_woe)
}
plot_continue=function(woetable,option,var_name=NULL){
  names_arg=c()
  for(i in 1:nrow(woetable)){
    if(i<=nrow(woetable)-2){
      names_arg[i]=paste0("<=",woetable[i,"UB"])
    }else if(i==nrow(woetable)-1){
      names_arg[i]=paste0(">",woetable[i-1,"UB"])
    }else{
      names_arg[i]="Missing"
    }
  }
  if(option == "dist"){
    dist=woetable$CntRec/sum(woetable$CntRec)
    y_upper = max(dist) * 1.25
    ch_dist = barplot(dist, names.arg=names_arg,axes = F,
                      main = "Percentage of Cases", ylim = c(0, y_upper), col = gray.colors(length(dist)))
    text(x = ch_dist, y = dist, label = round(dist * 100, 2), pos = 3, cex = 1)
    abline(h = 0)
    mtext(var_name, 3)
  }
  else if (option == "goodrate") {
    y_upper = max(woetable[, "GoodRate"], na.rm = T)*1.25
    ch_goodrate = barplot(woetable[, "GoodRate"],names.arg=names_arg,
                          axes = F, main = "Good Rate (%)", ylim = c(0,y_upper), col = gray.colors(length(woetable[, "GoodRate"])))
    text(x = ch_goodrate, y = woetable[,"GoodRate"], label=round(woetable[,"GoodRate"] * 100,2), pos = 3, cex = 1)
    abline(h = 0)
    mtext(var_name, 3)
  }
  else if (option == "badrate") {
    y_upper = max(woetable[,"BadRate"], na.rm = T)*1.25
    ch_badrate = barplot(woetable[, "BadRate"],names.arg=names_arg,
                         axes = F, main = "Bad Rate (%)", ylim = c(0, y_upper),col = gray.colors(length(woetable[,"GoodRate"])))
    text(x = ch_badrate, y = woetable[, "BadRate"], label = round(woetable[,"BadRate"] * 100,2), pos = 3, cex = 1)
    abline(h = 0)
    mtext(var_name, 3)
  }
  else if (option == "woe"){
    y_upper = max(woetable[, "WoE"], na.rm = T)*1.25
    y_lower = min(woetable[, "WoE"], na.rm = T)*1.25
    ch_woe = barplot(woetable[,"WoE"],names.arg=names_arg,
                     axes = F, main = "Weight of Evidence", ylim = c(y_lower,y_upper), col = gray.colors(length(woetable[,"WoE"])))
    text(x = ch_woe, y = woetable[,"WoE"], label = round(woetable[,"WoE"],2),pos = 3, cex = 1)
    abline(h = 0)
    mtext(var_name, 3)
  }else if(option=="both"){
    dist=woetable$CntRec/sum(woetable$CntRec)
    twoord.plot(1:nrow(woetable),dist,1:nrow(woetable),woetable[,"BadRate"],type=c("bar","b"),xlim=c(0,nrow(woetable)+1),
                xlab="bins",ylab="样本占比",rylab="违约率",lcol=4,rcol=2,do.first="plot_bg(\'gray\');grid(col=\'white\')",
                xticklab = names_arg,main=paste("样本占比与违约率\n",var_name),halfwidth=0.2)
  }else {
    return("Options are dist, goodrate, badrate, or woe")
  }
}
#离散变量分析函数
# woe_for_dispersed=function(X, Y, valueOfBad = 1){
#   yClasses <- unique(Y)
#   if (length(yClasses) == 2) {
#     Y[which(Y == valueOfBad)] <- 1
#     Y[which(!(Y == "1"))] <- 0
#     Y <- as.numeric(Y)
#     df <- data.frame(X, Y)
#     levels(X)=c(levels(X)) #若考虑Missing,则直接改为c(levels(X),"Missing")
#     woeTable <- as.data.frame(matrix(numeric((nlevels(X))* 8), nrow =nlevels(X), ncol = 8))
#     names(woeTable) <- c("CAT", "GOODS", "BADS", "TOTAL","PCT_G", "PCT_B","WOE", "IV")
#     woeTable$CAT <- c(levels(X))
#     for(catg in c(levels(X))){
#       a=sum(df$Y==1 & df$X %in% catg)
#       b=sum(df$Y==0 & df$X %in% catg)
#       woeTable[woeTable$CAT %in% catg,"BADS"]=ifelse(a==0,1,a)
#       woeTable[woeTable$CAT %in% catg,"GOODS"]=ifelse(b==0,1,b)
#     }
#     woeTable$TOTAL <- woeTable$GOODS+woeTable$BADS
#     woeTable$PCT_G <- woeTable$GOODS/sum(woeTable$GOODS,na.rm = T)
#     woeTable$PCT_B <- woeTable$BADS/sum(woeTable$BADS, na.rm = T)
#     woeTable$WOE <- log(woeTable$PCT_B/woeTable$PCT_G)
#     woeTable$IV <- (woeTable$PCT_B - woeTable$PCT_G) * woeTable$WOE
#     var_woe=woeTable[match(X,woeTable[,1]),"WOE"]
#     out=list()
#     out$var_woe=var_woe
#     out$woetable=woeTable
#     return(out)
#   }
#   else {
#     stop("WOE can't be computed because the Y is not binary.")
#   }
# } #返回woe表和离散变量woe值转换结果
woe_for_dispersed=function (X, Y, valueOfGood = 1){
  yClasses <- unique(Y)
  if (length(yClasses) == 2) {
    Y[which(Y == valueOfGood)] <- 1
    Y[which(!(Y == "1"))] <- 0
    Y <- as.numeric(Y)
    df <- data.frame(X, Y)
    woeTable <- as.data.frame(matrix(numeric((nlevels(X)+1)* 8), nrow = nlevels(X)+1, ncol = 8))
    names(woeTable) <- c("CAT", "GOODS", "BADS", "TOTAL","PCT_G", "PCT_B", "WOE", "IV")
    woeTable$CAT <- c(levels(X),NA)
    for (catg in c(levels(X),NA)) {
      a=sum(df$Y==1 & df$X %in% catg,na.rm=T)
      b=sum(df$Y==0 & df$X %in% catg,na.rm=T)
      woeTable[woeTable$CAT %in% catg,"BADS"]=ifelse(a==0,1,a)
      woeTable[woeTable$CAT %in% catg,"GOODS"]=ifelse(b==0,1,b)
    }
    woeTable$TOTAL <- woeTable$GOODS+woeTable$BADS
    woeTable$PCT_G <- woeTable$GOODS/sum(woeTable$GOODS,na.rm = T)
    woeTable$PCT_B <- woeTable$BADS/sum(woeTable$BADS, na.rm = T)
    woeTable$WOE <- log(woeTable$PCT_G/woeTable$PCT_B)
    woeTable$IV <- (woeTable$PCT_G - woeTable$PCT_B) * woeTable$WOE
    var_woe=woeTable[match(X,woeTable[,1]),"WOE"]
    out=list()
    out$var_woe=var_woe
    out$woetable=woeTable
    return(out)
  }
  else {
    stop("WOE can't be computed because the Y is not binary.")
  }
}
IV=function(X,Y,valueOfBad=1){
  res=woe_for_dispersed(X=X,Y=Y,valueOfBad = valueOfBad)
  iv=sum(res$woetable$IV,na.rm=T)
  if (iv < 0.03) {
    attr(iv, "howgood") <- "Not Predictive"
  }
  else if (iv < 0.1) {
    attr(iv, "howgood") <- "Somewhat Predictive"
  }
  else {
    attr(iv, "howgood") <- "Highly Predictive"
  }
  return(iv)
} #返回离散变量IV值
plot_dispersed=function(woetable,option,var_name=NULL){
  if(option=="WOE"){
    x_upper = nrow(woetable)
    y_upper = max(woetable[1:x_upper, "WOE"]) * 1.25
    y_lower = min(woetable[1:x_upper, "WOE"]) * 1.25
    woe_plot=barplot(woetable$WOE,names.arg = woetable$CAT[1:x_upper],axes=F,
                     ylim = c(y_lower,y_upper),main="Weight of Evidence",col=gray.colors(length(woetable$CAT)))
    text(x=woe_plot,y=woetable$WOE,label=round(woetable$WOE,2),pos=3,cex=1)
    abline(h=0)
    mtext(var_name,3)
  }else if(option=="dist"){
    dist=woetable$TOTAL/sum(woetable$TOTAL)
    x_upper = nrow(woetable)
    y_upper = max(dist) * 1.25
    y_lower = min(dist) * 1.25
    dist_plot=barplot(dist,names.arg = woetable$CAT[1:x_upper],axes=F,
                      ylim = c(0,y_upper),main="Percentage of
                      Cases",col=gray.colors(length(woetable$CAT)))
    text(x=dist_plot,y=dist,label=round(dist*100,2),pos=3,cex=1)
    abline(h=0)
    mtext(var_name,3)
  }else if(option=="badrate"){
    badrate=woetable$BADS/woetable$TOTAL
    x_upper = nrow(woetable)
    y_upper = max(badrate) * 1.25
    y_lower = min(badrate) * 1.25
    badrate_plot=barplot(badrate,names.arg = woetable$CAT[1:x_upper],axes=F,
                         ylim = c(0,y_upper),main="Bad
                         Rate(%)",col=gray.colors(length(woetable$CAT)))
    text(x=badrate_plot,y=badrate,label=round(badrate*100,2),pos=3,cex=1)
    abline(h=0)
    mtext(var_name,3)
  }else if(option=="goodrate"){
    goodrate=woetable$GOODS/woetable$TOTAL
    x_upper = nrow(woetable)
    y_upper = max(goodrate) * 1.25
    y_lower = min(goodrate) * 1.25
    goodrate_plot=barplot(goodrate,names.arg = woetable$CAT[1:x_upper],axes=F, ylim = c(0,y_upper),main="GoodRate(%)",col=gray.colors(length(woetable$CAT)))
    text(x=goodrate_plot,y=goodrate,label=round(goodrate*100,2),pos=3,cex=1)
    abline(h=0)
    mtext(var_name,3)
  }else if(option=="both"){
    dist=woetable$TOTAL/sum(woetable$TOTAL)
    badrate=woetable$BADS/woetable$TOTAL
    twoord.plot(1:nrow(woetable),dist,1:nrow(woetable),badrate,type=c("bar","b"),xlim=c(0,nrow(woetable)+1),
                xlab="bins",ylab="样本占比",rylab="违约率",lcol=4,rcol=2,do.first="plot_bg(\'gray\');grid(col=\'white\')",
                xticklab = woetable$CAT,main=paste("样本占比与违约率\n",var_name),halfwidth=0.2)
  }else{
    return("Options are dist, badrate,goodrate or WoE")
  }
}

############################demo################################
segment=get_segment(mydata,"cons_m12_RYBH_visits","y","optimal")
woetable=get_woetable(mydata,"cons_m12_RYBH_visits","y",segment)
var_woe=woe_for_continue(mydata$cons_m12_RYBH_visits,woetable)
plot_continue(woetable,"dist","cons_m12_RYBH_visits")
plot_continue(woetable,"badrate","cons_m12_RYBH_visits")
plot_continue(woetable,"woe","cons_m12_RYBH_visits")
plot_continue(woetable,"both","cons_m12_RYBH_visits")

woetable=woe_for_dispersed(as.factor(mydata$sl_id_bank_lost),mydata$y)
plot_dispersed(woetable$woetable,"badrate","sl_id_bank_lost")
plot_dispersed(woetable$woetable,"both","sl_id_bank_lost")




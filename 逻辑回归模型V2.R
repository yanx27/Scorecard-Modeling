library(readxl)
library(smbinning)
setwd("C:\\Users\\admin\\Desktop\\.....")
#####################数据读取########################################
dat=read_excel(".....xlsx")
dat=as.data.frame(dat)
######################变量选择：基于IV################################
continue_var_index=sapply(1:ncol(dat),function(i) is.numeric(dat[,i]))
var_continue=colnames(dat)[continue_var_index]
var_dispersed=colnames(dat)[!continue_var_index]
for(var in var_continue){
  dat[,var]=as.numeric(dat[,var])
}
for(var in var_dispersed){
  dat[,var]=as.character(dat[,var])
}
#连续变量选择
for(var in var_continue){
  try(sm<-smbinning(dat,"y",var),silent=T)
  if(!is.character(sm)){
    if(sm$iv>0.02){
      print(paste0(var,":",sm$iv))
    }
  }else{
    var_continue=var_continue[var_continue!=var]
  }
}
#离散变量选择
for(var in var_dispersed){
  iv=IV(as.factor(dat[,var]),dat$y)
  if(iv>0.02){
    print(paste0(var,":",as.numeric(iv)))
  }else{
    var_dispersed=var_dispersed[var_dispersed!=var]
  }
}
var_imp=union(var_continue,var_dispersed)
mydata=dat[c(var_imp,"y")]
############################WOE转换###################################
woetable_continue=list()  #记录每个入模连续变量的woetable
woetable_dispersed=list()  #记录每个入模离散变量的woetable
#连续变量woe转换
for(var in var_continue){
  var_con_woe=paste0(var,"_con_woe")
  woetable_name=paste0("woetable_",var)
  segment=get_segment(mydata,var,"y","optimal")
  woetable=get_woetable(mydata,var,"y",segment)
  woetable_continue[[woetable_name]]=woetable
  mydata[,var_con_woe]=woe_for_continue(mydata[,var],woetable)
}
#离散变量woe转换
for(var in var_dispersed){
  var_disp_woe=paste0(var,"_disp_woe")
  woetable_name=paste0("woetable_",var)
  woetable_dispersed[[woetable_name]]=woe_for_dispersed(as.factor(mydata[,var]),mydata$y)$woetable
  mydata[,var_disp_woe]=woe_for_dispersed(as.factor(mydata[,var]),mydata$y)$var_woe
}
mydata=mydata[,-c(1:length(var_imp))]
#逻辑回归
set.seed(20171017)
index=sample(1:nrow(mydata),1000)
train=mydata[index,]
test=mydata[-index,]
fit=glm(y~.,family=binomial(link="logit"),data=mydata)
fit_step=step(fit)
fit_new=glm(formula=fit_step$call$formula,family=binomial(link="logit"),data=train)
y_pred=predict(fit_new,test,type="response")
K_S(y_pred,test$y)







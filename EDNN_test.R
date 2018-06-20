#version 2
#date of update: 6/Feb/2018

#trainData: prepare the training data for regression.
#respVar | var1| var2| var3|...
#  10    | 1.1 | 1.1 | 5.4 |...
#  12    | 1.2 | 1.7 | 2.4 |...
#Note: First column is a response variable, the other columns are explanatory variables.

#testData: prepare the test data (samples that response variables are unknown).
# var1| var2| var3|...
# 0.1 | 1.3 | 2.4 |...
# 1.0 | 1.5 | 5.4 |...

#trainData and testData
library(MASS)
data("Boston", package = "MASS")
data <- cbind(Boston[,ncol(Boston)], Boston[,1:ncol(Boston)-1])
colnames(data)[1] = colnames(Boston)[ncol(Boston)]
trainData <- data[1:400,]
testData <- data[401:nrow(data), 2:ncol(data)]

#EDNN parameters
{
  nc<-300
  rrv<-0.2
  rwc<-0.2
}

#Remaining time setting
{
  ndata<-(ncol(trainData))
  Rmn1<-0
  ns<-ndata-1
  nss<-nc
}

#Initial data processing.
{
  library(mxnet)
  
  predata<-trainData
  predata<-cbind(predata[,1],scale(predata[2:ncol(predata)],center=FALSE,scale=FALSE))
  testData2<-scale((testData),center=FALSE,scale=FALSE)
  Ktest<-testData2
  K3<-as.numeric(NULL)
  K3.5<-as.numeric(NULL)
  K4<-as.numeric(NULL)
  ImpEdnn<-as.numeric(NULL)    
  pv.eval.raw<-as.numeric(NULL)
  rmse.perm.raw.all <- as.numeric(NULL)
  mod.rsme.lk <- as.numeric(NULL)
  val3 <- as.numeric(NULL)
  Kpv.eval.raw<-as.numeric(NULL)
  pnum<-5
}      

#EDNN classifier
for (h in 1:nc){
  #Bootstrap resampling and Random sampling of variables
  pre.num<-sample(nrow(predata),nrow(predata),replace = T)  
  prepre<-predata[,2:ncol(predata)]
  pre.val<-sample(ncol(prepre),ncol(prepre)*(1-rrv))
  pre.val<-pre.val+1                
  
  train<-predata[pre.num,-pre.val]
  test<-predata[-pre.num,-pre.val]
  Ktest1<-cbind(1:nrow(testData),Ktest)
  Ktest2<-Ktest1[,-pre.val]
  Ktest3<-rbind(Ktest2,Ktest2)
  
  #Parameter settings of DNN classifier
  {
    train.x<- train[,-1]
    train.y<- train[,1]
    
    data <- mx.symbol.Variable("data")
    fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=512)
    act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
    fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
    act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
    fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=1)
    output <- mx.symbol.LinearRegressionOutput(fc3,name="linreg")
    devices <- mx.cpu()
    
    
    model <- mx.model.FeedForward.create(
      output, X=train.x,
      y=train.y,
      ctx=devices, 
      num.round= 100, 
      array.batch.size=100,
      learning.rate=0.0012, 
      momentum=0.6,  
      eval.metric=mx.metric.rmse,
      initializer=mx.init.uniform(0.09),
      array.layout = "rowmajor",
      epoch.end.callback=mx.callback.log.train.metric(40))
  }
  
  #Evaluation of EDNN classifier
  {
    eval.x<- test[,-1]
    pred.eval <- predict(model, eval.x,array.layout='rowmajor')
    eval.y<-as.matrix(test[,1])
    
    pv.eval<-cbind(eval.y, t(pred.eval))
    pv.eval<-cbind(pv.eval,h)
    pv.eval.raw <- rbind(pv.eval.raw, pv.eval)
    
    mod.rsme <- sqrt(mean((pv.eval[,1]-pv.eval[,2])^2))
    mod.rsme2<-c(h,mod.rsme)
    mod.rsme.lk<-rbind(mod.rsme.lk,mod.rsme2)
  }    
  
  #Calculation of importance by Permutation
  {   
    rmse.perm.raw <- as.numeric(NULL)
    
    for (i in 1:ncol(eval.x)){
      test.perm <- eval.x
      mod.rsme.perm <- h
      
      
      for (j in 1:pnum){
        tesp <- sample(eval.x[,i])
        test.perm[,i] <- tesp
        pred.perm <- predict(model, test.perm,array.layout='rowmajor')
        pv.perm <- rbind(t(eval.y), pred.perm)
        perm.res <- sqrt(mean((pv.perm[1,]-pv.perm[2,])^2))
        mod.rsme.perm <- c(mod.rsme.perm, (mod.rsme - perm.res))
      }
      
      rmse.perm.raw <- rbind(rmse.perm.raw, mod.rsme.perm)
    }
    
    rowlabel <- colnames(eval.x)
    rownames(rmse.perm.raw) <- rowlabel
    rmse.perm.raw.all <-rbind(rmse.perm.raw.all,rmse.perm.raw)
  }      
  
  #Prediction of test data
  {
    Keval.x<- Ktest3[,-1]
    Kpred.eval <- predict(model, Keval.x,array.layout='rowmajor')
    Keval.y<-as.matrix(Ktest3[,1])
    
    Kpv.eval<-cbind(Keval.y, t(Kpred.eval))
    Kpv.eval<-cbind(Kpv.eval,h)
    Kpv.eval.raw <- rbind(Kpv.eval.raw, Kpv.eval)
    K3 <- rbind(K3, Kpv.eval.raw)
    K3.5<- rbind(K3.5, Kpv.eval.raw)
  }       
  
  #Remaining time
  {
    Rmn1<-Rmn1+1
    Rmn2<-Rmn1/nss
    Rmn3<-c(Rmn2,1-Rmn2)
    names(Rmn3)<-c("Dane","Remain")
    pie(Rmn3) 
  }
  
  
}

#Output of prediction data (RMSEs) and its importance
{
  lk<-mod.rsme.lk[order(mod.rsme.lk[,2])]
  hlk<-lk[1:round(nc*rwc)]
  hh<- length(hlk)
  
  K3<-as.numeric(NULL)
  impav<-rowMeans(rmse.perm.raw.all[,2:(pnum+1)])
  impavv<-cbind(rmse.perm.raw.all[,1],impav)
  himp2 <- as.numeric(NULL)
  
  K3<-as.numeric(NULL)
  for(hhh in 1:hh){
    Khrsme<-subset(Kpv.eval.raw,Kpv.eval.raw[,3]==hlk[hhh])
    K3<-rbind(K3,Khrsme)
  }
  K4<-rbind(K4,K3)
  K5<-tapply(K4[,2],K4[,1],mean)
  
  for(hhh in 1:hh){
    himp<-subset(impavv,impavv[,1]==hlk[hhh])
    himp2<-rbind(himp2,himp)
  }
  himp3<-cbind(rownames(himp2),as.numeric(himp2[,2]))  
  meanB2<-tapply(himp2[,2],himp3[,1],mean)
  sdB2<-tapply(himp2[,2],himp3[,1] ,sd)
  meansdB2<-cbind(meanB2,sdB2)
  ImpEdnn<-rbind(ImpEdnn,meansdB2)
  ImpEdnn2<-ImpEdnn*(-1)
  
  write.table(K5,"EDNN_predict.csv",sep=",")
  write.table(ImpEdnn2,"EDNN_Importance.csv",sep=",")
}
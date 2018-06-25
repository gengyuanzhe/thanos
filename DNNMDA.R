#version information: v03w3
#date of update: 25/Dec/2017


DNNMDA <- function(alldata,clasnum,k,hid1,hid2,numr,arbas,lrate,pnum){
  library(mxnet)
  
  #data preparation for k-fold cross validation
  residu <- k - nrow(alldata)%%k
  dumdat <-as.data.frame(matrix(NA,nrow=residu,ncol=ncol(alldata)))
  names(dumdat)<-names(alldata)
  alldum <-rbind(alldata, dumdat)
  alldum.split <-split(alldum,1:k)
  all.split <- lapply(alldum.split, na.omit)
  
  pred.eval.raw <- as.numeric(NULL)
  pv.eval.raw <- as.numeric(NULL)
  acc.perm.raw.all <- as.numeric(NULL)
  acc.perm.raw.all2 <- as.numeric(NULL)
  
  #k-fold cross validation
  for (h in 1:k){
    evaldata <- data.matrix(all.split[[h]])
    allspr <- all.split[-h]
    object <- data.matrix(do.call("rbind", allspr))
    
    #model construction using training data
    train.x<- object[,-1]
    train.y<- object[,1]
    
    data <- mx.symbol.Variable("data")
    fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=hid1)
    act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
    fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=hid2)
    act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
    fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=clasnum)
    softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")
    devices <- mx.cpu()
    model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,ctx=devices, num.round= numr, array.batch.size=arbas,learning.rate=lrate, momentum=0.9,  eval.metric=mx.metric.accuracy,initializer=mx.init.uniform(0.07),epoch.end.callback=mx.callback.log.train.metric(100))
    
    #calculation of classification accuracy using evaluation data
    eval.x<- evaldata[,-1]
    pred.eval <- predict(model, eval.x)
    pred.eval.label <- max.col(t(pred.eval))-1
    eval.y<-as.matrix(evaldata[,1])
    pv.eval<-rbind(t(eval.y), pred.eval.label)
    pred.eval.raw <- cbind(pred.eval.raw, pred.eval)
    pv.eval.raw <- cbind(pv.eval.raw, pv.eval)
    
    sum.pv <- table(pv.eval[1,], pv.eval[2,])
    mod.acc <- as.numeric(sum(diag(sum.pv))/sum(sum.pv))
    
    #calculation of importance by permutation
    acc.perm.raw <- as.numeric(NULL)
    acc.perm.raw2 <- as.numeric(NULL)
    for (i in 1:ncol(eval.x)){
      
      test.perm <- eval.x
      mod.acc.perm <- as.numeric(NULL)
      mod.acc.perm2 <- as.numeric(NULL)
      for (j in 1:pnum){
        tesp <- sample(eval.x[,i],nrow(eval.x))
        test.perm[,i] <- tesp
        pred.perm <- predict(model, test.perm)
        pred.label.perm <- max.col(t(pred.perm))-1
        pv.perm <- rbind(t(eval.y), pred.label.perm)
        sum.pv.perm <- table(pv.perm[1,], pv.perm[2,])
        perm.res <- as.numeric(sum(diag(sum.pv.perm))/sum(sum.pv.perm))
        mod.acc.perm <- c(mod.acc.perm, (mod.acc - perm.res))
        mod.acc.perm2 <- c(mod.acc.perm2, (mod.acc - perm.res)^2)
      }
      
      acc.perm.raw <- rbind(acc.perm.raw, mod.acc.perm)
      acc.perm.raw2 <- rbind(acc.perm.raw2, mod.acc.perm2)
    }
    
    rowlabel <- colnames(eval.x)
    rownames(acc.perm.raw) <- rowlabel
    rownames(acc.perm.raw2) <- rowlabel
    acc.perm.raw.all <-cbind(acc.perm.raw.all,acc.perm.raw)
    acc.perm.raw.all2 <-cbind(acc.perm.raw.all2,acc.perm.raw2)
  }
  
  #output of classification accuracy
  sum.pv.eval <- addmargins(table(pv.eval.raw[1,], pv.eval.raw[2,]))
  write.table(pred.eval.raw,"accuracy_output.csv",sep=",")
  write.table(pv.eval.raw,"accuracy_raw.csv",sep=",")
  write.table(sum.pv.eval,"confusion_matrix.csv",sep=",")
  
  #output of importance
  mean.dec.acc.all <- rowMeans(acc.perm.raw.all)
  write.table(acc.perm.raw.all,"importance_raw.csv",sep=",")
  write.table(mean.dec.acc.all,"importance.csv",sep=",")
  mean.dec.acc.all2 <- rowMeans(acc.perm.raw.all2)
  write.table(acc.perm.raw.all2,"importanceSQ_raw.csv",sep=",")
  write.table(mean.dec.acc.all2,"importanceSQ.csv",sep=",")
  
}
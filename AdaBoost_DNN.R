#AdaBoost for DNN to imporve the performance of classfication ability
#version 1.0
#Author: gengyuanzhe@gmail.com
#date: 25/June/2018

#refer to "周志华. 机器学习 : = Machine learning[M]. 清华大学出版社, 2016."
#--intput: 
#    Data: train set
#    E: leaning algorithm
#    nc: times of training or number of weak classifiers
#--procedure:
#    1. D[i] = 1/m,   i=1:m, m is count of training samples     # distribution of training sample weight
#    2. for t=1,....nc do 
#    3.   h[t] = E(Data, D[t])                                  # classifier from Data and D
#    4.   e[t] = P(h[t](x) != f(x)), x~D[t]                     # error rate for Distribution D
#    5.   if e[t] > 0.5 then break
#    6.   a[t] = 0.5*ln((1-e[t])/e[t])                          # weight of classifier
#    7.   D[t+1] = D[t]/e[t],            when h[t](x) != f(x)   # update distribution for next training
#    8.   D[t+1] = D[t]/(1-e[t]),        when h[t](x) == f(x)
#    9. end for
#--output:
#    final classifier: H(x) = sign(sum(a[t]h[t](x)))   

#usage
#input: trainData, nc
#--trainData: prepare the training data for regression.
#  respVar | var1| var2| var3|...
#    10    | 1.1 | 1.1 | 5.4 |...
#    12    | 1.2 | 1.7 | 2.4 |...
#  Note: First column is a response variable, the other columns are explanatory variables.
#--nc: times of training or number of weak classifiers


#input
{
  wbcd <- read.csv("wisc_bc_data.csv", stringsAsFactors = FALSE)
  wbcd <- wbcd[,-1]
  wbcd[,1] <- as.numeric(wbcd[,1]=='B')
  wbcd <- as.matrix(wbcd)
  wbcd[,2:ncol(wbcd)] = scale(wbcd[,2:ncol(wbcd)], center = TRUE, scale = TRUE)
  
  # parameters of adaboost dnn
  trainData = wbcd[1:400, ]
  nc = 1
}

#Initial data processing.
{
  library(mxnet)
  m = nrow(trainData)  #sample number
  D = rep(1/m, m)      #distribution of sample
}

#AdaBoost DNN classifier
for (i in 1:nc){
  #Bootstrap resampling
  {
    sample.index <- sample(m, m, replace = T, prob = D)
    train.x <- trainData[sample.index, -1]
    train.y <- trainData[sample.index, 1]
  }
  
  #dnn 
  {
    data <- mx.symbol.Variable("data")
    fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=512)
    act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
    fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
    act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
    fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
    net <- mx.symbol.SoftmaxOutput(fc3, name="sm")
    
    #cpu or gpu
    device.cpu <- mx.cpu()
    n.gpu <- 1
    device.gpu <- lapply(0:(n.gpu-1), function(i) {
      mx.gpu(i)
    })
    
    
    mx.set.seed(0)
    tic <- proc.time()
    model <- mx.model.FeedForward.create(
      net, 
      X=train.x,
      y=train.y,
      ctx=device.cpu, 
      num.round= 100, 
      array.batch.size=100, 
      learning.rate=0.05,                #or 0.0012
      momentum=0.9,                      #or 0.6
      wd=0.00001,                        #or omit
      eval.metric=mx.metric.accuracy,    #eval.metric=mx.metric.rmse when regression,
      initializer=mx.init.uniform(0.09), #or omit 
      array.layout = "rowmajor",
      epoch.end.callback=mx.callback.log.train.metric(40))  #or 100
    print(proc.time() - tic)
  }
  
  #adaboost
  {
    
    
  }
}










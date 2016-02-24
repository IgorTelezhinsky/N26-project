library(caret)        #machine learning
library(kernlab)      #svm
library(randomForest) #rf
library(gbm)          #gbm
library(nnet)         #nnet
library(mgcv)         #gam
library(doMC)         #multi-core
registerDoMC(cores = 4)

setwd("~/Projects/JobTests/number26")
source('prepareData.R')

buildModel<-function(dataList, methods="rf", combineMethod="svmRadial",outcome="signupCompleted")
{
    #for reproducability
    set.seed(1234)
    L<-length(methods)
    if(L>=2) 
    {   
        #matrix for storing accuracy of the predictions
        acc <-vector(mode="numeric",length=L+1)
        #model names
        model_names <- c(methods,"Combined")
    }
    else
    {
        acc <-vector(mode="numeric",length=1)
        model_names <- c(methods)
    }
    
    #model fits and predictions lists
    mL<-length(model_names)
    model_fits  <- vector("list", mL)
    prediction  <- vector("list", mL)
    names(model_fits) <- model_names
    names(prediction) <- model_names
    
    #let's use 8-fold cv for all methods 
    fitControl <- trainControl(method = "cv", number = 8)
    
    i <- 1
    ycol <- which(colnames(dataList$cvTrain) == outcome)
    for(Method in methods)
    {
        if(Method == "nnet")
        {
            model_fits[[i]] <- train(as.formula(paste(outcome,"~ .")), 
                                     method = Method, data = dataList$cvTrain, trControl = fitControl,
                                     trace = FALSE, verbose = FALSE)
        }
        else
        {
            model_fits[[i]] <- train(as.formula(paste(outcome,"~ .")), 
                                     method = Method, data = dataList$cvTrain, trControl = fitControl,
                                     verbose = FALSE)
        }
        prediction[[i]] <- predict(model_fits[[i]], newdata = dataList$cvTest[,-ycol])
        acc[i]          <- confusionMatrix(dataList$cvTest[,ycol],prediction[[i]])$overall['Accuracy']
        i<-i+1
    }
    
    if(L>=2) 
    {
        #combined model from the above models
        combPredDF   <- data.frame(prediction[[1]]);colnames(combPredDF)[1]<-"p1"
        for(j in 2:L) {combPredDF<-cbind(combPredDF,prediction[[j]]);colnames(combPredDF)[j]<-paste0("p",j)}
        combPredDF   <- cbind(combPredDF,dataList$cvTest[,ycol]);colnames(combPredDF)[mL]<-outcome
        
        ycolpr <- which(colnames(combPredDF) == outcome)
        if(Method == "nnet")
        {
            model_fits[[mL]] <- train(as.formula(paste(outcome,"~ .")), 
                                      trace = FALSE,  method = combineMethod, data = combPredDF, 
                                      trControl = fitControl, verbose = FALSE)
        }
        else
        {
            model_fits[[mL]] <- train(as.formula(paste(outcome,"~ .")), 
                                      method = combineMethod, data = combPredDF, 
                                      trControl = fitControl, verbose = FALSE)
        }
        
        prediction[[mL]] <- predict(model_fits[[mL]],newdata = combPredDF[,-ycolpr])
        acc[mL]          <- confusionMatrix(dataList$cvTest[,ycol],prediction[[mL]])$overall['Accuracy']
    }
    modelResult <- list("model_fits" = model_fits, "prediction" = prediction, "acc" = acc) #, "prDF" = combPredDF)
}

#test data must not contain predicted value
predictOutcome<-function(builtModel,testData)
{
    mL <- length(builtModel$model_fits)
    if(mL == 1) 
    {
        prediction  <- vector("list", 1)
        names(prediction) <-names(builtModel$prediction)
        prediction<-predict(builtModel$model_fits[[1]],testData)
    }
    else
    {
        prediction  <- vector("list", mL)
        names(prediction) <-names(builtModel$prediction)
        for(i in 1:(mL-1)) {prediction[[i]] <- predict(builtModel$model_fits[[i]],testData)}
        
        combPredDF   <- data.frame(prediction[[1]]);colnames(combPredDF)[1]<-"p1"
        for(j in 2:(mL-1)) {combPredDF<-cbind(combPredDF,prediction[[j]]);colnames(combPredDF)[j]<-paste0("p",j)}
        prediction[[mL]] <- predict(builtModel$model_fits[[mL]],combPredDF)
    }
    prediction
}
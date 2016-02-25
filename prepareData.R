library(data.table) #dt
library(plyr)       #df manipulations
library(caret)      #machine learning

readDT<-function(datafile="TRAIN.csv")
{
    #specify columns classes (NULL cols will not be read)
    #cols<-c("NULL", rep("numeric",3),"character","integer",rep("character",3),"integer")
    cols<-c("NULL", rep("numeric",3),"NULL","integer","NULL","character","NULL","integer")
    t <- fread(datafile, sep = ",", stringsAsFactors = FALSE, na.strings = c("NULL",""),colClasses = cols)
}

convert<-function(dtin,operations,returnNatRank=FALSE)
{
    dt<-copy(dtin)
    if("toPOSIX" %in% operations)
    { #convert time columns to POSIXct class
        dt[,1:3 := lapply(.SD, function(x) as.POSIXct(x/1000,origin="1970-01-01")), .SDcols=1:3]
    }
    if("numNatId" %in% operations)
    { #setting nationality as id by number of frequency occurance
        rank<-arrange(dt[,.N,by=nationality],desc(N))      
        rank$nationalityId<-rank[,.I]
        dt<-merge(dt,rank,by="nationality"); dt[,c("nationality","N"):=NULL]
    }
    if("abcNatId" %in% operations)
    { #setting nationality as id alphabetically
        rank<-arrange(dt[,.N,by=nationality],nationality)
        rank$nationalityId<-rank[,.I]
        dt<-merge(dt,rank,by="nationality"); dt[,c("nationality","N"):=NULL]
    }
    if("logNatId" %in% operations)
    {
        dt[,nationalityId:=log(nationalityId)]
    }
    if("scaleTimes" %in% operations)
    {#scaling time columns
        dt[,c(1:3) := lapply(.SD, scale), .SDcols=c(1:3)]
    }
    if("scaleAll" %in% operations)
    {#scaling all variables
        colScale<-ncol(dt)
        dt[, 1:colScale:= lapply(.SD, scale), .SDcols=1:colScale]
    }
    
    if (returnNatRank)
        return(dataAndRanks<-list("data" = as.data.frame(dt),"rank" = rank))
    else return(as.data.frame(dt))
}

#make df with PCs instead of original data
pcdf<-function(df) as.data.frame(prcomp(df)$x)

bindClusterId<-function(df)
{
    kmCl <- kmeans(df, 5, nstart=25, iter.max=1000)
    df<-cbind(df,clusterId=kmCl$clust)
}

prepTrainData<-function(df,outcome="signupCompleted")
{
    set.seed(1234)
    #converting some variables to factors
    ycol <- which(colnames(df) == outcome)
    df[,ycol]<-as.factor(df[,ycol])
    
    gcol <- which(colnames(df) == "gender")
    df[,gcol]<-as.factor(df[,gcol])
    
    if(outcome != "clusterId")
    {
        ccol <- which(colnames(df)== "clusterId")
        if(length(ccol) != 0) df[,ccol]<-as.factor(df[,ccol])
    }
    
    #selecting cross validation (cv) and final model testing data sets
    cvDataInd <- createDataPartition(y=df[,ycol], p=0.8,list=FALSE)
    cvDataset <- df[cvDataInd,]
    fmTesting <- df[-cvDataInd,]
    
    #selecting cv training and testing data sets
    inTrainInd <- createDataPartition(y=cvDataset[,ycol],p=0.8,list=FALSE)
    cvTraining <- cvDataset[inTrainInd,]
    cvTesting  <- cvDataset[-inTrainInd,]
    dataSetsList<-list("cvTrain" = cvTraining, "cvTest" = cvTesting, "fmTest" = fmTesting)
}

#to test prediction algorithm your test data needs to be in tha same format
#as my train set, i.e., same columns and their names.
#the test data here is prepared to have exactly same transformations
#that were used for the learning
prepTestData<-function(fileTrain,fileTest,outcome="signupCompleted")
{
#first read training data set
    trainDT <- readDT(fileTrain)
#reading testing data set
    testDT  <- readDT(fileTrain)
#check that read in variables have same names     
    if(!identical(names(trainDT),names(testDT))) stop("Data Files are not identical! Aborting!")
    
#transform it according to my transformations; trainLS is a list containing data and
#rank data table that I will use to transform your test data set
    trainLS <- convert(trainDT,operations = c("numNatId","scaleTimes","logNatId"),returnNatRank=TRUE)
#transform nationality variable
    testDT  <- merge(testDT,trainLS$rank,by="nationality"); testDT[,c("nationality","N"):=NULL]
#transform times and make log transform to nationalityId
    testDF  <- convert(testDT,operations = c("scaleTimes","logNatId"))
    
#converting some variables to factors
    ycol <- which(colnames(testDF) == outcome)
    testDF[,ycol]<-as.factor(testDF[,ycol])
    
    gcol <- which(colnames(testDF) == "gender")
    testDF[,gcol]<-as.factor(testDF[,gcol])
    
    if(outcome != "clusterId")
    {
        ccol <- which(colnames(testDF)== "clusterId")
        if(length(ccol) != 0) testDF[,ccol]<-as.factor(testDF[,ccol])
    }
    testDF
}

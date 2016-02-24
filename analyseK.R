library(cluster)    #clara
library(vegan)      #calinski method
library(mclust)     #expectation-maximization


#understand how many PCs we need
analysePC<-function(df)
{
    p<-princomp(df)
    plot(p,type = 'l')
    summary(p)
}

#modified from 
#http://www.r-statistics.com/2013/08/k-means-clustering-from-r-in-action/
wssplot <- function(data, default=TRUE, nc=15, seed=1234)
{
    wss <- (nrow(data)-1)*sum(apply(data,2,var))
    for(i in 2:nc)
    {
        if(default)
        {
            set.seed(seed)
            wss[i] <- sum(kmeans(data, centers=i)$withinss)
        }
        else wss[i] <- sum(kmeans(data, centers=i,nstart=25, iter.max=1000)$withinss)
    }
    plot(1:nc, wss, type="b", xlab="Number of Clusters",
         ylab="Within groups sum of squares")
}

#what is the best k depending on number of components
#using wss plot
analyseKM<-function(df,cmin,cmax,default=F)
{
    for(i in cmin:cmax)
    {
        print(paste("Nr.of components used:",i))
        wssplot(df[,1:i],default)
    }
}

#for clara method should be similar to pam but much faster
analyseCL <- function(df, nc=15, seed=1234)
{
    sil<-rep(NA,nc)
    for(i in 2:nc)
    {
        sil[i]<-clara(df,i,samples = 50, rngR=T)$silinfo$avg.width #clus.avg.widths #avg.width
    }
    print(paste("Best Nr. of clusters:", which.max(sil)))
    plot(2:nc, sil[2:nc], type="b", xlab="Number of Clusters",
         ylab="Silhouette Average")
    abline(h=mean(sil,na.rm = T)+sd(sil,na.rm = T))
    abline(h=mean(sil,na.rm = T)-sd(sil,na.rm = T))
}



#following methods are adapted from a very comprehensive reply on stackoverflow:
#http://stackoverflow.com/questions/15376075/cluster-analysis-in-r-determine-the-optimal-number-of-clusters/15376462#15376462

#calinski method
analyseCM<-function(df)
{
    fit <- cascadeKM(scale(df, center = TRUE,  scale = TRUE), 1, 8, iter = 1000)
    plot(fit, sortg = TRUE, grpmts.plot = TRUE)
    calinski.best <- as.numeric(which.max(fit$results[2,]))
    cat("Calinski criterion optimal number of clusters:", calinski.best, "\n")
}

#expectation-maximization
analyseEM<-function(df)
{
    d_clust <- Mclust(as.matrix(df), G=1:8)
    m.best <- dim(d_clust$z)[2]
    cat("model-based optimal number of clusters:", m.best, "\n")
    plot(d_clust$BIC)
}

analyseK<-function(method="KM",df,...)
{
    if("KM" %in% method) analyseKM(df,...)  #kMean wss
    if("CL" %in% method) analyseCL(df)      #clara silhuoette
    if("CM" %in% method) analyseCM(df)      #calinski method
    if("EM" %in% method) analyseEM(df)      #expectation-maximization
}
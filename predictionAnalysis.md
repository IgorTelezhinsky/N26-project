# Number26 Project: Prediction Analysis.
Igor Telezhinsky  
2016-02-24  

# Context

Task 2 can be solved with the classification analysis - a subject of supervised learning. There are numerous algorithms available. I will stop here at several most frequently used: *random forest*, *bossted decision trees*, *neural network*, and *support vector machines* with Gaussian kernel. I will also probe *general additive model* to combine other models into ensemble of models for the final model.

# Analysis

I will use here *caret* package. I implemented simple ensemble of several models defined by *method* parameter in my function *buildModel()*. Each of the methods is fitted by *train()* function many times to bootstrapped data during tuning of each method specific parameters. In addition, I use 8-fold cross-validation for the train control. Otherwise standard *train()* parameters are used (see *predictionAnalysis.R* file.)

## Loading and transforming data


```r
dt <- readDT()
df <- convert(dt,operations = c("numNatId","scaleTimes","logNatId"))
```
## Model selection: predicted outcome *signupCompleted*

Preparing data of training, cross-validating, and final model testing data sets for the prediction of *signupCompleted* variable:

```r
dlSign <- prepTrainData(df,outcome="signupCompleted")
```

### The data set w/o clustering information

The accuracy of each model in the *methods* parameter and the accuracy of the final combined model is given by 4 numbers following each calculation.


```r
modelSign1<-buildModel(dlSign, methods=c("rf","gbm","nnet"),
                              combineMethod="rf", outcome="signupCompleted")
```

```
## note: only 2 unique complexity parameters in default grid. Truncating the grid to 2 .
```

```r
modelSign1$acc
```

```
## [1] 0.9388393 0.8727679 0.9406250 0.9450893
```


```r
modelSign2<-buildModel(dlSign, methods=c("rf","gbm","nnet"),
                              combineMethod="nnet", outcome="signupCompleted")
modelSign2$acc
```

```
## [1] 0.9388393 0.8727679 0.9406250 0.9450893
```


```r
modelSign3<-buildModel(dlSign, methods=c("rf","gbm","nnet"),
                              combineMethod="svmRadial", outcome="signupCompleted")
modelSign3$acc
```

```
## [1] 0.9388393 0.8727679 0.9406250 0.9450893
```


```r
modelSign4<-buildModel(dlSign, methods=c("rf","gbm","nnet"),
                              combineMethod="gam", outcome="signupCompleted")
modelSign4$acc
```

```
## [1] 0.9388393 0.8727679 0.9406250 0.9401786
```

It's seen that models 1,2,3 are equally good. I prefer to stop at model 2. The method "gbm" failed to combine models. Adding another method to the list affects the fourth digit after coma in accuracy of combined model.

### Adding clustering information

Let's  add cluster information and see if it helps 


```r
dfc<-bindClusterId(df)
```

Preparing data for training, cross-validating, and testing prediction of *signupCompleted* variable given cluster ingormation

```r
dlSignC <- prepTrainData(dfc,outcome="signupCompleted")
```


```r
modelSignC1<-buildModel(dlSignC, methods=c("rf","gbm","nnet"),
                              combineMethod="rf", outcome="signupCompleted")
```

```
## note: only 2 unique complexity parameters in default grid. Truncating the grid to 2 .
```

```r
modelSignC1$acc
```

```
## [1] 0.9401786 0.8629464 0.9428571 0.9482143
```


```r
modelSignC2<-buildModel(dlSignC, methods=c("rf","gbm","nnet"),
                              combineMethod="nnet", outcome="signupCompleted")
modelSignC2$acc
```

```
## [1] 0.9401786 0.8629464 0.9428571 0.9482143
```


```r
modelSignC3<-buildModel(dlSignC, methods=c("rf","gbm","nnet"),
                              combineMethod="svmRadial", outcome="signupCompleted")
modelSignC3$acc
```

```
## [1] 0.9401786 0.8629464 0.9428571 0.9482143
```


```r
modelSignC4<-buildModel(dlSignC, methods=c("rf","gbm","nnet"),
                              combineMethod="gam", outcome="signupCompleted")
modelSignC4$acc
```

```
## [1] 0.9401786 0.8629464 0.9428571 0.9424107
```

Well, the clustering information improves a bit the prediction. The improvement, however, is not large, so I will *not* use the clustering information. Besides, if the clustering information in your data set is absent, the only way to create it is to train/build a model based on my classified data set to predict the *clusterId* in your test data set. Given the non-zero error of any prediction, the clustering information in this case may be missleading.

# Verifying model. Out of sample (OOS) error.

To estimate out of sample error we will perform prediction on the reserved testing set for the final model. It was not used during model building and selection process.


```r
#note the testing set should NOT contain predicted variable
testing <- dlSign$fmTest
testing$signupCompleted <- NULL
prediction<-predictOutcome(modelSign2,testing)
```

Let's estimate errors of each prediction model and the combined. Note, that I stopped at combined model, so I must not change our choice even if by some circumstances it's error will be larger.


```r
#rf
confusionMatrix(dlSign$fmTest$signupCompleted,prediction$rf)$overall['Accuracy']
```

```
##  Accuracy 
## 0.9457143
```

```r
#gbm
confusionMatrix(dlSign$fmTest$signupCompleted,prediction$gbm)$overall['Accuracy']
```

```
##  Accuracy 
## 0.8696429
```

```r
#neural network
confusionMatrix(dlSign$fmTest$signupCompleted,prediction$nnet)$overall['Accuracy']
```

```
##  Accuracy 
## 0.9496429
```

```r
#combined
confusionMatrix(dlSign$fmTest$signupCompleted,prediction$Combined)$overall['Accuracy']
```

```
## Accuracy 
##   0.9525
```

The OOS error is not increased, which is good and means we had no overfitting. It's even decreased a bit, but this is probably by chance.

# Final remarks

To test the model prediction your test data needs to be prepared in the same manner as my training set, i.e., include all transformations made by me to the data. I prepared a function that would do it. The condition is that *your test file* has the same columns and their names as *my training file*. To test the model do the following:  

1. Train my model as I did above:  

```r
dt <- readDT(myTrainFile)
df <- convert(dt,operations = c("numNatId","scaleTimes","logNatId"))
dl <- prepTrainData(df,outcome="signupCompleted")
model <- buildModel(dl, methods=c("rf","gbm","nnet"),
                           combineMethod="nnet", outcome="signupCompleted")
```
2. Prepare *your testing* data set:  

```r
testingData<-prepTestData(myTrainFile,yourTestFile) #must be in identical format
```
3. Perform prediction

```r
#note the testing set should NOT contain predicted variable
testing <- testingData
testing$signupCompleted <- NULL
prediction<-predictOutcome(model,testing)
```
4. Make model evaluation as you think is appropriate. *prediction* is a list of models predictions, including combined model prediction retrieved as:

```r
prediction$Combined
```





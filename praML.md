<div id="wrap"><div class="container"><div class="row row-offcanvas row-offcanvas-right"><div class="contents col-xs-12 col-md-10">---
title: "Practical Machine Learning on Activity Prediction"
author: "Zhao Hao"
date: "Friday, November 21, 2014"
output: html_document
---
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

The goal of this submission is to predict the manner in which they did the exercise which is defined as the "classe" variable in the training set.

#```{r setup, include=FALS}
opts_chunk$set(dev = 'pdf',cache=TRUE,eval=TRUE, message = FALSE, warning=F)
#```

## Survey Data
We first select 10 rows of data to see how many variables there are, and check the data structure, names of the variables and the definition of "classe".

<div class="row"><button class="source R toggle btn btn-xs btn-primary"><span class="glyphicon glyphicon-chevron-down"></span> R source</button><pre style=""><code class="source r">surveyTrain <- read.csv("data/pml-training.csv", nrows =10)
surveyTest <- read.csv("data/pml-testing.csv", nrows=10)
str(surveyTrain);
names(surveyTrain)
surveyTrain$classe</code></pre></div>

Some variables have many NAs. We will clean those variables up.

<div class="row"><button class="source R toggle btn btn-xs btn-primary"><span class="glyphicon glyphicon-chevron-down"></span> R source</button><pre style=""><code class="source r">surveyTrain <- surveyTrain[,complete.cases(t(surveyTrain))]
surveyTrain</code></pre></div>

And we also determines that the first 6 variables are not relevant to the data set, so we also will clean them up.

<div class="row"><button class="source R toggle btn btn-xs btn-primary"><span class="glyphicon glyphicon-chevron-down"></span> R source</button><pre style=""><code class="source r">irVar <- c(1:6)
surveyTrain[,-irVar]</code></pre></div>

## Build, Partition and Tidy Up Data Sets
Now we have a good idea about the data. We now read in all the data and tidy them up.

<div class="row"><button class="source R toggle btn btn-xs btn-primary"><span class="glyphicon glyphicon-chevron-down"></span> R source</button><pre style=""><code class="source r">rm(surveyTrain, surveyTest)

testing <- read.csv("data/pml-testing.csv", na.strings=c("", "NA", "NULL"))
testing <- testing[, complete.cases(t(testing))]
testing <- testing[,-irVar]
rawdata<-read.csv("data/pml-training.csv", na.strings=c("", "NA", "NULL"))
rawdata <- rawdata[, complete.cases(t(rawdata))]
rawdata <- rawdata[,-irVar]</code></pre></div>

And we partition the data into testing, validating and training sets. 

<div class="row"><button class="source R toggle btn btn-xs btn-primary"><span class="glyphicon glyphicon-chevron-down"></span> R source</button><pre style=""><code class="source r">set.seed(112014)
library(lattice)
library(ggplot2)
library(caret)
inTrain <- createDataPartition(rawdata$classe, p=0.7, list=F)
training <- rawdata[inTrain,]
valid <- rawdata[-inTrain,]</code></pre></div>

## Train with Different Models
Now we choose three different models to train on the training data. Meanwhile we record the running time of each model for future comparison.

<div class="row"><button class="source R toggle btn btn-xs btn-primary"><span class="glyphicon glyphicon-chevron-down"></span> R source</button><pre style=""><code class="source r">library(randomForest)
library(plyr)
library(splines)
library(survival)
library(parallel)
library(gbm)
set.seed(235)
ptm.start <- proc.time()
rfMod<-randomForest(classe~., data=training)
ptm.rf <- proc.time() - ptm.start
ptm.start <- proc.time()
gbmMod<-train(classe~., data=training, method="gbm", verbose=F)
ptm.gbm <- proc.time() - ptm.start
ptm.start <- proc.time()
ldaMod<-train(classe~., data=training, method="lda")
ptm.lda <- proc.time() - ptm.start</code></pre></div>

Now we can check which model runs the fastest. Here is the result:

<div class="row"><button class="output R toggle btn btn-xs btn-success"><span class="glyphicon glyphicon-chevron-down"></span> R output</button><pre style=""><code class="output r">## $rf
##    user  system elapsed 
##   39.61    0.13   39.91 
## 
## $gbm
##    user  system elapsed 
## 1773.57    2.15 1783.52 
## 
## $lda
##    user  system elapsed 
##   10.28    0.28   10.77
</code></pre></div>

The linear model finishes the first, not much a surprise. It took quite a while for GBM (Generalized Boosted Regression Modeling) to finish its work. Would it give us the best result?

Now let's figure that out by calculating the accuracy of the predictions on the cross-validation data.

<div class="row"><button class="source R toggle btn btn-xs btn-primary"><span class="glyphicon glyphicon-chevron-down"></span> R source</button><pre style=""><code class="source r">rfpred<-predict(rfMod, valid)
gbmpred<-predict(gbmMod, valid)
ldapred<-predict(ldaMod, valid)</code></pre></div>
with this accuracy function.
<div class="row"><button class="source R toggle btn btn-xs btn-primary"><span class="glyphicon glyphicon-chevron-down"></span> R source</button><pre style=""><code class="source r">accuracy<-function(pre,val){
  sum(pre==val)/length(val)
}</code></pre></div>
Here is the result:
<div class="row"><button class="output R toggle btn btn-xs btn-success"><span class="glyphicon glyphicon-chevron-down"></span> R output</button><pre style=""><code class="output r">## $rf
## [1] 0.9976211
## 
## $gbm
## [1] 0.9882753
## 
## $lda
## [1] 0.7135089
</code></pre></div>
So the winner is actually the random forest, which beats the GBM by 0.01% !!

## Prediction on Test Data
Finally we will use the randomforest model to predict the personal activity. The last cross-validation result is very impressive so we don't have to do a k-fold cross-validation to further evaluate the models. The test data will further prove that this prediction model is quite robust.

We save the result and submit the result for the final evaluation.

<div class="row"><button class="source R toggle btn btn-xs btn-primary"><span class="glyphicon glyphicon-chevron-down"></span> R source</button><pre style=""><code class="source r">answers<-predict(rfMod, testing)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("answers/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)</code></pre></div></div></div>
<div class="navbar navbar-fixed-bottom navbar-inverse"><div class="container"><div class="navbar-header"><button type="button" class="navbar-toggle" data-toggle="collapse" data-target=".navbar-responsive-collapse"><span class="icon-bar"></span>
<span class="icon-bar"></span>
<span class="icon-bar"></span></button></div>
<div id="bottom-navbar" class="navbar-collapse collapse navbar-responsive-collapse"><ul class="nav navbar-nav navbar-right"><li class="nav"><p class="navbar-text">Toggle</p></li>
<li class="dropup"><a href="#" class="dropdown-toggle" data-toggle="dropdown">Code 
<b class="caret"></b></a>
<ul class="dropdown-menu"><li class="dropdown-header">Languages</li>
<li class="active"><a href="#" class="toggle-global source R" type="source.R">R</a></li>
<li ><a href="#" type="all-source" class="toggle-global">All</a></li></ul></li>
<li class="dropup"><a href="#" class="dropdown-toggle" data-toggle="dropdown">Output
<b class="caret"></b></a>
<ul class="dropdown-menu"><li class="dropdown-header">Type</li>
<li class="active"><a href="#" class="toggle-global output" type="output">output</a></li>
<li ><a href="#" type="all-output" class="toggle-global">All</a></li></ul></li>
<li class="active"><a href="#" type="figure" class="toggle-global">Figures</a></li></ul></div></div></div></div>
<div id="push"></div>
<div id="footer"><div class="container"><p class="text-muted" id="credit">Styled with 
<a href="https://github.com/jimhester/knitrBootstrap">knitrBootstrap</a></p></div></div>
<link rel="stylesheet" id="theme" href="https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css" media="screen"></link><link rel="stylesheet" id="highlight" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/7.3/styles/default.min.css" media="screen"></link></div>

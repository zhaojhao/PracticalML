### course project draft
## survey data
surveyTrain <- read.csv("data/pml-training.csv", nrows =10)
surveyTest <- read.csv("data/pml-testing.csv", nrows=10)
str(surveyTrain)
names(surveyTrain)
surveyTrain$classe
## we don't need so many variables with NAs. try removes them
surveyTrain <- surveyTrain[,complete.cases(t(surveyTrain))]
surveyTrain
## we determine that the first 6 variables as irrelevant 
irVar <- c(1:6)
surveyTrain[,-irVar]
### now start to build datasets
rm(surveyTrain, surveyTest)
testing <- read.csv("data/pml-testing.csv", na.strings=c("", "NA", "NULL"))
testing <- testing[, complete.cases(t(testing))]
testing <- testing[,-irVar]
rawdata<-read.csv("data/pml-training.csv", na.strings=c("", "NA", "NULL"))
rawdata <- rawdata[, complete.cases(t(rawdata))]
rawdata <- rawdata[,-irVar]
set.seed(112014)
library(lattice)
library(ggplot2)
library(caret)
inTrain <- createDataPartition(rawdata$classe, p=0.7, list=F)
training <- rawdata[inTrain,]
valid <- rawdata[-inTrain,]
### now train with different models to see which one is the best for the cross-validation
library(randomForest)
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
ptm.lda <- proc.time() - ptm.start

ptm<-list(rf=ptm.rf, gbm=ptm.gbm, lda=ptm.lda)
ptm
rfpred<-predict(rfMod, valid)
gbmpred<-predict(gbmMod, valid)
ldapred<-predict(ldaMod, valid)
accuracy<-function(pre,val){
  sum(pre==val)/length(val)
}
accu <- list(rf=accuracy(rfpred,valid$classe),
     gbm = accuracy(gbmpred,valid$classe),
     lda = accuracy(ldapred,valid$classe))
accu
## the result is out: the winner is ....
## ...
## randomForest
## use it on the testing set and save the answer.
answers<-predict(rfMod, testing)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("answers/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)


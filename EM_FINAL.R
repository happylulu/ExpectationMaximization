
#------------------------------
# stats computing final project
# Programmer : Weilu Han
# EM imputation under GLM and GBM
#---------------------------------------


library(foreign)
require(Amelia)
library(ggplot2)
library(reshape2)
library(gbm)
library(caret)
library(cvAUC)
library(Matrix)
library(xgboost)
library(ROCR)
library(glmnet)
data <- read.arff("C:/Users/happy/Dropbox/Fall 2016/Statistical computing/Final Project/egg.arff.txt")
setwd("C:/Users/happy/Dropbox/Fall 2016/Statistical computing/Final Project/")
dim(data)
summary(data)

data(freetrade)
summary(freetrade)

hist(data$F7, freq =F, breaks =100)
plot.ts(data$F7)
set.seed(1234)
# generate missings for the complete data with 20% and 30 % proportions
insert_nas <- function(x, nmiss) {
  len <- nrow(x)
    i <- sample(1:len, nmiss)
  x[i] <- NA 
  x
}
propmiss <- function(dataframe) lapply(dataframe,function(x) data.frame(nmiss=sum(is.na(x)), n=length(x), propmiss=sum(is.na(x))/length(x)))
 
ny <-ncol(data)
len <- nrow(data)
nmiss<-floor(prop*len/ncol(data))

# missing at Random scenario
 
 
data2 <- data
data2$missgen<- sample(1:500, nrow(data),replace=T)
ids <-sample(data2$missgen, 14)
for (i in 1: 14)
{
data2[data2$missgen == ids[i], i] <- NA
}
require(graphics)
postscript("miss1.ps")
missmap(data2,col = c("white","darkred"))
dev.off()
summary(data2)
norm_em(data2[,1:14], 100,debug=2,10, 10^-10)
print(norm_em(data2[,1:14],20, 100, 10^-10, "imputed_data/imputed1.csv"))
print(norm_em2(data2[,1:14],20, 100, 10^-10, "imputed_data/imputed1b.csv"))


data3 <- data
data3$missgen<- sample(1:140, nrow(data),replace=T)
ids <-sample(data3$missgen, 14)
for (i in 1: 14)
{
  data3[data3$missgen == ids[i], i] <- NA
}
postscript("miss2.ps")
missmap(data3)
dev.off()
print(norm_em(data3[,1:14],20, 100, 10^-3, "imputed_data/imputed2.csv"))
print(norm_em2(data3[,1:14],20, 100, 10^-3, "imputed_data/imputed2b.csv"))

data4 <- data
data4$missgen<- sample(1:77, nrow(data),replace=T)
set.seed(0002)
ids <-sample(data4$missgen, 19, replace=F)
for (i in 1: 14)
{
  data4[data4$missgen == ids[i], i] <- NA
}
postscript("miss3.ps")
missmap(data4)
dev.off()
print(norm_em(data4[,1:14],20, 100, 10^-3, "imputed_data/imputed3.csv"))
print(norm_em2(data4[,1:14],20, 100, 10^-3, "imputed_data/imputed3b.csv"))

write.csv(data2,"genmissing_data/data2.csv")
write.csv(data3,"genmissing_data/data3.csv")
write.csv(data4,"genmissing_data/data4.csv")
write.csv(data,"genmissing_data/data.csv")


## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))


## Gradient boosting function include original and imputed dataset and missing dataset
previous_na_action <- options('na.action')
options(na.action='na.pass')


gb_model <- function(train, test )

{
  y <- "eyeDetection"
 
  train.mx <- sparse.model.matrix(eyeDetection ~ ., train)
  test.mx <- sparse.model.matrix(eyeDetection ~ ., test)
  dtrain <- xgb.DMatrix(train.mx, label = train[,y])
  dtest <- xgb.DMatrix(test.mx, label = test[,y])
  
  train.gdbt <- xgb.train(params = list(objective = "binary:logistic",
                                        #num_class = 2,
                                        #eval_metric = "mlogloss",
                                        eta = 0.3,
                                        max_depth = 5,
                                        subsample = 1,
                                        colsample_bytree = 0.5), 
                          data = dtrain, 
                          nrounds = 70,   
                          watchlist = list(train = dtrain, test = dtest))
  
  # Generate predictions on test dataset
  preds <- predict(train.gdbt, newdata = dtest)
  labels <- test[,y]
  
  # Compute AUC on the test set
  return(cvAUC::AUC(predictions = preds, labels = labels))
  
}


log_model <- function (train, test){
  y <- "eyeDetection"
  train <-train[complete.cases(train),]
  test <- test[complete.cases(test),]
  model <-  glmnet(as.matrix(train[,-ncol(train)]),train[,y],family= 'binomial' )
  p <- predict(model, as.matrix(test[,-ncol(test)]), type="response",s=0.01)
  pr <- prediction(p, test[,y])
  prf <- performance(pr, measure = "tpr", x.measure = "fpr")
  
  auc <- performance(pr, measure = "auc")
  auc <- auc@y.values[[1]]
  return(auc)
  
}
## set the seed to make your partition reproductible
set.seed(1234)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

imputed1 <- read.csv("C:/Users/happy/Dropbox/Fall 2016/Statistical computing/Final Project/imputed_data/imputed1.csv")
imputed2 <- read.csv("C:/Users/happy/Dropbox/Fall 2016/Statistical computing/Final Project/imputed_data/imputed2.csv")
imputed3 <- read.csv("C:/Users/happy/Dropbox/Fall 2016/Statistical computing/Final Project/imputed_data/imputed3.csv")
imputed1b <- read.csv("C:/Users/happy/Dropbox/Fall 2016/Statistical computing/Final Project/imputed_data/imputed1b.csv")
imputed2b <- read.csv("C:/Users/happy/Dropbox/Fall 2016/Statistical computing/Final Project/imputed_data/imputed2b.csv")
imputed3b <- read.csv("C:/Users/happy/Dropbox/Fall 2016/Statistical computing/Final Project/imputed_data/imputed3b.csv")

miss1 <- read.csv("C:/Users/happy/Dropbox/Fall 2016/Statistical computing/Final Project/genmissing_data/data2.csv")
miss2 <- read.csv("C:/Users/happy/Dropbox/Fall 2016/Statistical computing/Final Project/genmissing_data/data3.csv")
miss3 <- read.csv("C:/Users/happy/Dropbox/Fall 2016/Statistical computing/Final Project/genmissing_data/data4.csv")
miss_total <- cbind(do.call(rbind, propmiss(miss1)),
do.call(rbind, propmiss(miss2)),
do.call(rbind, propmiss(miss3)))

write.csv(miss_total, "miss_total.csv")
train <- data[train_ind, ]
train$eyeDetection <-as.numeric(train$eyeDetection)-1 
test <- data[-train_ind, ]
 
test$eyeDetection <-as.numeric(test$eyeDetection)-1

# create train and test sample
data1 <- cbind(imputed1, eyeDetection=data$eyeDetection)
train1 <- data1[train_ind, -1]
trainm1 <-miss1[train_ind, -c(1,ncol(miss1))]
train1$eyeDetection <-as.numeric(train1$eyeDetection)-1
test1 <- data1[-train_ind,-1 ]
testm1 <- miss1[-train_ind, -c(1,ncol(miss1))]
test1$eyeDetection <-as.numeric(test1$eyeDetection)-1

# Set seed because we column-sample
data2 <- cbind(imputed2, eyeDetection=data$eyeDetection)
train2 <- data2[train_ind, -1] 
trainm2 <-miss1[train_ind, -c(1,ncol(miss2))]
train2$eyeDetection <-as.numeric(train2$eyeDetection)-1
test2 <- data2[-train_ind,-1 ]
testm2 <- miss2[-train_ind, -c(1,ncol(miss2))]
test2$eyeDetection <-as.numeric(test2$eyeDetection)-1

data3 <- cbind(imputed3, eyeDetection=data$eyeDetection)
train3 <- data3[train_ind, -1] 
trainm3 <-miss3[train_ind, -c(1,ncol(miss3))]
train3$eyeDetection <-as.numeric(train3$eyeDetection)-1
test3 <- data3[-train_ind,-1 ]
testm3 <- miss3[-train_ind, -c(1,ncol(miss3))]
test3$eyeDetection <-as.numeric(test3$eyeDetection)-1

# create train and test sample with different initial values
data1b <- cbind(imputed1b, eyeDetection=data$eyeDetection)
train1b <- data1b[train_ind, -1]
trainmb1 <-miss1[train_ind, -c(1,ncol(miss1))]
train1b$eyeDetection <-as.numeric(train1b$eyeDetection)-1
test1b <- data1b[-train_ind,-1 ]
testm1b <- miss1[-train_ind, -c(1,ncol(miss1))]
test1b$eyeDetection <-as.numeric(test1b$eyeDetection)-1

# Set seed because we column-sample
data2b <- cbind(imputed2b, eyeDetection=data$eyeDetection)
train2b <- data2b[train_ind, -1]
trainmb2 <-miss2[train_ind, -c(1,ncol(miss2))]
train2b$eyeDetection <-as.numeric(train2b$eyeDetection)-1
test2b <- data2b[-train_ind,-1 ]
testm2b <- miss2[-train_ind, -c(1,ncol(miss2))]
test2b$eyeDetection <-as.numeric(test2b$eyeDetection)-1

data3b <- cbind(imputed3b, eyeDetection=data$eyeDetection)
train3b <- data3b[train_ind, -1]
trainmb3 <-miss3[train_ind, -c(1,ncol(miss3))]
train3b$eyeDetection <-as.numeric(train3b$eyeDetection)-1
test3b <- data3b[-train_ind,-1 ]
testm3b <- miss3[-train_ind, -c(1,ncol(miss3))]
test3b$eyeDetection <-as.numeric(test3b$eyeDetection)-1
# imputed missing
gb_model(train, test)
log_model(train, test)
gb_model(train1, test1)
gb_model(train1b, test1b)
log_model(train1, test1)
gb_model(trainm1, testm1)
log_model(trainm1, testm1)
gb_model(train2, test2)
log_model(train2, test2)
gb_model(trainm2, testm2)
log_model(trainm2, testm2)
gb_model(train3, test3)
log_model(train3, test3)
gb_model(trainm3, testm3)

results <- matrix(NA, nrow=6, ncol=4)
set.seed(1234)
colnames(results) <- c("missing","imputed","impued2","truth")
results[1,1:4] <- c(gb_model(trainm1, testm1),gb_model(train1, test1),gb_model(train1b, test1b),gb_model(train, test))
results[2,1:4] <- c(log_model(trainm1, testm1),log_model(train1, test1),log_model(train1b, test1b),log_model(train, test))
results[3,1:4] <- c(gb_model(trainm2, testm2),gb_model(train2, test2),gb_model(train2b, test2b),gb_model(train, test))
results[4,1:4] <- c(log_model(trainm2, testm2),log_model(train2, test2),log_model(train2b, test2b),log_model(train, test))
results[5,1:4] <- c(gb_model(trainm3, testm3),gb_model(train3, test3),gb_model(train3b, test3b),gb_model(train, test))
results[6,1:4] <- c(log_model(trainm3, testm3),log_model(train3, test3),log_model(train3b, test3b),log_model(train, test))

write.csv(results, "results.csv")
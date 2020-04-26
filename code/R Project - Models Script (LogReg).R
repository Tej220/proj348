rm(list=ls())

setwd("~/Documents/BIS 348/Data")

## install packages and load libraries
library(caret)
library(forecast)
library(gains)

## load data
organics.df <- read.csv("~/Documents/BIS 348/Data/organics.csv", header = TRUE)  # load data
View(organics.df)  # show all the data in a new tab
organics.df <- organics.df[ , -c(1, 13)] # filter out TargetAmt

## visualization
summary(organics.df)  # find summary statistics for each column
str(organics.df) # list the structure of the data

## handle missing data by deleting entire records with missing values
#organics.df[ organics.df == "" ] <- NA # changes all blank data fields to NA
#row.has.na <- apply(organics.df, 1, function(x){any(is.na(x))}) # aggregates all records that contain NA anywhere
#sum(row.has.na) # total number of records with an NA value
#organics.df <- organics.df[!row.has.na, ] # filters out records with NA from full data frame
#organics.df # view filtered data frame

## replace numeric N/A Values with mean values from non-missing data.
organics.df$DemAge[is.na(organics.df$DemAge)] <- mean(organics.df$DemAge, na.rm = TRUE)
organics.df$DemAffl[is.na(organics.df$DemAffl)] <- mean(organics.df$DemAffl, na.rm = TRUE)
organics.df$DemCluster[is.na(organics.df$DemCluster)] <- mean(organics.df$DemCluster, na.rm = TRUE)
organics.df$PromTime[is.na(organics.df$PromTime)] <- mean(organics.df$PromTime, na.rm = TRUE)
organics.df$PromSpend[is.na(organics.df$PromSpend)] <- mean(organics.df$PromSpend, nna.rm = TRUE)

## replace categorical N/A values with common values from non-missing data
organics.df$DemClusterGroup[organics.df$DemClusterGroup == ""] <-  "C"
organics.df$DemGender[organics.df$DemGender == ""] <- "F"
organics.df$DemReg[organics.df$DemReg == ""] <- "South East"
organics.df$DemTVReg[organics.df$DemTVReg == ""] <- "London"
organics.df$PromClass[organics.df$PromClass == ""] <- "Silver"

## correlation table for numeric predictors
round(cor(organics.df[ , c(2:4, 10:12)]), 2) 

## partition data
set.seed(12345)
train.index <- sample(c(1:dim(organics.df)[1]), dim(organics.df)[1]*0.6)
train.df <- organics.df[train.index, ]
valid.df <- organics.df[-train.index, ]

## logistic regression
logit.reg1 <- glm(TargetBuy ~ ., data = train.df, family = "binomial")
options(scipen=999)
summary(logit.reg1)

## prediction
logit.reg1.pred <- predict(logit.reg1, valid.df, type = "response")
data.frame(actual = valid.df$TargetBuy, predicted = round(logit.reg1.pred, 2))
logit.reg1.pred

## plot lift chart
gain.full <- gains(valid.df$TargetBuy, logit.reg1.pred, groups=10)
data.frame("depth" =  gain.full["depth"], "obs" = gain.full["obs"], "cume.obs" = gain.full["cume.obs"], 
           "mean.resp" = gain.full["mean.resp"], "cume.mean.resp" = gain.full["cume.mean.resp"], "cume.pct.of.total" = gain.full["cume.pct.of.total"], 
           "lift" = gain.full["lift"], "cume.lift" = gain.full["cume.lift"], "mean.prediction" = gain.full["mean.prediction"])
plot(c(0, gain.full$cume.pct.of.total*sum(valid.df$TargetBuy))~c(0,gain.full$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(valid.df$TargetBuy))~c(0, dim(valid.df)[1]), lty=2)

#compute deciles and plot decile-wise chart
heights <- gain.full$mean.resp/mean(valid.df$TargetBuy)
midpoints <- barplot(heights, names.arg = gain.full$depth, ylim = c(0,9), 
                     xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise Lift Chart")

## confusion matrix - determined the best cutoff is 0.5
cm0.1 <- confusionMatrix(as.factor(ifelse(logit.reg1.pred > 0.1, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
cm0.2 <- confusionMatrix(as.factor(ifelse(logit.reg1.pred > 0.2, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
cm0.3 <- confusionMatrix(as.factor(ifelse(logit.reg1.pred > 0.3, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
cm0.4 <- confusionMatrix(as.factor(ifelse(logit.reg1.pred > 0.4, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
cm0.5 <- confusionMatrix(as.factor(ifelse(logit.reg1.pred > 0.5, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
cm0.6 <- confusionMatrix(as.factor(ifelse(logit.reg1.pred > 0.6, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
cm0.7 <- confusionMatrix(as.factor(ifelse(logit.reg1.pred > 0.7, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
cm0.8 <- confusionMatrix(as.factor(ifelse(logit.reg1.pred > 0.8, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
cm0.9 <- confusionMatrix(as.factor(ifelse(logit.reg1.pred > 0.9, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]

cutoff <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
cm.all <- c(cm0.1, cm0.2, cm0.3, cm0.4, cm0.5, cm0.6,
                 cm0.7, cm0.8, cm0.9)
plot(cm.all ~ cutoff,
     xlab="Cutoff", ylab="Accuracy", main="", type="l")

## stepwise regression to eliminate correlated predictors
## directions = "backward", "forward", or "both".
organics.reg.step <- step(logit.reg1, direction = "both")
summary(organics.reg.step)  # DemReg, DemTVReg, DemClusterGroup, PromClass, PromSpend were dropped
organics.reg.step.pred <- predict(organics.reg.step, valid.df[ , c(1,2,3,5,10)], type = "response")

## confusion matrix for new model - result was insignificant
step.cm0.1 <- confusionMatrix(as.factor(ifelse(organics.reg.step.pred > 0.1, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
step.cm0.2 <- confusionMatrix(as.factor(ifelse(organics.reg.step.pred > 0.2, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
step.cm0.3 <- confusionMatrix(as.factor(ifelse(organics.reg.step.pred > 0.3, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
step.cm0.4 <- confusionMatrix(as.factor(ifelse(organics.reg.step.pred > 0.4, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
step.cm0.5 <- confusionMatrix(as.factor(ifelse(organics.reg.step.pred > 0.5, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
step.cm0.6 <- confusionMatrix(as.factor(ifelse(organics.reg.step.pred > 0.6, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
step.cm0.7 <- confusionMatrix(as.factor(ifelse(organics.reg.step.pred > 0.7, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
step.cm0.8 <- confusionMatrix(as.factor(ifelse(organics.reg.step.pred > 0.8, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]
step.cm0.9 <- confusionMatrix(as.factor(ifelse(organics.reg.step.pred > 0.9, 1, 0)), as.factor(valid.df$TargetBuy))$overall[1]

step.cutoff <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
step.cm.all <- c(step.cm0.1, step.cm0.2, step.cm0.3, step.cm0.4, step.cm0.5, step.cm0.6,
                 step.cm0.7, step.cm0.8, step.cm0.9)
plot(step.cm.all ~ step.cutoff,
     xlab="Cutoff", ylab="Accuracy", main="", type="l")

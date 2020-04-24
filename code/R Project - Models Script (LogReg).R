rm(list=ls())

setwd("~/Documents/BIS 348/Data")

## install packages and load libraries
library(neuralnet)
library(caret)
library(forecast)
library(gains)
library(pROC)

## load data
organics.df <- read.csv("~/Documents/BIS 348/Data/organics.csv", header = TRUE)  # load data
View(organics.df)  # show all the data in a new tab
organics.df <- organics.df[ , -13] # filter out TargetAmt

## visualization
summary(organics.df)  # find summary statistics for each column
str(organics.df) # list the structure of the data

## generate dummies and factor ?

## handle missing data by deleting entire records with missing values
organics.df[ organics.df == "" ] <- NA # changes all blank data fields to NA
row.has.na <- apply(organics.df, 1, function(x){any(is.na(x))}) # aggregates all records that contain NA anywhere
sum(row.has.na) # total number of records with an NA value
organics.df.filtered <- organics.df[!row.has.na, ] # filters out records with NA from full data frame
organics.df.filtered # view filtered data frame

## OR handle numeric fields by filling in missing data with column mean
organics.df$DemAge[is.na(organics.df$DemAge)] <- mean(organics.df$DemAge, na.rm = TRUE)
    # continue for all numeric variables

##correlation table
round(cor(organics.df.filtered[ , c(2:4, 10:13)]), 2) 

## partition data
set.seed(12345)
train.index <- sample(c(1:dim(organics.df.filtered)[1]), dim(organics.df.filtered)[1]*0.6)
train.df <- organics.df.filtered[train.index, ]
valid.df <- organics.df.filtered[-train.index, ]

## logistic regression
logit.reg1 <- glm(TargetBuy ~ ., data = train.df, family = "binomial")
options(scipen=999)
summary(logit.reg1)

## prediction
logit.reg1.pred <- predict(logit.reg1, valid.df, type = "response")
data.frame(actual = valid.df$TargetBuy, predicted = round(logit.reg1.pred, 2))
logit.reg1.pred

## confusion matrix - logistic regression model (79% accuracy)
cmatrix <- confusionMatrix(as.factor(ifelse(logit.reg1.pred > 0.5, 1, 0)), as.factor(valid.df$TargetBuy))
cmatrix


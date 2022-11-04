library(kernlab)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(Matrix)
library(caret)
library(gains)

data(spam)
df<- spam


df$type <- as.character(df$type)
df$type[df$type == "nonspam"] <- 0
df$type[df$type == "spam"] <- 1
df$type <- as.factor(df$type)

set.seed(1)  
train.index <- sample(c(1:dim(df)[1]), dim(df)[1]*0.6)  
valid.index <- setdiff(c(1:dim(df)[1]), train.index) 

train.df <- df[train.index,]
valid.df <- df[-train.index,]


# need to change minbucket
cart <- rpart(type ~ make + address + all + num3d + our + over + 
                remove + internet + order + receive + will + free + business + 
                email + you + credit + your + font + num000 + money + hp + 
                hpl + george + labs + data + num415 + num85 + technology + 
                num1999 + parts + meeting + original + project + re + edu + 
                table + conference + charSemicolon + charExclamation + charDollar + 
                charHash + capitalLong + capitalTotal, data = train.df)


options(scipen = 10)
printcp(cart)

pfit <- prune(cart, cp = cart$cptable[which.min(cart$cptable[,"xerror"]),"CP"])
pfit

prp(cart)
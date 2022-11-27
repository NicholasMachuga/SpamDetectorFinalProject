library(kernlab)
library(rpart)
library(rpart.plot)
library(Matrix)
library(caret)
library(gains)
require(tree)

data(spam)
validation.spam.ct <- spam
validation.spam.ct$type <- factor(ifelse(validation.spam.ct$type=="spam",1,0))

#partition the data into train (60%) and validation (40%) sets
#set the seed for the random number generator for reproducing the partition.
set.seed(1)  
train.index <- sample(c(1:dim(validation.spam.ct)[1]), dim(validation.spam.ct)[1]*0.6)  
valid.index <- setdiff(c(1:dim(validation.spam.ct)[1]), train.index) 

train.spam <- validation.spam.ct[train.index,]
valid.spam <- validation.spam.ct[-train.index,]

# Fits  classification tree model
spam.tree <- rpart(type ~ make + address + all + num3d + our + over + 
                     remove + internet + order + receive + will + free + business + 
                     email + you + credit + your + font + num000 + money + hp + 
                     hpl + george + labs + data + num415 + num85 + technology + 
                     num1999 + parts + meeting + original + project + re + edu + 
                     table + conference + charSemicolon + charExclamation + charDollar + 
                     charHash + capitalLong + capitalTotal, data = train.spam, method="class")

#Prints the complexity parameter of the fitted model.  
printcp(spam.tree)

# classification tree
prp(spam.tree, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)

# classify validation records 
spam.tree.pred.valid <- predict(spam.tree,valid.spam,type = "class")

# confusion matrix 
confusionMatrix(spam.tree.pred.valid, valid.spam$type)

# variable of importance
t(t(spam.tree$variable.importance))


#############################################################################
# 10 Fold CART Cross Validation 
set.seed(1)

# object, controls how the train function creates the model
tr_control <- trainControl(method="cv", number = 10)

CVDmodel <-  train(type ~ make + address + all + num3d + our + over + 
                     remove + internet + order + receive + will + free + business + 
                     email + you + credit + your + font + num000 + money + hp + 
                     hpl + george + labs + data + num415 + num85 + technology + 
                     num1999 + parts + meeting + original + project + re + edu + 
                     table + conference + charSemicolon + charExclamation + charDollar + 
                     charHash + capitalLong + capitalTotal, data = spam, method="rpart",trControl=tr_control)
print(CVDmodel)



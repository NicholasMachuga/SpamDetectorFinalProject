
library(kernlab)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(Matrix)
library(caret)
library(gains)
library(ISLR)
require(tree)

data(spam)
df<- spam

df$type <- as.factor(df$type)
table(df$type)

#partition the data into train (60%) and validation (40%) sets
#set the seed for the random number generator for reproducing the partition.
set.seed(1)  
train.index <- sample(c(1:dim(df)[1]), dim(df)[1]*0.6)  
valid.index <- setdiff(c(1:dim(df)[1]), train.index) 

train.df <- df[train.index,]
valid.df <- df[-train.index,]

# Fits  classification tree model
spam.tree <- rpart(type ~ make + address + all + num3d + our + over + 
                     remove + internet + order + receive + will + free + business + 
                     email + you + credit + your + font + num000 + money + hp + 
                     hpl + george + labs + data + num415 + num85 + technology + 
                     num1999 + parts + meeting + original + project + re + edu + 
                     table + conference + charSemicolon + charExclamation + charDollar + 
                     charHash + capitalLong + capitalTotal, data = train.df, method="class")

#Prints the complexity parameter of the fitted model.  
printcp(spam.tree)

#OUTPUT 
# Variables actually used in tree construction:
#   [1] capitalLong     charDollar      charExclamation  free  hp  remove         
# 
# Root node error: 1109/2760 = 0.40181
# n= 2760 
# CP nsplit rel error  xerror     xstd
# 1 0.475203      0   1.00000 1.00000 0.023225
# 2 0.075744      1   0.52480 0.56357 0.019827
# 3 0.060415      2   0.44905 0.45176 0.018260
# 4 0.030658      4   0.32822 0.33544 0.016177
# 5 0.023445      5   0.29757 0.29757 0.015370
# 6 0.015329      6   0.27412 0.28133 0.015000
# 7 0.013526      7   0.25879 0.27322 0.014809
# 8 0.010000      8   0.24527 0.25789 0.014438

prp(spam.tree, type=1, extra=1, split.font=1, varlen=-10)

# classification tree
spam.tree <- rpart(type ~., data =train.df, method="class")
prp(default.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)

# classify training records 
spam.tree.pred.train <- predict(spam.tree,train.df,type = "class")
# confusion matrix for training data
confusionMatrix(spam.tree.point.pred.train, train.df$type)

# Output
# Confusion Matrix and Statistics
# 
# Reference
# Prediction nonspam spam
# nonspam    1556  170
# spam         95  939
# 
# Accuracy : 0.904           
# 95% CI : (0.8924, 0.9147)
# No Information Rate : 0.5982          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.798           
# 
# Mcnemar's Test P-Value : 0.000005473     
#                                           
#             Sensitivity : 0.9425          
#             Specificity : 0.8467          
#          Pos Pred Value : 0.9015          
#          Neg Pred Value : 0.9081          
#              Prevalence : 0.5982          
#          Detection Rate : 0.5638          
#    Detection Prevalence : 0.6254          
#       Balanced Accuracy : 0.8946          
#                                           
#        'Positive' Class : nonspam         
#                                



# classify validation records 
spam.tree.pred.valid <- predict(spam.tree,valid.df,type = "class")
# confusion matrix for validation data
confusionMatrix(spam.tree.pred.valid, valid.df$type)

#OUTPUT
# Confusion Matrix and Statistics
# 
# Reference
# Prediction nonspam spam
# nonspam    1053  130
# spam         84  574
# 
# Accuracy : 0.8838          
# 95% CI : (0.8682, 0.8981)
# No Information Rate : 0.6176          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.7508          
# 
# Mcnemar's Test P-Value : 0.002097        
#                                           
#             Sensitivity : 0.9261          
#             Specificity : 0.8153          
#          Pos Pred Value : 0.8901          
#          Neg Pred Value : 0.8723          
#              Prevalence : 0.6176          
#          Detection Rate : 0.5720          
#    Detection Prevalence : 0.6426          
#       Balanced Accuracy : 0.8707          
#                                           
#        'Positive' Class : nonspam    

# [NOT COMPLETE ]Working on 10 fold cross validation
# cross-validation with 10 folds (argument xval)
# argument cp sets the smallest value for the complexity parameter.
spam.tree2 <-  rpart(type ~ make + address + all + num3d + our + over + 
                       remove + internet + order + receive + will + free + business + 
                       email + you + credit + your + font + num000 + money + hp + 
                       hpl + george + labs + data + num415 + num85 + technology + 
                       num1999 + parts + meeting + original + project + re + edu + 
                       table + conference + charSemicolon + charExclamation + charDollar + 
                       charHash + capitalLong + capitalTotal, data = train.df, method="class", cp=0.00001, xval=10)
printcp(spam.tree2)

# Prune tree by the lower complexity parameter
pfit <- prune(spam.tree2, cp = spam.tree2$cptable[which.min(spam.tree$cptable[,"xerror"]),"CP"])
pfit

# Return value of last node of low complexity parameter
pam.tree2$cptable[which.min(spam.tree2$cptable[,"xerror"]),"CP"]

# length(spam.tree2$frame$var[spam.tree2$frame$var == "<leaf>"])

# Best pruned tree 
prp(spam.tree2, type = 1, extra = 1, split.font = 1, varlen = -10)




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

# classification tree
prp(spam.tree, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)

# classify training records 
spam.tree.pred.train <- predict(spam.tree,train.df,type = "class")
# confusion matrix for training data
confusionMatrix(spam.tree.pred.train, train.df$type)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction nonspam spam
# nonspam    1543  164
# spam        108  945
# 
# Accuracy : 0.9014          
# 95% CI : (0.8897, 0.9123)
# No Information Rate : 0.5982          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.7933          
# 
# Mcnemar's Test P-Value : 0.0008534       
#                                           
#             Sensitivity : 0.9346          
#             Specificity : 0.8521          
#          Pos Pred Value : 0.9039          
#          Neg Pred Value : 0.8974          
#              Prevalence : 0.5982          
#          Detection Rate : 0.5591          
#    Detection Prevalence : 0.6185          
#       Balanced Accuracy : 0.8934          
#                                           
#        'Positive' Class : nonspam         
                                  

# classify validation records 
spam.tree.pred.valid <- predict(spam.tree,valid.df,type = "class")
# confusion matrix for validation data
confusionMatrix(spam.tree.pred.valid, valid.df$type)

#OUTPUT
# Confusion Matrix and Statistics
# 
# Reference
# Prediction nonspam spam
# nonspam    1065  117
# spam         72  587
# 
# Accuracy : 0.8973          
# 95% CI : (0.8826, 0.9108)
# No Information Rate : 0.6176          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.78            
# 
# Mcnemar's Test P-Value : 0.001372        
#                                           
#             Sensitivity : 0.9367          
#             Specificity : 0.8338          
#          Pos Pred Value : 0.9010          
#          Neg Pred Value : 0.8907          
#              Prevalence : 0.6176          
#          Detection Rate : 0.5785          
#    Detection Prevalence : 0.6420          
#       Balanced Accuracy : 0.8852          
#                                           
#        'Positive' Class : nonspam         

# variable of importance
t(t(spam.tree$variable.importance))

#############################################################################
# 10 Fold cross validation using naive Bayes
# naive Baye
set.seed(1)

# object, controls how the train function creates the model

set.seed(1)
tr_control <- trainControl(method="cv", number =10)
CVDmodel <-  train(type ~ make + address + all + num3d + our + over + 
                  remove + internet + order + receive + will + free + business + 
                  email + you + credit + your + font + num000 + money + hp + 
                  hpl + george + labs + data + num415 + num85 + technology + 
                  num1999 + parts + meeting + original + project + re + edu + 
                  table + conference + charSemicolon + charExclamation + charDollar + 
                  charHash + capitalLong + capitalTotal, data = df, method="rpart",trControl=tr_control)
print(CVDmodel)

# BEST ACCURACY FOR 10-FOLD-CVD CP= 0.0430226, ACCURACY=86.35%
# CART 
# 
# 4601 samples
# 43 predictor
# 2 classes: 'nonspam', 'spam' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 4142, 4141, 4141, 4141, 4140, 4140, ... 
# Resampling results across tuning parameters:
#   
#   cp          Accuracy   Kappa    
# 0.04302261  0.8635064  0.7087469
# 0.14892443  0.7898371  0.5474296
# 0.47655819  0.6907037  0.2518403
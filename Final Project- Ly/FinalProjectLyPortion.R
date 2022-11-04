library(leaps) 
library(gains) 
library(forecast) 
library(dplyr)
library(caret)
library(ISLR)
library(datatartium)
library(ggplot2)

#Data
data = data(spam, package = "kernlab")

#Correlation Matrix


#Changing $type to numeric for GLM
spam$type <- as.character(spam$type)
spam$type[spam$type == "nonspam"] <- 0
spam$type[spam$type == "spam"] <- 1
spam$type <- as.numeric(spam$type)

#Train and test split, 60% split
set.seed(1)
dt = sort(sample(nrow(spam), nrow(spam)*.6))
train<-spam[dt,]
validation<-spam[-dt,]


##############################################################
#                        Base Model                          #
##############################################################

#Creating base model log reg
base_model <- glm(type~., data = train)
base_model_pred <- predict(base_model, validation)

##Cutoff at .5
base_model_output <- ifelse(base_model_pred > 0.5, 1, 0)

###Changing classes to help with confusion matrix
base_model_output <- as.numeric(base_model_output)
accuracy(base_model_output, validation$type)
tab_base <- table(Predicted = base_model_output, Actual = validation$type)
confusionMatrix(tab_base)
#         Actual
#Predicted    0     1
#        0    1085  136
#        1    52    568
#Accuracy : 0.8979  



#Same base model, but with 10-fold CV
base_model_val <- train(type~.,
                   data = spam2,
                   method = "multinom",
                   trControl = 
                     trainControl(
                       method = "cv", number = 10,
                       verboseIter = TRUE))

base_model_val_output <- predict(base_model_val, validation)

base_model_val_output2 <- as.numeric(base_model_val_output)

base_model_val_output2[base_model_val_output2 == 1] <- 0
base_model_val_output2[base_model_val_output2 == 2] <- 1

#Accuracy and Confusion matrix
accuracy(base_model_val_output2, validation$type)
tab_base_val <- table(Predicted = base_model_val_output2, Actual = validation$type)
confusionMatrix(tab_base_val)
#                 Actual
#    Predicted    0      1
#           0     1079   74
#           1     58     630


##############################################################
#                        Model 2                             #
#   Using exhaustive search to find significant predictors   #
##############################################################


#Searching to find which columns are significant
search <- regsubsets(type~ ., data = train, nbest = 1, method = "exhaustive", really.big=T)
sum <- summary(search)

#Log regression model using significant results
model <- glm(type~ remove + free + your + num000 + money + hp + charDollar + capitalLong, data = train, family=binomial(link="logit"))


#Prediction using model
model_pred <- predict(model, validation)


#Cutoff at .5
model_output <- ifelse(model_pred > 0.5, 1, 0)

#Combining output prediction to test set
validation$output <- model_output

accuracy(validation$output, validation$type)
#         ME        RMSE      MAE       MPE   MAPE
#Test set 0.1075502 0.3955208 0.1564367 -Inf  Inf

#Confusion Matrix
tab1 <- table(Predicted = model_output, Actual = validation$type)
confusionMatrix(tab1)
##################################################
#             Actual                              
#Predicted    Nonspam  Spam
#     Nonspam 1094     187
#        Spam   43     517
# Model predicted 1094/1137 as non spam and 517/704 as spam
# Potential overfitting with 96% predicted as nonspam and 73% as spam

#Using cross val on model 1
model1_cv <- train(type~remove + free + your + num000 + money + hp + 
                        charDollar + capitalLong, 
                   data = spam2, 
                   method = "multinom",
                   trControl = trainControl(method = "cv", 
                                            number = 10,
                                            verboseIter = TRUE))
model1_cv


#Prediction using model
model1_pred_cv <- predict(model1_cv, validation)
model1_pred_cv <- 
accuracy(model1_pred_cv)

##############################################################
#                        Model 3                             #
#      With Cross Validation and Stepwise search             #
##############################################################
#Recreating test set
validation2 <- validation


#Using stepwise to find the best prediction
model_2 <- glm(type ~., data = train)
step(model_2, direction = "both")

#exclude: mail, people, report, addresses, num650, lab, telnet, num857, pm, direct, cs, charroundbracket, charsquarebracket
model_3 <- glm(formula = type ~ make + address + all + num3d + our + over + 
                 remove + internet + order + receive + will + free + business + 
                 email + you + credit + your + font + num000 + money + hp + 
                 hpl + george + labs + data + num415 + num85 + technology + 
                 num1999 + parts + meeting + original + project + re + edu + 
                 table + conference + charSemicolon + charExclamation + charDollar + 
                 charHash + capitalLong + capitalTotal, data = train)


#Second prediction
predict_2 <- predict(model_3, validation2)
model_output2 <- ifelse(predict_2 > 0.5, 1, 0)
validation2$output <- model_output2

#Accuracy
accuracy(validation2$output, validation2$type)
#         ME         RMSE     MAE       MPE   MAPE
#Test set 0.04997284 0.326288 0.1064639 -Inf  Inf

#Confusion Matrix
tab2 <- table(Predicted = validation2$output, Actual = validation2$type)
confusionMatrix(tab2)

# Actual
#Predicted    NonSpam  Spam
#     NonSpam 1085     144
#        Spam 52       560
#Model2 predicted 1085/1137 as non spam and 560/704 as spam
# Accuracy : 0.8935  <-- Increased accuracy by 2%
#increased spam detection to 80%

#Utilizing Cross validation on final model on full dataset
spam2 <- spam
spam2$type <- as.factor(spam2$type)


set.seed(1)

model_val <- train(type~make + address + all + num3d + our + over + 
                     remove + internet + order + receive + will + free + business + 
                     email + you + credit + your + font + num000 + money + hp + 
                     hpl + george + labs + data + num415 + num85 + technology + 
                     num1999 + parts + meeting + original + project + re + edu + 
                     table + conference + charSemicolon + charExclamation + charDollar + 
                     charHash + capitalLong + capitalTotal, 
                   data = spam2,
                   method = "multinom",
                        trControl = 
                              trainControl(
                                           method = "cv", number = 10,
                                           verboseIter = TRUE))

model_val

#Prediction using cross validation model
validation3 <- validation

validation3$type <- as.factor(validation3$type)

model_val_pred <- predict(model_val, validation3)

validation3$type <- as.numeric(validation3$type)
model_val_pred_num <- as.numeric(model_val_pred)
accuracy(model_val_pred_num, validation$type)

tab3 <- table(Predicted = model_val_pred_num, Actual = validation3$type)
confusionMatrix(tab3)

#             Actual
#Predicted    0      1
#        0    1084   79
#        1    53     625

#Accuracy : 0.9283  

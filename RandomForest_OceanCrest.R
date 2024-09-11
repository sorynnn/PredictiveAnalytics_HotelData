# Libraries
library(tidyverse)
library(caret)
library(car)
library(skimr)
library(DataExplorer)
library(ROSE)
library(e1071)
library(glmnet)
library(Matrix)
library(doParallel)
library(ROCR)
library(randomForest)
library(rpart.plot)
library(rattle)
library(randomForest)

#start parallel processing
cl <- makePSOCKcluster(2)
registerDoParallel(cl)

model_rf <- train(is_canceled ~ .,
                  data = OC_train_smote,
                  method = "rf",
                  tuneGrid= expand.grid(mtry = c(1,3,6,9)),
                  trControl = trainControl(method = "cv", number = 5))

#stop parallel processing
stopCluster(cl)
registerDoSEQ()

model_rf
plot(model_rf)
model_rf$bestTune

plot(model_rf)

###Step 3 - Get Predictions using Testing Set Data
#First, get the predicted probabilities of the test data.
predprob_rf <- predict(model_rf, OC_test, type="prob")

###Step 4 - Evaluate Model Performance
pred_rf <- prediction(predprob_rf$is_canceled, OC_test$is_canceled,label.ordering =c("not_is_canceled","is_canceled") )
perf_rf <- performance(pred_rf, "tpr", "fpr")
par(mar = c(5, 4, 4, 2) + 0.1)
plot(perf_rf, colorize=TRUE)

# visualize model
varImpPlot(model_rf$finalModel)
plot(varImp(model_rf))

twoClassSummary(OC_test, lev = levels(OC_test$obs))

#Get the AUC
auc_rf <- unlist(slot(performance(pred_rf, "auc"), "y.values"))

auc_rf

# Export
saveRDS(model_rf, "model_rf.rds")
# Load
loaded_model_rf <- readRDS("model_rf.rds")

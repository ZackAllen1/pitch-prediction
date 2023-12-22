library(baseballr)
library(dplyr)
library(ggplot2)
library(randomForest)
library(gbm)
library(caret)
library(pROC)
library(naivebayes)
library(MASS)

## BASELINE MODEL

getBaseline <- function(trainData, testData){
  baseline_pred <- ifelse((sum(trainData$isFastball == 1) / nrow(trainData)) > (sum(trainData$isFastball == 0) / nrow(trainData)), 1, 0)
  baseline_error <- mean(baseline_pred != testData$isFastball)
  
  return(baseline_error)
}

## NAIVE BAYES

getNB <- function(trainData, testData, nameNoSpace){
  # fit model
  model <- naive_bayes(as.factor(isFastball) ~ ., data = trainData)
  cat(summary(model), "\n")
  
  # class0prob, class1prob, prediction, outcome, isCorrect
  p_all <- predict(model, newdata=testData, type = 'prob')
  p_all <- cbind(p_all, pred = ifelse(p_all[, 2] > p_all[, 1], 1, 0))
  p_all <- cbind(p_all, outcome = testData$isFastball )
  p_all <- cbind(p_all, isCorrect = (testData$isFastball == p_all[,3] ))
  
  # find accuracy of test predictions, based on minimum threshold values
  result_df <- data.frame(threshold = numeric(), accuracy = numeric(), stringsAsFactors = FALSE)
  thresholds <- c(0.5, 0.6, 0.75, 0.8, 0.9, 0.95)
  # Iterate through thresholds
  for (threshold in thresholds) {
    # Filter rows where the greater probability is greater than the threshold
    filtered_rows <- subset(p_all, pmax(p_all[, 1], p_all[, 2]) > threshold)
    
    # Calculate the percentage of rows where isCorrect is equal to 1
    accuracy <- mean(filtered_rows[,5])
    
    # Store results in the data frame
    result_df <- rbind(result_df, data.frame(threshold = threshold,
                                             accuracy = accuracy,
                                             num_rows = nrow(filtered_rows),
                                             precentage_rows = (nrow(filtered_rows)/nrow(testData))*100))
  }
  
  cat(paste(nameNoSpace, "Naive Bayes Results\n"))
  print(result_df)
  return(result_df)
}


## MULTIPLE LOGISTIC REGRESSION

getMLR <- function(trainData, testData, nameNoSpace){
  # fit glm
  glm_model <- glm(isFastball ~ ., data = trainData)
  
  print(summary(glm_model))
  glm_pred <- ifelse(predict(glm_model, newdata = testData) > 0.5, 1, 0)
  test_error <- mean(glm_pred != testData$isFastball)
  cat(nameNoSpace, " testError = ", test_error)
  
  # plot ROC and record AUC
  auc_obj <- roc(testData$isFastball ~ predict(glm_model, newdata = testData))
  auc <- auc(auc_obj)
  cat(nameNoSpace, " auc = ", auc, "\n")
  
  roc(testData$isFastball ~ predict(glm_model, newdata = testData), plot = TRUE, print.auc = TRUE, asp=NA)
  
  # save plot
  file_path <- file.path(paste0("data/", nameNoSpace), paste0(nameNoSpace, "_roc.jpeg"))
  
  dev.copy(jpeg, filename = file_path)
  dev.off()
  
  return(c(test_error, auc))
}


## GENERALIZED BOOSTED MODEL

gbmCV <- function(trainData, nameNoSpace, printIter=FALSE, replaceParams = FALSE){
  # Hyperparameter Tuning for GBM with grid-search 5-fold CV
  # WARNING: This will take 20-30 minutes per-pitcher
  gbmGrid <- expand.grid(interaction.depth = c(1,2,3),
                         n.trees = (0:25)*100,
                         shrinkage = c(0.0005, 0.005, 0.05),
                         n.minobsinnode = c(1,5,10))
  
  cv <- train(as.factor(isFastball) ~ ., data = trainData,
                 method = "gbm", trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE),
                 tuneGrid = gbmGrid)
  
  # save/return best parameters
  bestParams <- cv$bestTune
  nTrees <- bestParams[[1]]
  depth <- bestParams[[2]]
  shrinkage <- bestParams[[3]]
  minObs <- bestParams[[4]]
  
  # plot and save learning curve
  file_path <- file.path(paste0("data/", nameNoSpace), paste0(nameNoSpace, "_gbmCV.png"))
  png(file_path, width=1120, height=625, units="px", pointsize = 14)
  p <- plot(cv)
  print(p)
  dev.off()
  
  # plot and save feature importance
  file_path <- file.path(paste0("data/", nameNoSpace), paste0(nameNoSpace, "_gbmFI.png"))
  png(file_path, width=940, height=650, units="px", pointsize = 14)
  par(mar = c(4, 8, 2, 2))
  summary(cv, las=2)
  dev.off()
  
  # if parameters have not been saved, save them
  gbmParamList <- read.csv("./best_gbm_params.csv")
  rowNum <- which(gbmParamList$Pitcher == nameNoSpace)
  if(length(rowNum) == 0){
    gbmParamList <- rbind(gbmParamList, c(nameNoSpace, nTrees, depth, shrinkage, minObs))
  } else{
    if(replaceParams == TRUE){
      gbmParamList[rowNum,] = c(nameNoSpace, nTrees, depth, shrinkage, minObs)
    }
  }
  write.csv(gbmParamList, "./best_gbm_params.csv", row.names = FALSE)
  
  
  return(c(nTrees, depth, shrinkage, minObs))
}

getGBM <- function(trainData, testData, nTrees, depth, shrink, minObs){
  # fit model with parameters
  gbmModel <- gbm(isFastball ~ ., data = trainData, distribution = "bernoulli",
                  n.trees = nTrees, interaction.depth = depth, shrinkage = shrink, n.minobsinnode = minObs)
  
  # make predictions on test set and get test error
  test_gbm_pred <- ifelse(predict(gbmModel, newdata= testData,
                                 n.trees= nTrees, interaction.depth = depth, shrinkage = shrink, n.minobsinnode = minObs,
                                 type="response") > 0.5, 1, 0)
  
  gbm_test_error <- mean(test_gbm_pred != testData$isFastball)
  
  return(gbm_test_error)
}
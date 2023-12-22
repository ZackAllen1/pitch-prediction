library(baseballr)
library(dplyr)
library(ggplot2)
library(randomForest)
library(gbm)
library(caret)
source("preprocess.R")
source("models.R")

# Set which pitchers are going to be tested
pitcher_names <- c("Justin Verlander", "Zach Eflin", "Clayton Kershaw", "Blake Snell", "Shohei Ohtani")

# -----------------------------------------------------------------------

## Data Gathering
## NOTE: This section can be skipped if any of the 5-original pitchers are being used,
##       otherwise run this section to gather the necessary data for each pitcher

for(p in pitcher_names){
  name_arr <- unlist(strsplit(p, split = " "))
  firstName <- name_arr[1]
  lastName <- name_arr[2]
  
  # get/clean data
  pitcherID <- getPitcherID(firstName, lastName)
  df <- getPitcherData(pitcherID, 2015, 2023)
  final_df <- cleanData(df)
  
  saveToCsv(final_df, firstName, lastName)
}

# -----------------------------------------------------------------------

## Get Results
results <- matrix(data = NA, nrow = length(pitcher_names), ncol = 7)
findBestGBM <- FALSE

for(i in 1:length(pitcher_names)){
  nameNoSpace <- gsub(" ", "", pitcher_names[i])
  
  # load pitcher data
  pData <- read.csv(paste0("./data/", nameNoSpace, "/", nameNoSpace, "_data.csv"))
  
  factor_cols = c("stand", "on_3b", "on_2b", "on_1b", "outs_when_up",
                  "count", "isRecentFB", "year")
  pData[factor_cols] = lapply(pData[factor_cols], factor)
  
  
  set.seed(2)
  
  # Split Data into Train, Val, Test (.70, .15, .15)
  splitSample <- sample(1:3, size=nrow(pData), prob=c(0.7,0.15,0.15), replace = TRUE)
  trainData <- pData[splitSample == 1,]
  testData <- pData[splitSample == 2,]
  valData <- pData[splitSample == 3,]
  
  # Get Baseline Test Error
  baseline_error <- getBaseline(trainData, testData)
  
  # Get Naive Bayes Varying Threshold Table
  nbTable <- getNB(trainData, testData, nameNoSpace)
  nb_test_error <- 1 - nbTable[1,2]
  
  # Get MLR Test Error and AUC
  mlr_result <- getMLR(trainData, testData, nameNoSpace)
  
  # get best parameters for GBM
  if(findBestGBM == TRUE){
    # WARNING: This will take 20-30 minutes per-pitcher
    # printIter = TRUE will print CV info in console
    # replaceParams = TRUE will update best_gbm_params.csv if possible
    gbmParams = gbmCV(trainData, nameNoSpace, printIter=TRUE, replaceParams = TRUE)
  } else {
    gbmParamList <- read.csv("./best_gbm_params.csv")
    rowNum <- which(gbmParamList$Pitcher == nameNoSpace)
    
    # if no match is found throw error
    if(length(rowNum) == 0){
      stop("Pitcher Name not found in best_gbm_params.csv")
    }
    else{
      gbmParams = c(gbmParamList[rowNum, 2], gbmParamList[rowNum, 3],
                    gbmParamList[rowNum, 4], gbmParamList[rowNum, 5])
      cat(paste("GBM Parameters for", nameNoSpace, "found.\n"))
    }
  }
  

  # fit gbm model based on best parameters and get test error
  gbm_test_error <- getGBM(trainData, testData,
                           gbmParams[1], gbmParams[2], gbmParams[3], gbmParams[4])
  
  # update results table
  results[i,] = c(pitcher_names[i], nrow(pData), baseline_error, nb_test_error, mlr_result[1], mlr_result[2], gbm_test_error)
}

# create final dataframe and save to csv
final_df <- as.data.frame(results)
colnames(final_df) <- c("Pitcher", "NumPitches", "Baseline_TestError", "NB_TestError", "MLR_TestError", "MLR_AUC", "GBM_TestError")
write.csv(final_df, paste0("./results_", format(Sys.time(), "%F_%H:%M:%S"), ".csv"), row.names = FALSE)
cat("Analysis Complete!")



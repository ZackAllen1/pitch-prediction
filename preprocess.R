library(baseballr)
library(dplyr)
library(ggplot2)
library(randomForest)
library(gbm)
library(caret)

# get pitcher MLB ID for data scraping
getPitcherID <- function(firstName, lastName){
  idData <- playerid_lookup(last_name = lastName, first_name = firstName)
  mlb_id <- idData %>% pull(mlbam_id)
  return(mlb_id)
}

# scrape pitcher data 
getPitcherData <- function(pitcherID, startYear, endYear){
  pitcherData <- data.frame()
  for(y in startYear:endYear){
    curr_year_data <- as.data.frame(statcast_search_pitchers(start_date = paste0(y, "-03-01", sep=""),
                                                             end_date = paste0(y, "-10-01", sep=""),
                                                             pitcherid = pitcherID))
    pitcherData <- rbind(pitcherData, curr_year_data)
    cat(paste0(y, "-> data collected"))
  }
  
  # get regular season only
  pitcherData <- pitcherData[pitcherData$game_type == "R",]
  
  return(pitcherData)
}


# HELPER FUNCTION 1: calculate # pitches since last breaking ball
calculateNumSinceBB <- function(atBat){
  result <- c(0)
  
  if(nrow(atBat) == 1){
    return(result)
  }
  
  count <- 0
  for(i in 2:nrow(atBat)){
    if(atBat[i-1, 4] == 1){
      count <- count + 1
    }
    if(atBat[i-1,4] == 0){
      count <- 0
    }
    result <- append(result, count)
  }
  return(result)
}

# HELPER FUNCTION 2: determine what the most recent pitch was
mostRecentPT <- function(atBat){
  result <- c(-1)
  
  if(nrow(atBat) == 1){
    return(result)
  }
  
  for(i in 2:nrow(atBat)){
    if(atBat[i-1, 4] == 1){
      result <- append(result, 1)
    }
    else{
      result <- append(result, 0)
    }
  }
  return(as.factor(result))
}



# preprocess pitcher data
cleanData <- function(pitcherData){
  # get columns necessary for cleaning
  base_cols = c("game_date", "game_pk", "pitch_type", "batter", "at_bat_number", "description", 
                "stand", "balls", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up",
                "pitch_number", "inning")
  
  p_data <- pitcherData %>% select(base_cols)
  
  # replace NA with 0, and any runner number with 1
  p_data$on_3b[is.na(p_data$on_3b)] <- 0
  p_data$on_3b[p_data$on_3b > 0] <- 1
  
  p_data$on_2b[is.na(p_data$on_2b)] <- 0
  p_data$on_2b[p_data$on_2b > 0] <- 1
  
  p_data$on_1b[is.na(p_data$on_1b)] <- 0
  p_data$on_1b[p_data$on_1b > 0] <- 1
  
  # convert some columns to factor
  factor_cols = c("pitch_type", "stand", "on_3b", "on_2b", "on_1b", "outs_when_up")
  p_data[factor_cols] = lapply(p_data[factor_cols], factor)
  
  # remove any rows if they dont have 0-3 balls and 0-2 strikes
  p_data <- p_data[p_data$balls %in% c(0,1,2,3) & p_data$strikes %in% c(0,1,2),]
  
  # convert balls and strikes to count
  p_data$count = as.factor(paste(as.character(p_data$balls), "-", as.character(p_data$strikes), sep = ""))
  
  # add factor for year
  p_data$year = as.factor(format(p_data$game_date, '%Y'))
  
  
  
  # create isFastball from pitch_type
  p_data <- p_data %>%
    mutate(isFastball = case_when(
      # 4-seam, 2-seam/sinker, cutter, or splitter
      pitch_type %in% c("FF", "FT", "SI", "FC", "FS") ~ 1,
      # change-up, curveball, eephus, forkball, gyroball, knuckle-curve,
      # knuckleball, screwball, slider, or sweeper
      pitch_type %in% c("CH", "CU", "EP", "FO", "GY", "KC", "KN", "SC", "SL", "ST") ~ 0,
      # anything else
      TRUE ~ -1
    )) %>% select(1:3, isFastball, everything())
  
  # create pitchResult from description
  p_data <- p_data %>%
    mutate(pitchResult = as.factor(case_when(
      # ball
      description %in% c("ball", "blocked_ball", "pitchout", "intent_ball") ~ "B",
      
      # looking strike
      description %in% c("called_strike") ~ "LS",
      
      # swinging strike
      description %in% c("foul", "foul_bunt", "foul_tip", "missed_bunt", "swinging_strike",
                         "swinging_strike_blocked") ~ "SS",
      
      # anything else
      TRUE ~ "O"
    ))) %>% select(1:4, pitchResult, everything())
  
  cat("added column set 1")
  
  # create FBStreak and isRecentFB using each atbat
  group_ids <- p_data %>% group_by(game_pk, batter, at_bat_number) %>%
    group_keys() %>% arrange(game_pk, at_bat_number)
  
  result <- data.frame()
  for(i in 1:nrow(group_ids)){
    g <- group_ids[i,]
    atbat <- filter(p_data, game_pk == g[[1]] & batter == g[[2]]  &
                      at_bat_number == g[[3]] ) %>% arrange(pitch_number)
    
    atbat$FBStreak <- calculateNumSinceBB(atbat)
    atbat$isRecentFB <- mostRecentPT(atbat)
    result <- rbind(result, atbat)
  }
  
  cat("added column set 2 (FBStreak and isRecentFB)")
  
  group_ids2 <- p_data %>% group_by(game_pk, batter) %>% group_keys()
  result2 <- data.frame()
  for(i in 1:nrow(group_ids2)){
    g2 <- group_ids2[i,]
    gameBatter <- filter(result, game_pk == g2[[1]] & batter == g2[[2]]) %>% arrange(at_bat_number)
    
    rle_result <- rle(gameBatter$at_bat_number)
    gameBatter$abNum <- rep(seq_along(rle_result$values), rle_result$lengths)
    result2 <- rbind(result2, gameBatter)
  }
  
  cat("added column set 3 (abNum)")
  
  # remove all pitches which aren't FB/OS
  result2 <- result2[result2$isFastball != -1,]
  
  # drop unnecessary columns for analysis
  drops <- c("game_date", "game_pk","pitch_type", "batter", "at_bat_number", "description", "pitchResult",
             "balls", "strikes", "pitch_number", "bat_score", "fld_score", "runadv", "recentRes", "type")
  final_df <- result2[, !(names(result2) %in% drops)]
  
  cat("completed!")
  return(final_df)
}


saveToCsv <- function(df, firstName, lastName){
  fullName <- paste0(firstName, lastName)
  subdirectory_path  <- paste0("data/", fullName)
  fileName <- paste0(fullName, "_data.csv")
  
  # Check if the subdirectory exists, create it if not
  if (!file.exists(subdirectory_path)) {
    dir.create(subdirectory_path, recursive = TRUE)
  }
  
  csv_file_path <- file.path(subdirectory_path, fileName)
  write.csv(df, file = csv_file_path, row.names = FALSE)
  cat("CSV file saved in:", csv_file_path, "\n")
}



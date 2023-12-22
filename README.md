# pitch-prediction
Per-pitcher Binary Classifier Analysis as part of STA4241 Final Project

## Setup
Clone the repository then move into the pitch-prediction directory
```bash
git clone https://github.com/ZackAllen1/pitch-prediction.git
cd ./pitch-prediction
```
Open the `main.R` file and load the necessary packages and helper files in lines 1-8. To properly install the [`baseballr` package](https://billpetti.github.io/baseballr/) you may have to run the following code to install it off of GitHub instead of CRAN:
```R
if (!requireNamespace('pacman', quietly = TRUE)){
  install.packages('pacman')
}
pacman::p_load_current_gh("BillPetti/baseballr")
```

Insert the names of the pitchers you wish to analyze as `"FirstName LastName"` elements in the `pitcher_names` list on Line 11. If you wish to analyze the original five pitchers (Justin Verlander, Zach Eflin, Clayton Kershaw, Blake Snell, and Shohei Ohtani), the proceeding **Data Gathering** section of code can be skipped since the datasets for those pitchers is already collected.

In the **Get Results** section, finding the best hyperparameters for GBM using grid-search and 5-fold CV can take 20-30 minutes per pitcher. For the original five pitchers, the best hyperparameters found in the study have been saved in the `best_gbm_params.csv` file. To use these parameters instead of recalculating, set `findBestGBM` to `FALSE` on Line 36. Anytime a new pitcher has their hyperparameters calculated they will be automatically saved in `best_gbm_params.csv` so they do not have to be recalculated on future runs.

After running the **Get Results** section, all results will be saved in `results.csv`. Any plots or datasets generated will be saved in each pitcher's `./data/pitcherName/` directory.

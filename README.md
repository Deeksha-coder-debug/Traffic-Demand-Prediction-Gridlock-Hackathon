# Traffic-Demand-Prediction-Gridlock-Hackathon

## 1st version of the solution 

==============================================================================
GRIDLOCK HACKATHON — TRAFFIC DEMAND PREDICTION
METHODOLOGY & APPROACH DOCUMENTATION
==============================================================================

PROBLEM STATEMENT
-----------------
Predict traffic demand (a continuous value) at various geographic locations and
timestamps using features: road type, lanes, weather, temperature, landmarks,
and geohash-encoded location.

Evaluation Metric: score = max(0, 100 * R²(actual, predicted))

==============================================================================
KEY DESIGN DECISION — LEAKAGE-FREE APPROACH
==============================================================================

An earlier version of this solution used lag features (demand_lag_1h,
demand_lag_24h, etc.) and rolling window features computed per geohash group.
While those inflated the CV R² to 0.9427, they constituted look-ahead leakage:
at inference time, future demand values are not available to construct lags for
the test set. This solution deliberately removes all lag and rolling features
to ensure CV scores are realistic and predictions generalise to unseen data.

==============================================================================
SOLUTION OVERVIEW
==============================================================================

A four-model simple-average ensemble:
  1. LightGBM         — gradient-boosted trees
  2. CatBoost         — gradient-boosted trees
  3. Random Forest    — bagged decision trees
  4. Gradient Boosting — sklearn GradientBoostingRegressor

All models trained on identical 22 base features (no lags, no rolling).
Final predictions = arithmetic mean of the four models' outputs.

==============================================================================
CROSS-VALIDATION RESULTS
==============================================================================

TimeSeriesSplit with 5 folds (respects temporal ordering):

  Fold 1 — Train: 12,884  |  Val: 12,883
  Fold 2 — Train: 25,767  |  Val: 12,883
  Fold 3 — Train: 38,650  |  Val: 12,883
  Fold 4 — Train: 51,533  |  Val: 12,883
  Fold 5 — Train: 64,416  |  Val: 12,883

  Model              CV R²           Expected Score
  -------------------------------------------------------
  LightGBM           0.7586 ± 0.0368     75.86
  CatBoost           0.7631 ± 0.0354     76.31
  Random Forest      0.7630 ± 0.0417     76.30
  Gradient Boosting  0.7657 ± 0.0390     76.57
  -------------------------------------------------------
  Ensemble average                    ~  76.26 / 100

Note: Folds 4–5 show lower R² (~70–73) because later time periods in the
dataset appear to have different demand distributions, making them harder
to predict from earlier training data alone.

==============================================================================
FEATURE ENGINEERING (22 FEATURES — NO LAGS)
==============================================================================

1. TIMESTAMP PARSING
   - HH:MM timestamps combined with the `day` column to build a proper datetime
     anchored to 2024-01-01.
   - Rows with unparseable timestamps dropped.

2. TIME FEATURES
   - hour, dayofweek, month, quarter
   - Binary flags: is_weekend, is_morning_rush (7–9 AM weekdays),
     is_evening_rush (5–7 PM weekdays), is_night (10 PM–6 AM)
   - Cyclical encoding: hour_sin, hour_cos, dow_sin, dow_cos

3. GEOSPATIAL FEATURES
   - geohash decoded to (latitude, longitude) via geohash2 library
   - geohash_4: 4-character prefix for region-level grouping
   - dist_from_center: Euclidean distance from spatial median centroid

4. CATEGORICAL ENCODING
   - RoadType, Weather: LabelEncoder fitted on training data only;
     unseen test values mapped to -1
   - LargeVehicles, Landmarks: boolean strings mapped to binary 0/1

5. INTERACTION FEATURES
   - temp_x_hour: Temperature × hour
   - temp_x_weekend: Temperature × is_weekend
   - bad_weather_x_rush: (Rain/Snow/Storm) × rush hour indicator
   - lanes_x_temperature: NumberofLanes × Temperature

6. MISSING VALUE HANDLING
   - Temperature: filled with training set median (applied to both train/test)
   - NumberofLanes: filled with column median
   - Categorical: filled with 'Missing' before encoding

==============================================================================
MODEL DETAILS
==============================================================================

1. LightGBM
   - objective: regression (RMSE), num_leaves: 31, learning_rate: 0.05
   - feature_fraction: 0.8, bagging_fraction: 0.8, bagging_freq: 5
   - CV rounds: 300  |  Final rounds: 500

2. CatBoost
   - iterations: 300, learning_rate: 0.05, depth: 6, random_seed: 42

3. Random Forest
   - CV: n_estimators=100, max_depth=10
   - Final: n_estimators=200, max_depth=12

4. Gradient Boosting (sklearn)
   - CV: n_estimators=100, max_depth=5, learning_rate=0.05
   - Final: n_estimators=200, max_depth=6, learning_rate=0.05

Final test predictions = (LGB + CB + RF + GB) / 4, clipped to [0, max(y_train)]

==============================================================================
POST-PROCESSING
==============================================================================

- Predictions clipped to [0, max(y_train)] — prevents out-of-range values
- No extreme predictions: 0% of predictions equal exactly 0 or 1
- Prediction stats: min=0.0154, max=0.9867, mean=0.1382

==============================================================================
TOOLS & LIBRARIES USED
==============================================================================

Language     : Python 3
ML Libraries : lightgbm, catboost, scikit-learn (RandomForestRegressor,
               GradientBoostingRegressor, Ridge, LabelEncoder,
               StandardScaler, TimeSeriesSplit, r2_score)
Geo Encoding : geohash2
Data         : pandas, numpy
Utilities    : warnings, gc, random, datetime

==============================================================================
FILE DESCRIPTIONS
==============================================================================

gridlock_leakage_free_solution.py  — Full leakage-free solution script
submission.csv                     — Final predictions (41778 × 2: Index, demand)
methodology.txt                    — This file

==============================================================================


## 2nd version of the solution

==============================================================================
GRIDLOCK HACKATHON — TRAFFIC DEMAND PREDICTION
METHODOLOGY & APPROACH DOCUMENTATION
==============================================================================

PROBLEM STATEMENT
-----------------
Predict traffic demand (a continuous value) at various geographic locations and
timestamps using features: geohash, road type, lanes, weather, temperature,
landmarks, day, and timestamp.

Evaluation Metric: score = max(0, 100 * R²(actual, predicted))

==============================================================================
LEAKAGE-FREE DESIGN PRINCIPLE
==============================================================================

This solution deliberately excludes lag features and rolling window features.
An earlier version used demand_lag_Nh and roll_mean_Nh which inflated CV R²
to 0.94 but scored 0 on the leaderboard — those features require future target
values that are unavailable at test time. This solution uses only features
derivable from the raw inputs (time, location, weather, road), making CV
scores a realistic estimate of test performance.

==============================================================================
PIPELINE OVERVIEW (8 STEPS)
==============================================================================

STEP 1 — LOADING DATA
----------------------
  Train shape: (77299, 11)
  Test shape:  (41778, 10)

STEP 2 — PREPROCESSING
-----------------------
  Train after preprocessing: (77299, 11)
  Test after preprocessing:  (41778, 10)

  - Numeric coercion: NumberofLanes, Temperature, day cast via pd.to_numeric
  - Boolean mapping: LargeVehicles and Landmarks ('true'/'false'/etc.) → 0/1
  - Timestamp parsing: HH:MM strings + day column → full datetime anchored to
    2024-01-01; unparseable rows dropped
  - Missing values:
      Temperature   → filled with training-set median (same value applied to test)
      NumberofLanes → filled with column median (default 1 if all missing)
      Categoricals  → filled with 'Missing' before encoding

STEP 3 — FEATURE ENGINEERING (31 features, no lags)
-----------------------------------------------------
  Final training shape: (77299, 31)
  Final test shape:     (41778, 31)

  1. TIME FEATURES
     - Extracted: hour, dayofweek, month, quarter, dayofyear, weekofyear
     - Binary flags: is_weekend, is_morning_rush (7–9 AM weekdays),
       is_evening_rush (5–7 PM weekdays), is_night (10 PM–6 AM)
     - Cyclical encoding (preserves circular nature of time):
         hour_sin, hour_cos   → sin/cos of hour over 24-hour cycle
         dow_sin, dow_cos     → sin/cos of day-of-week over 7-day cycle
         month_sin, month_cos → sin/cos of month over 12-month cycle

  2. GEOSPATIAL FEATURES
     - geohash decoded to (latitude, longitude) using geohash2 library
     - Hierarchical geohash prefixes: geohash_4, geohash_5, geohash_6
       (encoded via LabelEncoder fitted on training data only)
     - dist_from_center: Euclidean distance from spatial median centroid

  3. INTERACTION FEATURES
     - temp_x_hour:        Temperature × hour of day
     - temp_x_weekend:     Temperature × is_weekend
     - temp_squared:       Temperature² (captures nonlinear temperature effects)
     - weather_code:       Ordinal encoding (Clear=0, Clouds=1, Rain=2, Snow=3, Storm=4)
     - bad_weather_x_rush: (Rain/Snow/Storm) × rush hour indicator
     - road_code:          Ordinal encoding (Residential=0, Arterial=1, Highway=2)
     - lanes_x_temperature: NumberofLanes × Temperature
     - lanes_x_hour:        NumberofLanes × hour

STEP 4 — TRAIN/VALIDATION SPLIT
---------------------------------
  Temporal 80/20 split (preserves time ordering, prevents leakage):
    Training samples:   61,839  (first 80% chronologically)
    Validation samples: 15,460  (last 20% chronologically)

STEP 5 — TRAINING ADVANCED ENSEMBLE
-------------------------------------
  (See detailed model descriptions below)

STEP 6 — SELECTING BEST ENSEMBLE METHOD
-----------------------------------------
  Three ensemble strategies compared on the validation set:

  ┌──────────────────────┬──────────┬───────────────┐
  │ Ensemble Method      │ Val R²   │ Expected Score│
  ├──────────────────────┼──────────┼───────────────┤
  │ Voting Ensemble      │ 0.6773   │ 67.73         │
  │ Blending Ensemble    │ 0.7046   │ 70.46         │
  │ Stacking Ensemble  🏆│ 0.7416   │ 74.16         │
  └──────────────────────┴──────────┴───────────────┘

  → Stacking Ensemble selected as best method.

STEP 7 — TEST PREDICTIONS
---------------------------
  Predictions - Min: 0.0104, Max: 1.0000, Mean: 0.1508
  Extreme values at 0 or 1: 0%
  Predictions clipped to [0, max(y_train)]

STEP 8 — SUBMISSION
---------------------
  submission_ensemble.csv  →  Shape (41778, 2), columns: Index, demand

==============================================================================
BASE MODELS — DETAILS & RATIONALE
==============================================================================

  ┌──────────────────┬──────────┬───────────────┐
  │ Model            │ Val R²   │ Expected Score│
  ├──────────────────┼──────────┼───────────────┤
  │ LightGBM         │ 0.5967   │ 59.67         │
  │ CatBoost         │ 0.6353   │ 63.53         │
  │ Random Forest    │ 0.6891   │ 68.91         │
  │ Gradient Boosting│ 0.6567   │ 65.67         │
  │ Extra Trees      │ 0.6423   │ 64.23         │
  │ AdaBoost         │ 0.6000   │ 60.00         │
  │ XGBoost          │ 0.5954   │ 59.54         │
  └──────────────────┴──────────┴───────────────┘

1. LightGBM (n_estimators=500, num_leaves=31, lr=0.05, max_depth=8)
   WHY: Microsoft's gradient-boosted framework optimised for speed and memory.
   Handles large tabular datasets efficiently via leaf-wise tree growth and
   histogram-based binning. Strong default performance on structured data.

2. CatBoost (iterations=500, depth=6, lr=0.05)
   WHY: Yandex's gradient booster with native handling of categorical features
   via ordered target statistics, avoiding target leakage during encoding.
   Often outperforms other boosters on datasets with mixed feature types.

3. Random Forest (n_estimators=300, max_depth=12, min_samples_split=10)
   WHY: Bagging ensemble of independent decision trees; low variance via
   averaging. Provides diversity to the ensemble — its predictions are less
   correlated with the boosting models, which improves stacking performance.
   Best single-model score (R²=0.6891) in this experiment.

4. Gradient Boosting — sklearn (n_estimators=300, max_depth=6, lr=0.05)
   WHY: Sequential additive tree learner; each tree corrects residuals of the
   previous. More conservative than LightGBM (no GPU, no histogram tricks) but
   often generalises better on smaller splits and adds model diversity.

5. Extra Trees (n_estimators=200, max_depth=10)
   WHY: Extremely Randomised Trees use fully random split thresholds rather than
   best-split search. This added randomness reduces variance further and makes
   predictions decorrelated from Random Forest — valuable for ensemble diversity.

6. AdaBoost (n_estimators=200, lr=0.05)
   WHY: Adaptive Boosting up-weights misclassified samples in each iteration,
   focusing subsequent learners on hard examples. Complementary to tree-based
   boosters as it uses shallow base learners and a different loss mechanism.
   Blending assigns it 33.2% weight — it contributes uniquely to the ensemble.

7. XGBoost (n_estimators=500, max_depth=6, lr=0.05)
   WHY: Extreme Gradient Boosting with L1/L2 regularisation built in. Included
   conditionally (if installed). Adds an additional boosting perspective with
   column and row subsampling; slightly lower individual score than LightGBM
   but adds ensemble diversity.

==============================================================================
ENSEMBLE METHODS — DETAILS & RATIONALE
==============================================================================

1. VOTING ENSEMBLE (Weighted Average)
   - Each model's weight = its validation R² / sum(all R² scores)
   - Models with higher individual accuracy get proportionally more influence
   - Simple, interpretable, robust to meta-overfitting
   - Result: R² = 0.6773 | Score = 67.73

2. BLENDING ENSEMBLE (Scipy Optimised Weights)
   - Uses scipy.optimize.minimize (SLSQP) to find the convex combination of
     base model predictions that maximises R² on the validation set
   - Constraints: weights ≥ 0, weights sum to 1
   - Optimal weight distribution found:
       Random Forest:    0.266
       Gradient Boosting: 0.402
       AdaBoost:         0.332
   - Result: R² = 0.7046 | Score = 70.46

3. STACKING ENSEMBLE (Meta-Learner) ← SELECTED
   - Base model predictions on validation set used as meta-features (7 columns)
   - Four meta-learners compared on the same validation set:
       Ridge (α=1.0):          R² = 0.7404
       Lasso (α=0.01):         R² = 0.1192
       ElasticNet (α=0.01):    R² = 0.5118
       Linear Regression:      R² = 0.7416  ← selected
   - LinearRegression selected as best meta-learner (R² = 0.7416)
   - WHY STACKING WINS: Rather than fixed or optimised weights, the meta-learner
     learns an unconstrained linear function of all 7 base model outputs,
     allowing it to capture complementary strengths across models. This beats
     both weighted averaging and blending on this dataset.
   - Final test predictions generated using stacking with LinearRegression
     meta-learner combining all 7 base model predictions.
   - Result: R² = 0.7416 | Score = 74.16

==============================================================================
FINAL EXPECTED SCORE
==============================================================================

  🏆 74.16 / 100

  This is a realistic, leakage-free estimate. CV is performed on held-out
  temporal data (last 20% of sorted training set) with no target information
  from the test set used during feature engineering or training.

==============================================================================
TOOLS & LIBRARIES USED
==============================================================================

Language     : Python 3 (Jupyter Notebook / Google Colab)
ML Libraries : lightgbm, catboost, xgboost, scikit-learn
               (RandomForestRegressor, GradientBoostingRegressor,
                ExtraTreesRegressor, AdaBoostRegressor,
                Ridge, Lasso, ElasticNet, LinearRegression,
                LabelEncoder, StandardScaler, r2_score)
Optimisation : scipy.optimize.minimize (SLSQP) for blending weights
Geo Encoding : geohash2
Data         : pandas, numpy
Utilities    : warnings, gc, random, datetime

==============================================================================
FILE DESCRIPTIONS
==============================================================================

gridlock_hackathon_ensemble_solution.ipynb — Full solution notebook
submission_ensemble.csv                    — Final predictions (41778 × 2)
methodology.txt                            — This file

==============================================================================


##3rd version of the solution

##Final version of the solution

==============================================================================
GRIDLOCK HACKATHON — TRAFFIC DEMAND PREDICTION
METHODOLOGY & APPROACH DOCUMENTATION
==============================================================================

PROBLEM STATEMENT
-----------------
Predict traffic demand (a continuous value) at geographic locations and
timestamps using: geohash, road type, number of lanes, large vehicles,
landmarks, weather, temperature, day, and timestamp.

Evaluation Metric: score = max(0, 100 * R²(actual, predicted))

==============================================================================
LEAKAGE-FREE DESIGN
==============================================================================

This solution uses only features derivable from raw inputs (time, location,
weather, road characteristics). No lag or rolling window features are used,
ensuring CV scores reflect genuine generalisation to unseen test data.

==============================================================================
PIPELINE OVERVIEW
==============================================================================

STEP 1 — DATA LOADING
----------------------
  Train : (77299, 11)
  Test  : (41778, 10)

STEP 2 — DATA CLEANING
-----------------------
  - NumberofLanes, Temperature, day: cast via pd.to_numeric, errors='coerce'
  - LargeVehicles: mapped ('Allowed' → 1, 'Not Allowed' → 0), NaN → 0
  - Landmarks: mapped ('Yes' → 1, 'No' → 0), NaN → 0
  - NumberofLanes: NaN filled with 1
  - Temperature: NaN filled with column median
  - RoadType, Weather: NaN filled with 'Unknown'
  - Timestamp parsed as HH:MM strings → (hour, minute) integers;
    unparseable values default to hour=12, minute=0
  - dayofweek derived as (day - 1) % 7

STEP 3 — GEOHASH PROCESSING
-----------------------------
  - Hierarchical geohash prefixes extracted:
      gh_4 (first 4 chars), gh_5 (first 5), gh_6 (first 6)
  - All three encoded via LabelEncoder fitted on training data only;
    unseen test values mapped to -1

STEP 4 — FEATURE ENGINEERING (32 features total)
--------------------------------------------------
  1. CYCLICAL TIME ENCODING
     Preserves the circular nature of time — hour 23 is close to hour 0:
       hour_sin, hour_cos  →  sin/cos of hour over 24-hour cycle
       dow_sin,  dow_cos   →  sin/cos of dayofweek over 7-day cycle

  2. BINARY TIME FLAGS
       is_weekend    : dayofweek ≥ 5
       is_rush_hour  : hour 7–9 or 17–19
       is_night      : hour ≥ 22 or ≤ 5

  3. GAUSSIAN RUSH PROBABILITY
       rush_prob = exp(-((hour-8)/2)²) + exp(-((hour-18)/2)²)
     Smooth continuous proxy for rush hour intensity — peaks at 8 AM and
     6 PM, tapering naturally. More informative than a binary flag.

  4. WEATHER FEATURES
       weather_severity: ordinal map
         Sunny=0, Clear=0.1, Clouds=0.2, Foggy=0.4,
         Rainy=0.7, Snowy=0.9, Storm=1.0, Unknown=0.3
       bad_weather: weather_severity > 0.5 (binary)

  5. TEMPERATURE FEATURES
       temp_normalized : (Temperature - 20) / 15
       temp_optimal    : exp(-((Temperature - 22) / 10)²)
         Gaussian centred at 22°C — captures the idea that extreme
         temperatures (hot or cold) reduce travel demand.

  6. ROAD TYPE FEATURES
       road_type    : ordinal (Residential=0, Arterial=1, Highway=2)
       road_capacity: domain-mapped throughput estimate
         (Residential=500, Arterial=1200, Highway=2000)
       lanes_normalized : NumberofLanes / 5
       total_capacity   : NumberofLanes × road_capacity

  7. INTERACTION FEATURES
     Cross-product terms capture combined effects:
       weather_rush      : weather_severity × rush_prob
       weekend_rush      : is_weekend × rush_prob
       landmark_rush     : Landmarks × rush_prob
       large_vehicle_rush: LargeVehicles × rush_prob
       road_rush         : road_type × rush_prob
       hour_squared      : (hour / 12)²

  8. GEOHASH ENCODED FEATURES
       gh_4_enc, gh_5_enc, gh_6_enc (LabelEncoder)

  Feature selection: all numeric columns except Index, demand, timestamp,
  raw geohash strings, minute, day, dayofweek (32 retained features).
  All NaNs replaced with 0 before training.

STEP 5 — SCALING
-----------------
  StandardScaler fitted on training data only; applied to both train and test.
  Required by gradient boosted models for numerical stability.

STEP 6 — TRAIN / VALIDATION SPLIT
-----------------------------------
  Temporal 80/20 split (preserves time ordering):
    Train      : 61,839 samples (first 80%)
    Validation : 15,460 samples (last 20%)

==============================================================================
MODELS — DETAILS & RATIONALE
==============================================================================

Five base models used in both Version 1 and Version 2 configurations:

┌───────────────┬──────────────────────────────────────────────────────────┐
│ Model         │ Rationale                                                │
├───────────────┼──────────────────────────────────────────────────────────┤
│ Extra Trees   │ Extremely Randomised Trees use fully random split        │
│ (ET)          │ thresholds. Lower variance via double randomisation      │
│               │ (random features + random thresholds). Computationally   │
│               │ fast and provides decorrelated predictions vs RF.        │
├───────────────┼──────────────────────────────────────────────────────────┤
│ Random Forest │ Bagging ensemble of CART trees. Each tree trained on a  │
│ (RF)          │ bootstrap sample with random feature subsets. Robust to │
│               │ overfitting; best individual R² (0.774) in this run.    │
├───────────────┼──────────────────────────────────────────────────────────┤
│ LightGBM      │ Gradient boosting with leaf-wise tree growth and        │
│ (LGBM)        │ histogram binning. Highly efficient on large tabular     │
│               │ datasets. Row/column subsampling + L1/L2 regularisation │
│               │ prevent overfitting.                                     │
├───────────────┼──────────────────────────────────────────────────────────┤
│ XGBoost       │ Gradient boosting with built-in L1 (reg_alpha) + L2    │
│ (XGB)         │ (reg_lambda) regularisation, row/column subsampling.    │
│               │ Adds a different boosting perspective to the ensemble.   │
├───────────────┼──────────────────────────────────────────────────────────┤
│ CatBoost      │ Gradient boosting with symmetric trees and ordered       │
│               │ target statistics for categorical features. Avoids       │
│               │ target leakage during encoding; strong regularisation.   │
└───────────────┴──────────────────────────────────────────────────────────┘

==============================================================================
ENSEMBLE STRATEGY — TWO VERSIONS + BLENDING
==============================================================================

VERSION 1 (800 estimators / moderate depth)
--------------------------------------------
  Model       Hyperparameters                             Val R²
  ET          n_estimators=800, max_depth=15              0.74784
  RF          n_estimators=800, max_depth=15              0.77388
  LGBM        n_estimators=1000, num_leaves=40, lr=0.025  0.72295
  XGB         n_estimators=1000, max_depth=9,  lr=0.025  0.72549
  CatBoost    iterations=1000,   depth=8,      lr=0.025  0.73112

  Optimised weights (scipy L-BFGS-B, convex combination):
    RF: 1.000  (optimizer concentrated all weight on the best model)
  Ensemble R²: 0.773876  |  Score: 77.39

VERSION 2 (1000–1500 estimators / greater depth)
--------------------------------------------------
  Model       Hyperparameters                             Val R²
  ET          n_estimators=1000, max_depth=16             0.74911
  RF          n_estimators=1000, max_depth=16             0.77471
  LGBM        n_estimators=1500, num_leaves=45, lr=0.02   0.71271
  XGB         n_estimators=1500, max_depth=10, lr=0.02   0.72416
  CatBoost    iterations=1500,   depth=9,      lr=0.02   0.71906

  Optimised weights:
    RF: 1.000
  Ensemble R²: 0.774714  |  Score: 77.47

CROSS-VERSION BLENDING
-----------------------
  Grid search over blend weights [0.1, 0.2, … 0.9] on validation set:
    Best weight for Version 1: 0.1
    Final predictions = 0.1 × preds_v1 + 0.9 × preds_v2
  Blended validation R²: 0.774678  |  Blended Score: 77.47

  Final predictions clipped to [0, 1].

==============================================================================
WEIGHT OPTIMISATION METHOD
==============================================================================

scipy.optimize.minimize with method='L-BFGS-B' (Limited-memory BFGS with
box constraints). Minimises negative R² subject to weights ≥ 0. Weights
normalised to sum to 1 after optimisation. Consistently selected RF as the
dominant model because RF's individual R² (0.774) exceeded all boosting
models on this validation split.

==============================================================================
VALIDATION RESULTS SUMMARY
==============================================================================

  ┌────────────────────────────────┬──────────┬──────────────┐
  │ Configuration                  │ Val R²   │ Score (/100) │
  ├────────────────────────────────┼──────────┼──────────────┤
  │ Version 1 (800 estimators)     │ 0.773876 │ 77.39        │
  │ Version 2 (1000–1500 est.)     │ 0.774714 │ 77.47        │
  │ Blended (0.1×V1 + 0.9×V2)  🏆 │ 0.774678 │ 77.47        │
  └────────────────────────────────┴──────────┴──────────────┘

  🏆 EXPECTED HACKATHON SCORE: 77.47 / 100

==============================================================================
SUBMISSION FILE
==============================================================================

  File  : submission_final_optimized 2.csv
  Shape : (41778, 2)  — columns: Index, demand
  Range : min=0.0043, max=1.0000, mean=0.1255
  Nulls : 0  |  Negatives: 0  |  Values above 1: 0

==============================================================================
TOOLS & LIBRARIES USED
==============================================================================

Language     : Python 3 (Google Colab)
ML Libraries : lightgbm, xgboost, catboost, scikit-learn
               (RandomForestRegressor, ExtraTreesRegressor,
                LabelEncoder, StandardScaler, r2_score)
Optimisation : scipy.optimize.minimize (L-BFGS-B) for weight optimisation
Geo Encoding : geohash2
Data         : pandas, numpy
Utilities    : warnings, os, zipfile

==============================================================================
FILE DESCRIPTIONS
==============================================================================

gridlock_hackathon_final.ipynb          — Full solution notebook (Colab)
submission_final_optimized 2.csv        — Final predictions (41778 × 2)
methodology.txt                         — This file

==============================================================================

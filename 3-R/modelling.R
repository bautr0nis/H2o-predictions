library(h2o)
library(tidyverse)
library(lightgbm)
library(pROC)

h2o.init(max_mem_size = "8g")
getwd()
setwd("/Users/zydrunasbautronis/Documents/KTU-DVDA-PROJECT/project")
df <- h2o.importFile("1-data/train_data.csv")
test_data <- h2o.importFile("1-data/test_data.csv")
df
class(df)
summary(df)

#XPBOOST, random forest, light gbg
y <- "y"
x <- setdiff(names(df), c(y, "id"))
df$y <- as.factor(df$y)
summary(df)

splits <- h2o.splitFrame(df, c(0.6,0.2), seed=123)
train  <- h2o.assign(splits[[1]], "train") # 60%
valid  <- h2o.assign(splits[[2]], "valid") # 20%
test   <- h2o.assign(splits[[3]], "test")  # 20%

"
aml <- h2o.automl(x = x,
                  y = y,
                  training_frame = train,
                  validation_frame = valid,
                  max_runtime_secs = 3600)

aml@leaderboard
"
# Access the AutoML leaderboard


# 
# model <- h2o.getModel("XGBoost_1_AutoML_1_20231212_191845")
# 
# model_id <- "XGBoost_1_AutoML_1_20231212_191845"
# # Retrieve the model using the model ID
# best_model <- h2o.getModel(model_id)
# # Get the hyperparameters of the model
# best_model_params <- best_model@parameters
# # Print the hyperparameters
# print(best_model_params)
# 
# # Get all hyperparameters including default ones
# best_model_all_params <- best_model@allparameters
# # Print all the hyperparameters
# print(best_model_all_params)

model_path <- "4-model/xgb_grid_model_1"

# Load the model
model <- h2o.loadModel(model_path)

##########
h2o.performance(model, train = TRUE)
h2o.performance(model, valid = TRUE)
perf <- h2o.performance(model, newdata = test)
perf

h2o.auc(perf)
plot(perf, type = "roc")

################ PASSING ZE PARAMETERS ###############

xgboost_params <- list(
  model_id = "xgboost_model",
  training_frame = train,
  validation_frame = valid,
  x = names(train)[names(train) != "y"], # replace "response" with your target variable
  y = "y", # replace "response" with your target variable
  learn_rate = 0.3,
  sample_rate = 0.6,
  col_sample_rate = 1.0,
  col_sample_rate_per_tree = 0.9,
  ntrees = 80,
  max_depth = 16,
  min_rows = 10,
  min_child_weight = 10,
  seed = 127,
  stopping_rounds = 3,
  stopping_metric = "AUC",
  stopping_tolerance = 0.001,
  tree_method = "approx"
)

# Training with specific parameters
best_model_so_far <- do.call(h2o.xgboost, xgboost_params)

# Evaluating model
perf <- h2o.performance(best_model_so_far, newdata = valid)
h2o.auc(perf)

h2o.saveModel(best_model_so_far, "4-model/", filename = "best_model_089")

##### FINDING HYPER PARAMS COMBINATION
xgb_hyper_params <- list(
  ntrees = c(70, 80, 90), # Range around your best value
  max_depth = c(14, 15, 16),
  min_rows = c(9, 10, 11),
  min_child_weight = c(9, 10, 11),
  sample_rate = c(0.5, 0.6, 0.7),
  col_sample_rate = c(0.7, 0.8, 0.9)
  # Ensure there's no trailing comma here
)


# Set up grid search
xgb_grid_v2 <- h2o.grid(
  algorithm = "xgboost",
  grid_id = "xgb_grid",
  x = x,
  y = y,
  training_frame = train,
  validation_frame = valid,
  hyper_params = xgb_hyper_params,
  search_criteria = list(strategy = "RandomDiscrete", max_models = 10, seed = 123)
)

# Retrieve and inspect the grid results
grid_results <- h2o.getGrid(grid_id = "xgb_grid", sort_by = "auc", decreasing = TRUE)
print(grid_results)

model_path <- "4-model/best_model_089"

# Load the model
model <- h2o.loadModel(model_path)

perf <- h2o.performance(model, newdata = test)
h2o.auc(perf)


# Adjusting hyperparameters
xgboost_params <- list(
  model_id = "xgboost_model_adjusted",
  training_frame = train,
  validation_frame = valid,
  x = names(train)[names(train) != "y"],
  y = "y",
  learn_rate = 0.01, # Lowered learning rate
  sample_rate = 0.7, # Adjusted sample rate
  col_sample_rate = 0.9, # Adjusted column sample rate
  col_sample_rate_per_tree = 0.8, # Adjusted column sample rate per tree
  ntrees = 100, # Increased number of trees
  max_depth = 20, # Increased max depth
  min_rows = 5, # Adjusted min rows
  min_child_weight = 5, # Adjusted min child weight
  seed = 127,
  stopping_rounds = 5, # Adjusted stopping rounds
  stopping_metric = "AUC",
  stopping_tolerance = 0.001,
  tree_method = "hist" # Changed tree method to hist
)

# Train the adjusted model
xgb_model_adjusted <- do.call(h2o.xgboost, xgboost_params)

# Evaluate the adjusted model
perf <- h2o.performance(xgb_model_adjusted, newdata = valid)
h2o.auc(perf)


summary(xgb_model_adjusted)
h2o.auc(xgb_model_adjusted)
h2o.auc(h2o.performance(xgb_model_adjusted, valid = TRUE))
h2o.auc(h2o.performance(xgb_model_adjusted, newdata = test))

predictions <- h2o.predict(xgb_model_adjusted, test_data)

predictions

predictions %>%
  as_tibble() %>%
  mutate(id = row_number(), y = p0) %>%
  select(id, y) %>%
  write_csv("5-predictions/predictions3(best).csv")

h2o.saveModel(xgb_model_adjusted, "../4-model/", filename = "XGB_MODEL_BEST_SO_FAR")















#h2o.performance(model, newdata = test_data)

predictions <- h2o.predict(model, test_data)

predictions

predictions %>%
  as_tibble() %>%
  mutate(id = row_number(), y = p0) %>%
  select(id, y) %>%
  write_csv("5-predictions/predictions2.csv")

### ID, Y

h2o.saveModel(model, "../4-model/", filename = "my_best_automlmode")
model <- h2o.loadModel("../4-model/my_best_automlmode")
h2o.varimp_plot(model)

h2o.saveModel(rf_model, "../4-model/", filename = "rf_model")

# Write GBM

gbm_model <- h2o.gbm(x = x,
                     y = y,
                     training_frame = train,
                     validation_frame = valid,
                     ntrees = 20,
                     max_depth = 10,
                     stopping_metric = "AUC",
                     seed = 1234)



h2o.auc(gbm_model)
h2o.auc(h2o.performance(gbm_model, valid = TRUE))
h2o.auc(h2o.performance(gbm_model, newdata = test))

# model performance
summary(dl_model)
h2o.auc(dl_model)
h2o.auc(h2o.performance(dl_model, valid = TRUE))
h2o.auc(h2o.performance(dl_model, newdata = test))

# Grid search

dl_params <- list(hidden = list(50, c(50, 50), c(50,50,50)))

dl_grid <- h2o.grid(algorithm = "deeplearning",
                    grid_id = "ktu_grid",
                    x,
                    y,
                    training_frame = train,
                    validation_frame = valid,
                    epochs = 5,
                    stopping_metric = "AUC",
                    hyper_params = dl_params)

hyper_params <- list(
  ntrees = c(50, 100, 150),
  max_depth = c(3, 5, 7),
  learn_rate = c(0.01, 0.1, 0.3),
  sample_rate = c(0.8, 1.0),
  col_sample_rate = c(0.8, 1.0)
)

xgb_grid <- h2o.grid(
  algorithm = "xgboost",
  grid_id = "xg_grid",
  x = x,
  y = y,
  training_frame = train,
  validation_frame = valid,
  nfolds = 5,
  hyper_params = hyper_params,
  search_criteria = list(strategy = "RandomDiscrete", max_models = 10, seed = 123)
)


h2o.getGrid(dl_grid@grid_id, sort_by = "auc")

best_grid <- h2o.getModel(dl_grid@model_ids[[3]])
h2o.auc(h2o.performance(best_grid, newdata = test))
####################


###################

# Pause


##### RANDOM FOREST ####

# Train a Random Forest model
rf_model <- h2o.randomForest(x = x, y = y, training_frame = train, validation_frame = valid, 
                             ntrees = 30, seed = 1234)

# Evaluate AUC on the validation set for Random Forest
rf_perf <- h2o.performance(rf_model, valid = TRUE)
auc_rf <- h2o.auc(rf_perf)
print(paste("Validation AUC for Random Forest: ", auc_rf))


# Destytojo
rf_model <- h2o.randomForest(x = x,
                             y = y,
                             training_frame = train,
                             validation_frame = valid,
                             ntrees = 20,
                             max_depth = 10,
                             stopping_metric = "AUC",
                             seed = 1234)
rf_model
h2o.auc(rf_model)
h2o.auc(h2o.performance(rf_model, valid = TRUE))
h2o.auc(h2o.performance(rf_model, newdata = test))

###############
# Example of hyperparameter tuning for Random Forest in H2O
hyper_params <- list(
  ntrees = c(100, 200, 500), 
  max_depth = c(20, 30, 40),
  min_rows = c(1, 2, 5) # Equivalent to min_samples_leaf in scikit-learn
)

# This is the grid search
rf_grid <- h2o.grid(
  algorithm = "randomForest", 
  grid_id = "rf_grid", 
  hyper_params = hyper_params,
  training_frame = train,
  validation_frame = valid,
  x = x, 
  y = y,
  seed = 123
)

# Get the best model from the grid search
best_rf <- h2o.getGrid("rf_grid", sort_by = "auc", decreasing = True)[[1]]
best_rf_perf <- h2o.performance(best_rf, valid = TRUE)
best_auc_rf <- h2o.auc(best_rf_perf)

print(paste("Best Validation AUC for Random Forest after Grid Search: ", best_auc_rf))


#####################
### XGBoost model ####
# Train an XGBoost model
xgb_model <- h2o.xgboost(x = x, y = y, training_frame = train, validation_frame = valid, 
                         ntrees = 100, max_depth = 5, seed = 123)

# Evaluate AUC on the validation set for XGBoost
xgb_perf <- h2o.performance(xgb_model, valid = TRUE)
auc_xgb <- h2o.auc(xgb_perf)
print(paste("Validation AUC for XGBoost: ", auc_xgb))

#######

# Explain 
exa <- h2o.explain(rf_model, test)
print(exa)
h2o.explain_row(rf_model, test, row_index = 1)

# Imputation
summary(df)
h2o.impute(df, "max_open_credit", method = "median")
summary(df)
h2o.impute(df, "yearly_income", method = "mean", by = c("home_ownership"))
summary(df)

h2o.shutdown(F)
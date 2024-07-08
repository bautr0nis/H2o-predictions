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

h2o.shutdown(F)
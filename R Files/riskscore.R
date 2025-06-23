ls()
rm(list = ls())
getwd()
setwd("E:/3rd year/sem2/ST 3082/Final project")
# Load the data set
data=read.csv(file="Loan.csv",header = TRUE,sep = ",")

library(dplyr)
library(caret)
library(randomForest)
library(ggplot2)
library(xgboost)
library(glmnet)


data <- data %>%
  select(-c(ApplicationDate, AnnualIncome, DebtToIncomeRatio,EducationLevel,Experience,PaymentHistory,
            UtilityBillsPaymentHistory,JobTenure,SavingsAccountBalance,CheckingAccountBalance,TotalAssets))  # Remove LoanApproved column

# Split into training and testing
set.seed(123)
index <- sample(1:nrow(data), 0.2 * nrow(data))  # 20% for test

test_data  <- data[index, ]
train_data <- data[-index, ]

train_data <- train_data %>%select(-LoanApproved)
    

test_data <- test_data %>%select(-LoanApproved) 
  

# Define cross-validation method
train_control <- trainControl(method = "cv", number = 10)  # 10-fold CV

# Define hyperparameter grid
tune_grid <- expand.grid(mtry = c(2, 3),   # Number of variables per split
                         splitrule = "variance", # Default for regression
                         min.node.size = c(5, 10))  # Minimum node size

# Train Random Forest model with cross-validation
rf_cv <- train(RiskScore ~ ., 
               data = train_data,
               method = "ranger",  # Faster implementation of RF
               trControl = train_control,
               tuneGrid = tune_grid,
               num.trees = 100,  # Number of trees
               importance = "impurity")  # Get feature importance

# Print best parameters
print(rf_cv$bestTune)
# Fit Random Forest model
rf_model <- randomForest(RiskScore ~ ., data = train_data, ntree = 100,mtry=3,nodesize=10,importance=TRUE)

# Predictions on training and test data
train_pred<- predict(rf_model, train_data)
test_pred <- predict(rf_model, test_data)


# Function to calculate RMSE
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Calculate RMSE for training and test sets
train_rmse <- rmse(train_data$RiskScore, train_pred)
test_rmse <- rmse(test_data$RiskScore, test_pred)

# Calculate R² for training and test sets
train_r2 <- cor(train_data$RiskScore, train_pred)^2
test_r2 <- cor(test_data$RiskScore, test_pred)^2

# Print results
cat("Train RMSE:", train_rmse, "\n")
cat("Train R²:", train_r2, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R²:", test_r2, "\n")


# Variable Importance Plot
varImpPlot(rf_model)

# Get variable importance from the model
importance_df <- importance(rf_model)
importance_df <- data.frame(Variable = rownames(importance_df), Importance = importance_df[, 1])

# Sort and select top 10 variables
top10_vars <- importance_df[order(importance_df$Importance, decreasing = TRUE), ][1:10, ]

# Reorder factors for plotting
top10_vars$Variable <- factor(top10_vars$Variable, levels = top10_vars$Variable[order(top10_vars$Importance)])

# Create barplot
ggplot(top10_vars, aes(x = Variable, y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Top 10 Important Variables", x = "Variables", y = "Importance") +
  theme_minimal()




#################### Reduced model #####################

# Step 1: Extract the names of the top 10 variables
top10_var_names <- top10_vars$Variable  # This assumes you have the previous 'top10_vars' data frame

# Step 2: Subset the training and test data to only include those variables + target
train_data_top10 <- train_data[, c(as.character(top10_var_names), "RiskScore")]
test_data_top10  <- test_data[, c(as.character(top10_var_names), "RiskScore")]

# Step 3: Fit a new Random Forest model using only the top 10 variables
set.seed(123)  # For reproducibility
rf_model_top10 <- randomForest(
  RiskScore ~ .,
  data = train_data_top10,
  ntree = 200,
  importance = TRUE
)

# Fit Random Forest model
rf_model_top10 <- randomForest(RiskScore ~ ., data = train_data_top10, ntree = 100,mtry=3,nodesize=10,importance=TRUE)


# Step 4: Summary of the model
print(rf_model_top10)

# Step 5: Predictions
pred_train_rf_top10 <- predict(rf_model_top10, train_data_top10)
pred_test_rf_top10 <- predict(rf_model_top10, test_data_top10)



# Train
rmse_train_rf_top10 <- rmse(train_data_top10$RiskScore, pred_train_rf_top10)
r2_train_rf_top10 <- R2(pred_train_rf_top10, train_data_top10$RiskScore)

# Test
rmse_test_rf_top10 <- rmse(test_data_top10$RiskScore, pred_test_rf_top10)
r2_test_rf_top10 <- R2(pred_test_rf_top10, test_data_top10$RiskScore)

# Step 7: Print results
cat("Train RMSE:", rmse_train_rf_top10, "\n")
cat("Train R²:", r2_train_rf_top10, "\n")
cat("Test RMSE:", rmse_test_rf_top10, "\n")
cat("Test R²:", r2_test_rf_top10, "\n")


###########################################################################################
################### XGB ##################


# ************************************** XG Boost ********************************************************

# Convert to matrix (XGBoost requirement)
x_train <- train_data %>% select(-RiskScore)
y_train <- train_data$RiskScore

x_test <- test_data %>% select(-RiskScore)
y_test <- test_data$RiskScore

# Convert data frame with categorical variables to numeric matrix using model.matrix
x_train_matrix <- model.matrix(~ . -1, data = x_train)  
x_test_matrix <- model.matrix(~ . -1, data = x_test)

# Convert data to xgboost's DMatrix format
train_matrix <- xgb.DMatrix(data = x_train_matrix, label = y_train)

test_matrix <- xgb.DMatrix(data = x_test_matrix, label = y_test)


# ************** Optimizing the model ************************

# Define XGBoost parameters
xgb_grid <- expand.grid(
  nrounds = c(50,100),     # Number of boosting rounds
  eta = c(0.01, 0.1),   # Learning rate
  max_depth = c(3, 6) , # Depth of trees
  gamma = c(0, 1),        # Minimum loss reduction
  colsample_bytree = c(0.6,1),  # Feature selection
  min_child_weight = c( 3, 5),
  subsample = c( 0.8, 1)
  
)
set.seed(123)

# Define training control for cross-validation
train_control <- trainControl(
  method = "cv",            # Cross-validation
  number = 5,               # 5-fold CV
  verboseIter = TRUE        # Show progress
)

# Train XGBoost model with CV
xgb_model <- train(
  x = x_train_matrix, 
  y = y_train,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid,
  metric = "RMSE"
)

# Get best tuned parameters
best_params <- xgb_model$bestTune
print(best_params)


# Make predictions on training and testing data
train_pred <- predict(xgb_model, x_train_matrix)
test_pred <- predict(xgb_model, x_test_matrix)

# Calculate RMSE and R-squared for training data
train_rmse <- sqrt(mean((train_pred - y_train)^2))
train_r2 <- 1 - sum((train_pred - y_train)^2) / sum((y_train - mean(y_train)^2)

# Calculate RMSE and R-squared for testing data
test_rmse <- sqrt(mean((test_pred - y_test)^2))
test_r2 <- 1 - sum((test_pred - y_test)^2) / sum((y_test - mean(y_test))^2)

# Print the results
cat("Train RMSE:", train_rmse, "\n")
cat("Train R squared:", train_r2, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R squared:", test_r2, "\n")


xgb_booster <- xgb_model$finalModel  # Only works if you used caret::train(method = "xgbTree")
xgb_var_imp <- xgb.importance(model = xgb_booster)

xgb.plot.importance(xgb_var_imp[1:10, ],col="orange")



################## XGB with important variables ############

top_10_vars <- xgb_var_imp$Feature[1:10]

x_train_top10 <- x_train_matrix[, top_10_vars]
x_test_top10 <- x_test_matrix[, top_10_vars]  # assuming you already have this matrix


dtrain_top10 <- xgb.DMatrix(data = x_train_top10, label = y_train)
dtest_top10 <- xgb.DMatrix(data = x_test_top10, label = y_test)

xgb_model_top10 <- xgboost(
  data = dtrain_top10,
  objective = "reg:squarederror",
  nrounds = 100,
  verbose = 0
)

# Make predictions on training and testing data
train_pred <- predict(xgb_model_top10, dtrain_top10)
test_pred <- predict(xgb_model_top10, dtest_top10)

# Calculate RMSE and R-squared for training data
train_rmse <- sqrt(mean((train_pred - y_train)^2))
train_r2 <- 1 - sum((train_pred - y_train)^2) / sum((y_train - mean(y_train)^2)
                                                    
# Calculate RMSE and R-squared for testing 
test_rmse <- sqrt(mean((test_pred - y_test)^2))
test_r2 <- 1 - sum((test_pred - y_test)^2) / sum((y_test - mean(y_test))^2)
                                                    
# Print the results
cat("Train RMSE:", train_rmse, "\n")
cat("Train R squared:", train_r2, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Test R squared:", test_r2, "\n")
                                                    
 ######################### LssoR #################################



# Prepare data
x <- model.matrix(RiskScore ~ ., train_data)[, -1]  
y <- train_data$RiskScore

# Fit Lasso with CV to find best lambda
lasso_cv <- cv.glmnet(x, y, alpha = 1)  # alpha = 1 for Lasso
plot(lasso_cv)

# Best lambda
best_lambda <- lasso_cv$lambda.min
cat("Best Lambda:", best_lambda, "\n")

# Final Lasso model
lasso_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)

# Coefficients
coef(lasso_model)

# Prepare test data
x_test <- model.matrix(RiskScore ~ ., test_data)[, -1]
y_test <- test_data$RiskScore

# Predict
pred_train <- predict(lasso_model, newx = x)
pred_test <- predict(lasso_model, newx = x_test)

# Evaluation
library(Metrics)
cat("Train RMSE:", rmse(y, pred_train), " | R²:", R2(pred_train, y), "\n")
cat("Test RMSE :", rmse(y_test, pred_test), " | R²:", R2(pred_test, y_test), "\n")



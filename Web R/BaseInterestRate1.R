# --- Load necessary packages ---
if (!require(caret)) install.packages("caret")
if (!require(xgboost)) install.packages("xgboost")
if (!require(Matrix)) install.packages("Matrix")
library(caret)
library(xgboost)
library(Matrix)

# --- Set the formula ---
formula_xgb <- BaseInterestRate ~ Age + AnnualIncome + EmploymentStatus + EducationLevel + Experience +
  LoanAmount + LoanDuration + MaritalStatus + NumberOfDependents + HomeOwnershipStatus +
  LoanPurpose + SavingsAccountBalance + CheckingAccountBalance + MonthlyIncome + JobTenure +
  TotalAssets + TotalLiabilities + NetWorth

# --- Train control (5-fold CV) ---
set.seed(123)
ctrl_xgb <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

# --- Define tuning grid for XGBoost ---
tune_grid_xgb <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.05, 0.1),
  gamma = 0,
  colsample_bytree = c(0.6, 0.8),
  min_child_weight = 1,
  subsample = c(0.7, 1)
)

# --- Train the XGBoost model ---
xgb_model <- train(
  formula_xgb,
  data = Loan_train,
  method = "xgbTree",
  trControl = ctrl_xgb,
  tuneGrid = tune_grid_xgb,
  verbosity = 0
)

# --- Print best parameters and model summary ---
print(xgb_model$bestTune)
print(xgb_model)

# --- Predict on training set ---
xgb_pred_train <- predict(xgb_model, newdata = Loan_train)
actual_train <- Loan_train$BaseInterestRate
r_squared_train_xgb <- 1 - sum((actual_train - xgb_pred_train)^2) / sum((actual_train - mean(actual_train))^2)

# --- Predict on test set ---
xgb_pred_test <- predict(xgb_model, newdata = Loan_test)
actual_test <- Loan_test$BaseInterestRate
r_squared_test_xgb <- 1 - sum((actual_test - xgb_pred_test)^2) / sum((actual_test - mean(actual_test))^2)

# --- Report R² ---
cat("Tuned XGBoost Training R²:", round(r_squared_train_xgb, 4), "\n")
cat("Tuned XGBoost Test R²:", round(r_squared_test_xgb, 4), "\n")

# --- Variable importance plot ---
xgb_imp <- varImp(xgb_model)
plot(xgb_imp, top = 20, main = "XGBoost - Top 20 Important Features")


# --- Extract matrix and label data (used internally by caret) ---
dtrain <- xgb.DMatrix(data = model.matrix(formula_xgb, Loan_train)[, -1],
                      label = Loan_train$BaseInterestRate)

# --- Train a final XGBoost model using best tuned params for importance ---
best_params <- xgb_model$bestTune

xgb_final <- xgboost(
  data = dtrain,
  nrounds = best_params$nrounds,
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  gamma = best_params$gamma,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  subsample = best_params$subsample,
  objective = "reg:squarederror",
  verbose = 0
)

# --- Get feature names from model matrix ---
feature_names <- colnames(model.matrix(formula_xgb, Loan_train))[-1]

# --- Extract variable importance using xgboost ---
importance_matrix <- xgb.importance(feature_names = feature_names, model = xgb_final)

# --- Compute importance percentages ---
importance_matrix$ImportancePercent <- round(100 * importance_matrix$Gain / sum(importance_matrix$Gain), 2)

# --- Display top 10 important features with percentages ---
top_10 <- head(importance_matrix[, c("Feature", "ImportancePercent")], 10)
print(top_10)


library(xgboost)
library(Metrics)

# --- Define formula for CreditScore prediction ---
formula_xgb_credit_score <- CreditScore ~ BaseInterestRate + LoanAmount + Age + LoanDuration + 
  MonthlyLoanPayment + InterestRate

# --- Set up train control ---
ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

# --- Define XGBoost tuning grid (customizable) ---
grid <- expand.grid(
  nrounds = 150,
  max_depth = 4,
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.8
)

# --- Train the model ---
xgb_model_credit_score <- train(
  formula_xgb_credit_score,
  data = Loan_train,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = grid,
  verbosity = 0
)

# --- Predictions ---
pred_train <- predict(xgb_model_credit_score, newdata = Loan_train)
pred_test <- predict(xgb_model_credit_score, newdata = Loan_test)

# --- Actual values ---
actual_train <- Loan_train$CreditScore
actual_test <- Loan_test$CreditScore

# --- Evaluation ---
r2_train <- R2(pred_train, actual_train)
mae_train <- MAE(pred_train, actual_train)
rmse_train <- RMSE(pred_train, actual_train)

r2_test <- R2(pred_test, actual_test)
mae_test <- MAE(pred_test, actual_test)
rmse_test <- RMSE(pred_test, actual_test)

# --- Print metrics ---
cat("XGBoost Model for CreditScore:\n\n")

cat("Training Set:\n")
cat("  R²   =", round(r2_train, 4), "\n")
cat("  MAE  =", round(mae_train, 4), "\n")
cat("  RMSE =", round(rmse_train, 4), "\n\n")

cat("Test Set:\n")
cat("  R²   =", round(r2_test, 4), "\n")
cat("  MAE  =", round(mae_test, 4), "\n")
cat("  RMSE =", round(rmse_test, 4), "\n\n")

# --- Feature Importance ---
importance <- varImp(xgb_model_credit_score)$importance
importance$Feature <- rownames(importance)
importance <- importance[order(-importance$Overall), ]
top_10 <- head(importance, 10)
top_10$ImportancePercent <- round(100 * top_10$Overall / sum(importance$Overall), 2)

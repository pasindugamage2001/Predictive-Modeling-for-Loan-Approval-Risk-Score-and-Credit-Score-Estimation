# --- Load libraries ---
library(caret)
library(xgboost)
library(Matrix)

# --- Define formula ---
formula_xgb_payment <- MonthlyLoanPayment ~ BaseInterestRate + InterestRate + Age + AnnualIncome + 
  EmploymentStatus + EducationLevel + Experience + LoanAmount + LoanDuration + 
  MaritalStatus + NumberOfDependents + HomeOwnershipStatus + LoanPurpose + 
  SavingsAccountBalance + CheckingAccountBalance + MonthlyIncome + 
  JobTenure + TotalAssets + TotalLiabilities + NetWorth + 
  BankruptcyHistory + PreviousLoanDefaults

# --- Train control ---
set.seed(123)
ctrl <- trainControl(method = "cv", number = 5)

# --- Hyperparameter grid ---
grid <- expand.grid(
  nrounds = 150,
  max_depth = 4,
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

# --- Fit model ---
xgb_model_payment <- train(
  formula_xgb_payment,
  data = Loan_train,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = grid,
  verbosity = 0
)

# --- Predictions ---
pred_train <- predict(xgb_model_payment, newdata = Loan_train)
pred_test <- predict(xgb_model_payment, newdata = Loan_test)

# --- Actual values ---
actual_train <- Loan_train$MonthlyLoanPayment
actual_test <- Loan_test$MonthlyLoanPayment

# --- Metrics ---
r2_train <- R2(pred_train, actual_train)
mae_train <- MAE(pred_train, actual_train)
rmse_train <- RMSE(pred_train, actual_train)

r2_test <- R2(pred_test, actual_test)
mae_test <- MAE(pred_test, actual_test)
rmse_test <- RMSE(pred_test, actual_test)

# --- Output metrics ---
cat("XGBoost Performance (MonthlyLoanPayment):\n\n")
cat("Training Set:\n")
cat("  R²   =", round(r2_train, 4), "\n")
cat("  MAE  =", round(mae_train, 4), "\n")
cat("  RMSE =", round(rmse_train, 4), "\n\n")

cat("Test Set:\n")
cat("  R²   =", round(r2_test, 4), "\n")
cat("  MAE  =", round(mae_test, 4), "\n")
cat("  RMSE =", round(rmse_test, 4), "\n\n")

# --- Variable importance ---
importance <- varImp(xgb_model_payment)$importance
importance$Feature <- rownames(importance)
importance <- importance[order(-importance$Overall), ]
top_10 <- head(importance, 10)

# --- Importance as percentage ---
top_10$ImportancePercent <- round(100 * top_10$Overall / sum(importance$Overall), 2)

# --- Display top 10 features ---
cat("Top 10 Important Features for MonthlyLoanPayment (%):\n")
print(top_10[, c("Feature", "ImportancePercent")])


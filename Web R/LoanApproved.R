library(xgboost)
library(Metrics)
library(caret)
library(pROC)

# --- Define formula ---
formula_xgb_loan_approved <- LoanApproved ~ RiskScore + TotalDebtToIncomeRatio + 
  InterestRate + MonthlyIncome + CreditScore

# --- Train control ---
ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE, classProbs = TRUE, 
                     summaryFunction = twoClassSummary)

# --- Tuning grid ---
grid <- expand.grid(
  nrounds = 150,
  max_depth = 6,
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.8
)

# --- Recode LoanApproved as factor with labels (for ROC etc.) ---
Loan_train$LoanApproved <- factor(ifelse(Loan_train$LoanApproved == 1, "Yes", "No"))
Loan_test$LoanApproved <- factor(ifelse(Loan_test$LoanApproved == 1, "Yes", "No"))

# --- Train model ---
xgb_model <- train(
  formula_xgb_loan_approved,
  data = Loan_train,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = grid,
  metric = "ROC",
  verbosity = 0
)

# --- Predict class and probabilities ---
pred_train_class <- predict(xgb_model, newdata = Loan_train)
pred_test_class <- predict(xgb_model, newdata = Loan_test)

pred_train_prob <- predict(xgb_model, newdata = Loan_train, type = "prob")[, "Yes"]
pred_test_prob <- predict(xgb_model, newdata = Loan_test, type = "prob")[, "Yes"]

# --- Confusion Matrices ---
cat("\nConfusion Matrix - Train:\n")
print(conf_train <- confusionMatrix(pred_train_class, Loan_train$LoanApproved))

cat("\nConfusion Matrix - Test:\n")
print(conf_test <- confusionMatrix(pred_test_class, Loan_test$LoanApproved))

# --- AUC ---
auc_train <- auc(Loan_train$LoanApproved, pred_train_prob)
auc_test <- auc(Loan_test$LoanApproved, pred_test_prob)

# --- LogLoss ---
logloss_train <- logLoss(ifelse(Loan_train$LoanApproved == "Yes", 1, 0), pred_train_prob)
logloss_test <- logLoss(ifelse(Loan_test$LoanApproved == "Yes", 1, 0), pred_test_prob)

# --- Print evaluation summary ---
cat("\nModel Evaluation Summary:\n")
cat("\nTraining Set:\n")
cat("  Accuracy   =", round(conf_train$overall["Accuracy"], 4), "\n")
cat("  AUC        =", round(auc_train, 4), "\n")
cat("  LogLoss    =", round(logloss_train, 4), "\n")

cat("\nTest Set:\n")
cat("  Accuracy   =", round(conf_test$overall["Accuracy"], 4), "\n")
cat("  AUC        =", round(auc_test, 4), "\n")
cat("  LogLoss    =", round(logloss_test, 4), "\n")

# --- Feature Importance ---
importance <- varImp(xgb_model)$importance
importance$Feature <- rownames(importance)
importance <- importance[order(-importance$Overall), ]
top_10 <- head(importance, 10)
top_10$ImportancePercent <- round(100 * top_10$Overall / sum(importance$Overall), 2)

# --- Show Top 10 Important Features ---
cat("\nTop 10 Important Features for LoanApproval Prediction (%):\n")
print(top_10[, c("Feature", "ImportancePercent")])

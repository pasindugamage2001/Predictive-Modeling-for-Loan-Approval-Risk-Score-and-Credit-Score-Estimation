ls()
rm(list = ls())
getwd()
setwd("E:/3rd year/sem2/ST 3082/Final project")
# Load the data set
data=read.csv(file="Loan.csv",header = TRUE,sep = ",")


# load necessary libraries 
library(dplyr)
library(ROSE)
library(pROC)
library(ggplot2)
library(caret)
library(gbm)
library(randomForest)
library(xgboost)
data <- data %>%
  select(-c(ApplicationDate, AnnualIncome, DebtToIncomeRatio,EducationLevel,Experience,PaymentHistory,
            UtilityBillsPaymentHistory,JobTenure,SavingsAccountBalance,CheckingAccountBalance,TotalAssets))  # Remove LoanApproved column

df_for_logit <- data
str(data)


nominal_vars <- c("EmploymentStatus","MaritalStatus","HomeOwnershipStatus")
df_for_logit[nominal_vars] <- lapply(df_for_logit[nominal_vars], as.factor)

# Ratio variables
ratio_vars <- c("Age","CreditScore","LoanAmount","LoanDuration","NumberOfDependents","NumberOfOpenCreditLines","NumberOfCreditInquiries",
                "BankruptcyHistory","PreviousLoanDefaults","LengthOfCreditHistory","MonthlyDebtPayments",
                "TotalLiabilities","NetWorth","LoanApproved")
df_for_logit[ratio_vars] <- lapply(df_for_logit[ratio_vars], as.numeric)

#setdiff(ratio_vars, names(df_for_logit))

# target
target <- df_for_logit$LoanApproved
df_for_logit$LoanApproved <- NULL


# Dummy encoding
dummies <- dummyVars(" ~ .", data = df_for_logit)

# Apply to full dataset to ensure all levels are present
df_encoded <- as.data.frame(predict(dummies, newdata = df_for_logit))
df_encoded$LoanApproved <- target

# Split
set.seed(123)
train_index <- createDataPartition(df_encoded$LoanApproved, p = 0.8, list = FALSE)
train_data <- df_encoded[train_index, ]
test_data  <- df_encoded[-train_index, ]

# Ensure column names match
train_data <- train_data %>% mutate(across(everything(), identity))
test_data  <- test_data %>% mutate(across(everything(), identity))
colnames(train_data) <- make.names(colnames(train_data))
colnames(test_data)  <- make.names(colnames(test_data))


sum(is.na(df_encoded)) 


train_data <- ROSE::ovun.sample(LoanApproved ~ ., data = train_data, method = "both",N = nrow(train_data))$data
table(train_data$LoanApproved)



# -----------------------------
# STEP 2: Full Model (All Variables)
# -----------------------------
# Fit GBM model on all variables
gbm_model <- gbm(
  LoanApproved ~ .,
  data = train_data,
  distribution = "bernoulli",
  n.trees = 100,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  n.minobsinnode = 10,
  verbose = FALSE
)


# Determine best number of trees
best_iter <- gbm.perf(gbm_model, method = "cv")

# Show variable importance
summary(gbm_model, n.trees = best_iter)

# Predict on test set
pred_probs <- predict(gbm_model, newdata = test_data, n.trees = best_iter, type = "response")
pred_class <- ifelse(pred_probs > 0.5, 1, 0)

# Evaluate model
cm_test=confusionMatrix(factor(pred_class, levels = c(0, 1)), factor(test_data$LoanApproved, levels = c(0, 1)))

# Extract precision, recall, and calculate F1 score
precision <- cm_test$byClass["Pos Pred Value"]
recall <- cm_test$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the F1 score
print(f1_score)

# Predict on train set
pred_probs <- predict(gbm_model, newdata = train_data, n.trees = best_iter, type = "response")
pred_class <- ifelse(pred_probs > 0.5, 1, 0)

# Evaluate model
cm_train=confusionMatrix(factor(pred_class, levels = c(0, 1)), factor(train_data$LoanApproved, levels = c(0, 1)))

# Extract precision, recall, and calculate F1 score
precision <- cm_train$byClass["Pos Pred Value"]
recall <- cm_train$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the F1 score
print(f1_score)
# -----------------------------
# STEP 3: Reduced Model (Important Variables Only)
# -----------------------------
varImp(gbm_model)
# Select top variables (based on importance)
important_vars <- c("RiskScore", "TotalDebtToIncomeRatio", "MonthlyIncome","InterestRate", "LoanDuration","LoanApproved")

# Reorder factors for plotting
important_vars <- factor(important_vars, levels = [order(top10_vars$Importance)])

# Create barplot
ggplot(important_vars, aes(x = Variable, y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Top 10 Important Variables", x = "Variables", y = "Importance") +
  theme_minimal()
# Create reduced train and test sets
train_top <- train_data[, important_vars]
test_top  <- test_data[, important_vars[1:4]]  # exclude target for prediction

# Fit reduced GBM model
gbm_model_top <- gbm(
  LoanApproved ~ .,
  data = train_top,
  distribution = "bernoulli",
  n.trees = 100,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  n.minobsinnode = 10,
  verbose = FALSE
)

# Best number of trees
best_iter_top <- gbm.perf(gbm_model_top, method = "cv")

# Predict with reduced model
pred_probs_top <- predict(gbm_model_top, newdata = train_top, n.trees = best_iter_top, type = "response")
pred_class_top <- ifelse(pred_probs_top > 0.5, 1, 0)

# Evaluate reduced model
cm_train=confusionMatrix(factor(pred_class_top, levels = c(0, 1)), factor(train_data$LoanApproved, levels = c(0, 1)))

# Extract precision, recall, and calculate F1 score
precision <- cm_train$byClass["Pos Pred Value"]
recall <- cm_train$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the F1 score
print(f1_score)


# Predict with reduced model
pred_probs_top <- predict(gbm_model_top, newdata = test_top, n.trees = best_iter_top, type = "response")
pred_class_top <- ifelse(pred_probs_top > 0.5, 1, 0)

# Evaluate reduced model
cm_test=confusionMatrix(factor(pred_class_top, levels = c(0, 1)), factor(test_data$LoanApproved, levels = c(0, 1)))

# Extract precision, recall, and calculate F1 score
precision <- cm_test$byClass["Pos Pred Value"]
recall <- cm_test$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the F1 score
print(f1_score)




####################################################################################################

# Convert LoanApproved to numeric 0/1
train_data$LoanApproved <- as.numeric(as.factor(train_data$LoanApproved)) - 1
test_data$LoanApproved  <- as.numeric(as.factor(test_data$LoanApproved)) - 1

# Remove rows with missing values
train_data <- na.omit(train_data)
test_data  <- na.omit(test_data)

# Convert character columns to factors
train_data[] <- lapply(train_data, function(x) if (is.character(x)) as.factor(x) else x)
test_data[]  <- lapply(test_data, function(x) if (is.character(x)) as.factor(x) else x)

# Ensure matching factor levels between train and test
factor_vars <- names(train_data)[sapply(train_data, is.factor)]
for (var in factor_vars) {
  if (var %in% names(test_data)) {
    test_data[[var]] <- factor(test_data[[var]], levels = levels(train_data[[var]]))
  }
}

# -----------------------------
# STEP 2: Full Model (All Variables)
# -----------------------------

# Ensure LoanApproved is a factor (for classification)
train_data$LoanApproved <- as.factor(train_data$LoanApproved)
test_data$LoanApproved  <- as.factor(test_data$LoanApproved)

# Set up cross-validation using 10-fold
train_control <- trainControl(
  method = "cv",       # Cross-validation
  number = 10,         # 10-fold
  search = "grid"      # Search method for tuning parameters
)

# Set up a grid for tuning parameters (optional, but can be useful for tuning `mtry` or `ntree`)
tune_grid <- expand.grid(
  mtry = sqrt(ncol(train_data) - 1)  # Default value for classification
)

# Fit Random Forest model using caret with cross-validation
rf_cv_model <- train(
  LoanApproved ~ .,        # Formula for classification
  data = train_data,       # Training data
  method = "rf",           # Random Forest model
  trControl = train_control, # Cross-validation setup
  tuneGrid = tune_grid,    # Tuning grid (optional)
  ntree = 50,             # Number of trees in the forest
  importance = TRUE# Get feature importance
)

# View the results of cross-validation
print(rf_cv_model)

# Best model performance
cat("Best Accuracy:", rf_cv_model$results$Accuracy[which.max(rf_cv_model$results$Accuracy)], "\n")


# Best random forest model after cross-validation
best_rf_model <- rf_cv_model$finalModel

final_rf_model <- randomForest(LoanApproved ~ ., data = train_data, ntree=50,mtry=6)  # Adjust the weights as needed


#Predict on train set
train_pred_class <- predict(final_rf_model, newdata = train_data)
train_pred_class <- as.numeric(train_pred_class) - 1  # Convert to 0/1

# Predict on test set
test_pred_class <- predict(final_rf_model, newdata = test_data)
test_pred_class <- as.numeric(test_pred_class) - 1  # Convert to 0/1

# -----------------------------
# STEP 4: Evaluate Model
# -----------------------------

# Training Set Evaluation
train_cm <- confusionMatrix(factor(train_pred_class, levels = c(0, 1)), factor(train_data$LoanApproved, levels = c(0, 1)))
train_f1_score <- 2 * (train_cm$byClass["Pos Pred Value"] * train_cm$byClass["Sensitivity"]) / 
  (train_cm$byClass["Pos Pred Value"] + train_cm$byClass["Sensitivity"])

# Test Set Evaluation
test_cm <- confusionMatrix(factor(test_pred_class, levels = c(0, 1)), factor(test_data$LoanApproved, levels = c(0, 1)))
test_f1_score <- 2 * (test_cm$byClass["Pos Pred Value"] * test_cm$byClass["Sensitivity"]) / 
  (test_cm$byClass["Pos Pred Value"] + test_cm$byClass["Sensitivity"])

# Print results
cat("Training Set Classification Metrics:\n")
print(train_cm)
cat("\nTraining Set F1 Score: ", train_f1_score, "\n\n")

cat("Test Set Classification Metrics:\n")
print(test_cm)
cat("\nTest Set F1 Score: ", test_f1_score, "\n")

varImp(final_rf_model)


# Get variable importance from the model
importance_df <- importance(final_rf_model)
importance_df <- data.frame(Variable = rownames(importance_df), Importance = importance_df[, 1])

# Sort and select top 10 variables
top10_vars <- importance_df[order(importance_df$Importance, decreasing = TRUE), ][1:5, ]

# Reorder factors for plotting
top10_vars$Variable <- factor(top10_vars$Variable, levels = top10_vars$Variable[order(top10_vars$Importance)])

# Create barplot
ggplot(top10_vars, aes(x = Variable, y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = " Important Variables", x = "Variables", y = "Importance") +
  theme_minimal()
# -----------------------------
# STEP 3: Reduced Model (Important Variables Only)
# -----------------------------


# -----------------------------
# STEP 1: Data Preparation
# -----------------------------

# Subset the dataset to include only important variables
important_vars <- c("RiskScore", "TotalDebtToIncomeRatio", "MonthlyIncome", 
                    "InterestRate","LoanApproved")

train_Data_reduced <- train_data[, important_vars]
test_Data_reduced  <- test_data[, important_vars]


# Convert LoanApproved to numeric (0/1)
train_Data_reduced$LoanApproved <- as.numeric(as.factor(train_Data_reduced$LoanApproved)) - 1
test_Data_reduced$LoanApproved  <- as.numeric(as.factor(test_Data_reduced$LoanApproved)) - 1

# -----------------------------
# STEP 2: Fit Random Forest Model
# -----------------------------

# Ensure LoanApproved is a factor (for classification)
train_Data_reduced$LoanApproved <- as.factor(train_Data_reduced$LoanApproved)
test_Data_reduced$LoanApproved  <- as.factor(test_Data_reduced$LoanApproved)

# Fit Random Forest model on all variables
rf_model_reduced <- randomForest(
  LoanApproved ~ .,
  data = train_Data_reduced,
  ntree = 100,  # Number of trees
  mtry = sqrt(ncol(train_Data_reduced) - 1),  # Default value for classification
  importance = TRUE
)




rf_model_reduced <- randomForest(LoanApproved ~ ., data = train_Data_reduced, ntree=50)  # Adjust the weights as needed

# -----------------------------
# STEP 3: Predict on Train Set and Test Set
# -----------------------------

# Predict on train set
train_pred_class <- predict(rf_model_reduced, newdata = train_Data_reduced)
train_pred_class <- as.numeric(train_pred_class) - 1  # Convert to 0/1

# Predict on test set
test_pred_class <- predict(rf_model_reduced, newdata = test_Data_reduced)
test_pred_class <- as.numeric(test_pred_class) - 1  # Convert to 0/1

# -----------------------------
# STEP 4: Evaluate Model
# -----------------------------

# Training Set Evaluation
train_cm <- confusionMatrix(factor(train_pred_class, levels = c(0, 1)), factor(train_Data_reduced$LoanApproved, levels = c(0, 1)))
train_f1_score <- 2 * (train_cm$byClass["Pos Pred Value"] * train_cm$byClass["Sensitivity"]) / 
  (train_cm$byClass["Pos Pred Value"] + train_cm$byClass["Sensitivity"])

# Test Set Evaluation
test_cm <- confusionMatrix(factor(test_pred_class, levels = c(0, 1)), factor(test_Data_reduced$LoanApproved, levels = c(0, 1)))
test_f1_score <- 2 * (test_cm$byClass["Pos Pred Value"] * test_cm$byClass["Sensitivity"]) / 
  (test_cm$byClass["Pos Pred Value"] + test_cm$byClass["Sensitivity"])

# Print results
cat("Training Set Classification Metrics:\n")
print(train_cm)
cat("\nTraining Set F1 Score: ", train_f1_score, "\n\n")

cat("Test Set Classification Metrics:\n")
print(test_cm)
cat("\nTest Set F1 Score: ", test_f1_score, "\n")


##################################################################################################
# XGBoost

# Convert LoanApproved to binary numeric (0/1)
train_data$LoanApproved <- as.numeric(as.factor(train_data$LoanApproved)) - 1
test_data$LoanApproved  <- as.numeric(as.factor(test_data$LoanApproved)) - 1

# Convert to matrix (XGBoost requirement)
x_train <- as.matrix(train_data %>% select(-LoanApproved))
y_train <- train_data$LoanApproved

x_test <- as.matrix(test_data %>% select(-LoanApproved))
y_test <- test_data$LoanApproved

# Convert factors to numeric via model.matrix (automatically one-hot encodes)
x_train <- model.matrix(LoanApproved ~ . - 1, data = train_data)
x_test  <- model.matrix(LoanApproved ~ . - 1, data = test_data)

# Target variable
y_train <- train_data$LoanApproved
y_test  <- test_data$LoanApproved

# -----------------------------
# STEP 2: Train XGBoost Model
# -----------------------------

best_xgb_model <- xgboost(
  data = x_train,
  label = y_train,
  nrounds = 100,
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 3,
  eta = 0.1,
  verbose = 0
)
library(caret)
library(xgboost)
# Ensure LoanApproved is a factor (for binary classification)
y_train <- as.factor(y_train)  # Convert y_train to factor
y_test <- as.factor(y_test)    # Similarly, ensure y_test is a factor

# Set up a grid for hyperparameters
param_grid <- expand.grid(
  nrounds = c(50, 100),          # Number of boosting rounds
  max_depth = c(3, 6),            # Maximum depth of the trees
  eta = c(0.01, 0.1),
  gamma = c( 0.1, 0.3),             # Gamma (minimum loss reduction)
  min_child_weight = c(1, 2),       # Minimum sum of hessian in child nodes# Learning rate
  subsample = c(0.6, 0.8),         # Fraction of training data for each round
  colsample_bytree = c(0.6, 0.8)   # Fraction of features for each tree
)

# Set up training control for 5-fold cross-validation
train_control <- trainControl(
  method = "cv",                       # Cross-validation method
  number = 5,                           # Number of folds
  verboseIter = TRUE                    # Show progress
)

# Train the xgboost model with tuning
xgb_tune_model <- train(
  x = x_train,                         # Training data
  y = y_train,                         # Target variable
  method = "xgbTree",                  # Use xgboost
  trControl = train_control,           # Set cross-validation
  tuneGrid = param_grid,               # Grid of hyperparameters
  metric = "Accuracy",                 # Optimization metric
  objective = "binary:logistic",       # Binary classification
  eval_metric = "logloss",             # Evaluation metric
  verbose = 0                           # Silence verbose output
)

# Check the results of the grid search
print(xgb_tune_model)

best_xgb_model <- xgboost(
  data = x_train,
  label = y_train,
  nrounds = 100,
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 3,
  eta = 0.1,
  verbose = 0
)



# -----------------------------
# STEP 3: Predict and Classify
# -----------------------------

# Predict probabilities
train_preds_prob <- predict(best_xgb_model, x_train)
test_preds_prob  <- predict(best_xgb_model, x_test)

# Classify using threshold 0.5
train_preds <- ifelse(train_preds_prob > 0.5, 1, 0)
test_preds  <- ifelse(test_preds_prob > 0.5, 1, 0)

# -----------------------------
# STEP 4: Evaluation
# -----------------------------

# Training Set Evaluation
train_cm <- confusionMatrix(factor(train_preds, levels = c(0, 1)), factor(y_train, levels = c(0, 1)))
train_precision <- train_cm$byClass["Pos Pred Value"]
train_recall <- train_cm$byClass["Sensitivity"]
train_f1 <- 2 * (train_precision * train_recall) / (train_precision + train_recall)

# Test Set Evaluation
test_cm <- confusionMatrix(factor(test_preds, levels = c(0, 1)), factor(y_test, levels = c(0, 1)))
test_precision <- test_cm$byClass["Pos Pred Value"]
test_recall <- test_cm$byClass["Sensitivity"]
test_f1 <- 2 * (test_precision * test_recall) / (test_precision + test_recall)

# -----------------------------
# STEP 5: Print Results
# -----------------------------

cat("===== Training Set =====\n")
print(train_cm)
cat("\nF1 Score (Train):", round(train_f1, 4), "\n\n")

cat("===== Test Set =====\n")
print(test_cm)

cat("\nF1 Score (Test):", round(test_f1, 4), "\n")


# Get feature importance
importance_matrix <- xgb.importance(model = best_xgb_model)

# View as a table
print(importance_matrix)
par(mfrow=c(1,1))
# Plot variable importance
xgb.plot.importance(importance_matrix, top_n = 5, measure = "Gain")

# -----------------------------
# STEP 1: Data Preparation
# -----------------------------

# Define important variables
important_vars <- c("RiskScore", "TotalDebtToIncomeRatio", "MonthlyIncome",  
                    "InterestRate", "CreditScore", "LoanApproved")

train_Data_reduced <- train_data[, important_vars]
test_Data_reduced  <- test_data[, important_vars]

# Convert LoanApproved to binary numeric (0/1)
train_Data_reduced$LoanApproved <- as.numeric(as.factor(train_Data_reduced$LoanApproved)) - 1
test_Data_reduced$LoanApproved  <- as.numeric(as.factor(test_Data_reduced$LoanApproved)) - 1

# Convert to matrix (XGBoost requirement)
x_train <- as.matrix(train_Data_reduced %>% select(-LoanApproved))
y_train <- train_Data_reduced$LoanApproved

x_test <- as.matrix(test_Data_reduced %>% select(-LoanApproved))
y_test <- test_Data_reduced$LoanApproved

# -----------------------------
# STEP 2: Train XGBoost Model
# -----------------------------

xgb_model <- xgboost(
  data = x_train,
  label = y_train,
  nrounds = 100,
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 3,
  eta = 0.1,
  verbose = 0
)

# -----------------------------
# STEP 3: Predict and Classify
# -----------------------------

# Predict probabilities
train_preds_prob <- predict(xgb_model, x_train)
test_preds_prob  <- predict(xgb_model, x_test)

# Classify using threshold 0.5
train_preds <- ifelse(train_preds_prob > 0.5, 1, 0)
test_preds  <- ifelse(test_preds_prob > 0.5, 1, 0)

# -----------------------------
# STEP 4: Evaluation
# -----------------------------

# Training Set Evaluation
train_cm <- confusionMatrix(factor(train_preds, levels = c(0, 1)), factor(y_train, levels = c(0, 1)))
train_precision <- train_cm$byClass["Pos Pred Value"]
train_recall <- train_cm$byClass["Sensitivity"]
train_f1 <- 2 * (train_precision * train_recall) / (train_precision + train_recall)

# Test Set Evaluation
test_cm <- confusionMatrix(factor(test_preds, levels = c(0, 1)), factor(y_test, levels = c(0, 1)))
test_precision <- test_cm$byClass["Pos Pred Value"]
test_recall <- test_cm$byClass["Sensitivity"]
test_f1 <- 2 * (test_precision * test_recall) / (test_precision + test_recall)

# -----------------------------
# STEP 5: Print Results
# -----------------------------

cat("===== Training Set =====\n")
print(train_cm)
cat("\nF1 Score (Train):", round(train_f1, 4), "\n\n")

cat("===== Test Set =====\n")
print(test_cm)
cat("\nF1 Score (Test):", round(test_f1, 4), "\n")


#################################################################################################

library(class)
library(caret)
library(e1071)  # for confusionMatrix


# Copy original full train and test data
train_full <- train_data
test_full  <- test_data



# Remove non-numeric columns (KNN requires numeric features)
train_full <- train_full[, sapply(train_full, is.numeric)]
test_full  <- test_full[, sapply(test_full, is.numeric)]

# Normalize features (excluding target)
features <- setdiff(colnames(train_full), "LoanApproved")

normalize <- function(x) (x - min(x)) / (max(x) - min(x))
train_norm <- as.data.frame(lapply(train_full[, features], normalize))
test_norm  <- as.data.frame(lapply(test_full[, features], normalize))

# Add LoanApproved target
train_norm$LoanApproved <- train_full$LoanApproved
test_norm$LoanApproved  <- test_full$LoanApproved


# Separate features and labels
x_train <- train_norm[, features]
y_train <- factor(train_norm$LoanApproved, levels = c(0, 1))

x_test <- test_norm[, features]
y_test <- factor(test_norm$LoanApproved, levels = c(0, 1))

# Define a grid of k values to search
k_grid <- expand.grid(k = seq(1, 10, by = 2))  # Trying odd values for k

# Train control for cross-validation
train_control <- trainControl(
  method = "cv",        # Cross-validation
  number = 5,           # 5-fold cross-validation
  verboseIter = TRUE    # Show progress
)

# Train the KNN model with tuning for k
knn_tune_model <- train(
  x = x_train,             # Training data
  y = y_train,             # Target variable
  method = "knn",          # KNN method
  trControl = train_control,  # Cross-validation setup
  tuneGrid = k_grid,        # Grid of k values
  metric = "Accuracy"       # Optimize for accuracy
)

# Print the best tuning parameter (k)
print(knn_tune_model)

# Fit KNN with k = 5
knn_pred_test <- knn(train = x_train, test = x_test, cl = y_train, k = 3)
knn_pred_train <- knn(train = x_train, test = x_train, cl = y_train, k = 3)


cm_test <- confusionMatrix(knn_pred_test, y_test)

# F1 Score
precision <- cm_test$byClass["Pos Pred Value"]
recall <- cm_test$byClass["Sensitivity"]
f1_test <- 2 * (precision * recall) / (precision + recall)

cat(" Test Set F1 Score:", f1_test, "\n")
print(cm_test)


cm_train <- confusionMatrix(knn_pred_train, y_train)

# F1 Score
precision <- cm_train$byClass["Pos Pred Value"]
recall <- cm_train$byClass["Sensitivity"]
f1_train <- 2 * (precision * recall) / (precision + recall)

cat(" Training Set F1 Score:", f1_train, "\n")
print(cm_train)

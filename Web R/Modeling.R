# Install and load the readxl package if not already installed
if (!require(readxl)) install.packages("readxl", dependencies = TRUE)
library(readxl)

# Import the Excel file
Loan <- read_excel("C:/Users/USER/Desktop/Bsc (Hons) Statistics - University of Colombo/Statistics Hons/3rd year/Second sem/ST 3082 - Statistical Learning I/Final Project/Loan.xlsx")

# View the dataset
View(Loan)
# Load necessary libraries
if (!require(readxl)) install.packages("readxl", dependencies = TRUE)
if (!require(dplyr)) install.packages("dplyr", dependencies = TRUE)

library(readxl)
library(dplyr)

# Load dataset
Loan <- read_excel("C:/Users/USER/Desktop/Bsc (Hons) Statistics - University of Colombo/Statistics Hons/3rd year/Second sem/ST 3082 - Statistical Learning I/Final Project/Loan.xlsx")

# ---- Data Preprocessing ----

# 1. Check for missing values
missing_values <- colSums(is.na(Loan))
print(missing_values) # Print number of missing values per column

# Option: Impute missing values (example: mean for numerical, mode for categorical)
Loan <- Loan %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .))) %>%  # Impute with mean
  mutate(across(c(EmploymentStatus, EducationLevel, MaritalStatus, HomeOwnershipStatus, 
                  BankruptcyHistory, LoanPurpose, PreviousLoanDefaults, LoanApproved), 
                ~ ifelse(is.na(.), as.character(names(sort(table(.), decreasing = TRUE)[1])), .))) # Impute with mode

# 2. Remove duplicates
Loan <- Loan %>% distinct()

# 3. Convert categorical variables to factors
categorical_vars <- c("EmploymentStatus", "EducationLevel", "MaritalStatus", "HomeOwnershipStatus", 
                      "BankruptcyHistory", "LoanPurpose", "PreviousLoanDefaults", "LoanApproved")

Loan[categorical_vars] <- lapply(Loan[categorical_vars], as.factor)

# 4. Split dataset into Quantitative and Categorical variables
quantitative_vars <- Loan %>% select(-all_of(categorical_vars))  # Numeric variables only
categorical_vars_df <- Loan %>% select(all_of(categorical_vars)) # Categorical variables only

# 5. Print structure of the datasets
str(quantitative_vars)
str(categorical_vars_df)

# Save preprocessed data
write.csv(Loan, "C:/Users/USER/Desktop/Loan_Cleaned.csv", row.names = FALSE)

# Display final dataset
View(Loan)

# Load necessary library
if (!require(caret)) install.packages("caret", dependencies = TRUE)
library(caret)

# Set seed for reproducibility
set.seed(123)

# Create an 80-20 split
train_index <- createDataPartition(Loan$LoanApproved, p = 0.8, list = FALSE)

# Split data into training and testing sets
Loan_train <- Loan[train_index, ]
Loan_test <- Loan[-train_index, ]

# Print dimensions of the datasets
cat("Training Set Dimensions:", dim(Loan_train), "\n")
cat("Testing Set Dimensions:", dim(Loan_test), "\n")

# Save training and testing datasets
write.csv(Loan_train, "C:/Users/USER/Desktop/Loan_Train.csv", row.names = FALSE)
write.csv(Loan_test, "C:/Users/USER/Desktop/Loan_Test.csv", row.names = FALSE)



# --- Fit the simple linear regression model ---
model_simple <- lm(InterestRate ~ BaseInterestRate, data = Loan_train)

# --- Summary of the model ---
summary_model <- summary(model_simple)
r_squared_train <- summary_model$r.squared
adj_r_squared_train <- summary_model$adj.r.squared

cat("\n--- TRAINING SET EVALUATION ---\n")
cat("Training R-squared:", r_squared_train, "\n")
cat("Training Adjusted R-squared:", adj_r_squared_train, "\n")

# --- Predict on the test set ---
pred_test <- predict(model_simple, newdata = Loan_test)
actual_test <- Loan_test$InterestRate

# --- Calculate R-squared on the test set manually ---
ss_total_test <- sum((actual_test - mean(actual_test))^2)
ss_res_test <- sum((actual_test - pred_test)^2)
r_squared_test <- 1 - (ss_res_test / ss_total_test)

cat("\n--- TEST SET EVALUATION ---\n")
cat("Test R-squared:", r_squared_test, "\n")


# Plot: Actual vs Predicted InterestRate on Test Set
plot(actual_test, pred_test,
     main = "Actual vs Predicted Interest Rate (Test Set)",
     xlab = "Actual Interest Rate",
     ylab = "Predicted Interest Rate",
     col = "darkgreen", pch = 16)
abline(0, 1, col = "red", lwd = 2)



# --- Fit the model ---
model_simple <- lm(InterestRate ~ BaseInterestRate, data = Loan_train)

# --- Get training R² values ---
summary_model <- summary(model_simple)
r_squared_train <- summary_model$r.squared
adj_r_squared_train <- summary_model$adj.r.squared

# --- Predict on test set ---
pred_test <- predict(model_simple, newdata = Loan_test)
actual_test <- Loan_test$InterestRate

# --- Calculate R² for test set ---
ss_total_test <- sum((actual_test - mean(actual_test))^2)
ss_res_test <- sum((actual_test - pred_test)^2)
r_squared_test <- 1 - (ss_res_test / ss_total_test)

# --- Extract regression coefficients ---
intercept <- coef(model_simple)[1]
slope <- coef(model_simple)[2]
eqn <- paste0("InterestRate = ", round(intercept, 3), 
              " + ", round(slope, 3), " * BaseInterestRate")

# --- Create plot ---
plot(actual_test, pred_test,
     main = "Actual vs Predicted Interest Rate (Test Set)",
     xlab = "Actual Interest Rate",
     ylab = "Predicted Interest Rate",
     pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)

# --- Add equation and R² values to the plot ---
legend_text <- c(
  eqn,
  paste0("Train R² = ", round(r_squared_train, 4)),
  paste0("Test R² = ", round(r_squared_test, 4))
)

legend("topleft", legend = legend_text, bty = "n", text.col = "black")

# --- Fit model for BaseInterestRate using selected predictors ---

# Fit the model on the training set
model_base <- lm(BaseInterestRate ~ Age + AnnualIncome + EmploymentStatus + EducationLevel +
                   Experience + LoanAmount + LoanDuration + MaritalStatus + NumberOfDependents +
                   HomeOwnershipStatus + LoanPurpose + SavingsAccountBalance + CheckingAccountBalance +
                   MonthlyIncome + JobTenure + TotalAssets + TotalLiabilities + NetWorth, 
                 data = Loan_train)

# --- Training set predictions and R-squared ---
pred_train <- predict(model_base, newdata = Loan_train)
actual_train <- Loan_train$BaseInterestRate

# Calculate R-squared for training set
ss_total_train <- sum((actual_train - mean(actual_train))^2)
ss_res_train <- sum((actual_train - pred_train)^2)
r_squared_train <- 1 - (ss_res_train / ss_total_train)

# Adjusted R-squared
n_train <- nrow(Loan_train)
p <- length(model_base$coefficients) - 1
adj_r_squared_train <- 1 - ((1 - r_squared_train) * (n_train - 1) / (n_train - p - 1))

# --- Test set predictions and R-squared ---
pred_test <- predict(model_base, newdata = Loan_test)
actual_test <- Loan_test$BaseInterestRate

ss_total_test <- sum((actual_test - mean(actual_test))^2)
ss_res_test <- sum((actual_test - pred_test)^2)
r_squared_test <- 1 - (ss_res_test / ss_total_test)

# --- Plot: Actual vs Predicted for Test Set ---
library(ggplot2)

ggplot(data = NULL, aes(x = actual_test, y = pred_test)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE, color = "darkred") +
  labs(title = "Predicted vs Actual: BaseInterestRate (Test Set)",
       x = "Actual BaseInterestRate",
       y = "Predicted BaseInterestRate") +
  annotate("text", x = min(actual_test), y = max(pred_test),
           label = paste("Train R² =", round(r_squared_train, 4), 
                         "\nTest R² =", round(r_squared_test, 4)),
           hjust = 0, vjust = 1.5, size = 5, color = "black")

# Load necessary packages
if (!require(randomForest)) install.packages("randomForest")
library(randomForest)

# --- Define formula ---
formula_rf <- BaseInterestRate ~ Age + AnnualIncome + EmploymentStatus + EducationLevel + Experience +
  LoanAmount + LoanDuration + MaritalStatus + NumberOfDependents + HomeOwnershipStatus +
  LoanPurpose + SavingsAccountBalance + CheckingAccountBalance + MonthlyIncome + JobTenure +
  TotalAssets + TotalLiabilities + NetWorth

# --- Fit Random Forest model ---
set.seed(123)
rf_model <- randomForest(formula_rf, data = Loan_train, ntree = 500, importance = TRUE)

# --- Predict on Training Set ---
rf_pred_train <- predict(rf_model, newdata = Loan_train)
actual_train <- Loan_train$BaseInterestRate

# R-squared for Training
ss_total_train <- sum((actual_train - mean(actual_train))^2)
ss_res_train <- sum((actual_train - rf_pred_train)^2)
r_squared_train_rf <- 1 - (ss_res_train / ss_total_train)

# --- Predict on Test Set ---
rf_pred_test <- predict(rf_model, newdata = Loan_test)
actual_test <- Loan_test$BaseInterestRate

# R-squared for Test
ss_total_test <- sum((actual_test - mean(actual_test))^2)
ss_res_test <- sum((actual_test - rf_pred_test)^2)
r_squared_test_rf <- 1 - (ss_res_test / ss_total_test)

# --- Print Results ---
cat("Random Forest Training R²:", round(r_squared_train_rf, 4), "\n")
cat("Random Forest Test R²:", round(r_squared_test_rf, 4), "\n")

# Optional: Plot variable importance
importance(rf_model)
varImpPlot(rf_model)






# Load necessary packages
if (!require(caret)) install.packages("caret")
if (!require(randomForest)) install.packages("randomForest")
library(caret)
library(randomForest)

# Set formula
formula_rf <- BaseInterestRate ~ Age + AnnualIncome + EmploymentStatus + EducationLevel + Experience +
  LoanAmount + LoanDuration + MaritalStatus + NumberOfDependents + HomeOwnershipStatus +
  LoanPurpose + SavingsAccountBalance + CheckingAccountBalance + MonthlyIncome + JobTenure +
  TotalAssets + TotalLiabilities + NetWorth

# Set up training control with 5-fold CV
set.seed(123)
ctrl <- trainControl(method = "cv", number = 5)

# Grid of tuning parameters
tune_grid <- expand.grid(
  .mtry = c(2, 4, 6, 8, 10)  # Try different mtry values
)

# Train the tuned random forest model
rf_tuned <- train(
  formula_rf,
  data = Loan_train,
  method = "rf",
  trControl = ctrl,
  tuneGrid = tune_grid,
  ntree = 500,
  importance = TRUE
)

# Print best model and results
print(rf_tuned$bestTune)
print(rf_tuned)

# Predict on train
rf_pred_train <- predict(rf_tuned, newdata = Loan_train)
actual_train <- Loan_train$BaseInterestRate
r_squared_train_rf <- 1 - sum((actual_train - rf_pred_train)^2) / sum((actual_train - mean(actual_train))^2)

# Predict on test
rf_pred_test <- predict(rf_tuned, newdata = Loan_test)
actual_test <- Loan_test$BaseInterestRate
r_squared_test_rf <- 1 - sum((actual_test - rf_pred_test)^2) / sum((actual_test - mean(actual_test))^2)

# Report
cat("Tuned Random Forest Training R²:", round(r_squared_train_rf, 4), "\n")
cat("Tuned Random Forest Test R²:", round(r_squared_test_rf, 4), "\n")

# Plot variable importance
varImpPlot(rf_tuned$finalModel)



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


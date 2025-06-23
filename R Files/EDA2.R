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

# Load necessary libraries
library(car)      # For VIF
library(dplyr)    # For data manipulation
library(lmtest)   # For diagnostic tests
library(readxl)   # For reading Excel files

# Reload dataset if needed
Loan <- read_excel("C:/Users/USER/Desktop/Bsc (Hons) Statistics - University of Colombo/Statistics Hons/3rd year/Second sem/ST 3082 - Statistical Learning I/Final Project/Loan.xlsx")

# Convert categorical variables to factors
categorical_vars <- c("EmploymentStatus", "EducationLevel", "MaritalStatus", "HomeOwnershipStatus", 
                      "BankruptcyHistory", "LoanPurpose", "PreviousLoanDefaults", "LoanApproved")
Loan[categorical_vars] <- lapply(Loan[categorical_vars], as.factor)

# Fit MLR model predicting RiskScore
mlr_risk_model <- lm(RiskScore ~ Age + AnnualIncome + Experience + LoanAmount + LoanDuration + 
                       MonthlyDebtPayments + CreditCardUtilizationRate + NumberOfOpenCreditLines + 
                       NumberOfCreditInquiries + DebtToIncomeRatio + NumberOfDependents + 
                       SavingsAccountBalance + CheckingAccountBalance + TotalAssets + TotalLiabilities + 
                       MonthlyIncome + UtilityBillsPaymentHistory + JobTenure + NetWorth + 
                       BaseInterestRate + InterestRate + MonthlyLoanPayment + TotalDebtToIncomeRatio,
                     data = Loan)

# ---- Summary of the model ----
summary(mlr_risk_model)

# ---- Assumption Checks ----

# 1. Linearity - Residuals vs Fitted plot
plot(mlr_risk_model, which = 1)

# 2. Independence - Durbin-Watson test
dwtest(mlr_risk_model)

# 3. Homoscedasticity - Scale-Location plot
plot(mlr_risk_model, which = 3)

# 4. Homoscedasticity - Breusch-Pagan test
bptest(mlr_risk_model)

# 5. Normality - Q-Q plot
qqnorm(residuals(mlr_risk_model))
qqline(residuals(mlr_risk_model), col = "red")

# 6. Normality - Shapiro-Wilk test
shapiro.test(residuals(mlr_risk_model))

# 7. Multicollinearity - VIF
vif(mlr_risk_model)
# Load caret for evaluation metrics
if (!require(caret)) install.packages("caret", dependencies = TRUE)
library(caret)

# Refit the MLR model on training data
mlr_risk_model <- lm(RiskScore ~ Age + AnnualIncome + Experience + LoanAmount + LoanDuration + 
                       MonthlyDebtPayments + CreditCardUtilizationRate + NumberOfOpenCreditLines + 
                       NumberOfCreditInquiries + DebtToIncomeRatio + NumberOfDependents + 
                       SavingsAccountBalance + CheckingAccountBalance + TotalAssets + TotalLiabilities + 
                       MonthlyIncome + UtilityBillsPaymentHistory + JobTenure + NetWorth + 
                       BaseInterestRate + InterestRate + MonthlyLoanPayment + TotalDebtToIncomeRatio,
                     data = Loan_train)

# ---- Predictions ----
train_preds <- predict(mlr_risk_model, newdata = Loan_train)
test_preds <- predict(mlr_risk_model, newdata = Loan_test)

# ---- Evaluation on Training Set ----
train_metrics <- postResample(pred = train_preds, obs = Loan_train$RiskScore)
cat("Training Set Metrics:\n")
print(train_metrics)

# ---- Evaluation on Test Set ----
test_metrics <- postResample(pred = test_preds, obs = Loan_test$RiskScore)
cat("\nTest Set Metrics:\n")
print(test_metrics)

approval_by_purpose <- Loan %>%
  group_by(LoanPurpose) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

print(approval_by_purpose)


# Load required libraries
if (!require(dplyr)) install.packages("dplyr", dependencies = TRUE)
if (!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
library(dplyr)
library(ggplot2)

# Calculate approval rate by loan purpose, converting LoanApproved factor to numeric correctly:
approval_by_purpose <- Loan %>%
  group_by(LoanPurpose) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

print(approval_by_purpose)

# Create a bar graph for the approval rate by loan purpose
ggplot(approval_by_purpose, aes(x = LoanPurpose, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Loan Approval Rate by Loan Purpose",
       x = "Loan Purpose",
       y = "Approval Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()





# Load necessary libraries
if (!require(dplyr)) install.packages("dplyr", dependencies = TRUE)
if (!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
library(dplyr)
library(ggplot2)

# Calculate approval rate by EducationLevel
approval_by_education <- Loan %>%
  group_by(EducationLevel) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the computed summary
print(approval_by_education)

# Create the bar graph for EducationLevel approval rate
ggplot(approval_by_education, aes(x = EducationLevel, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  labs(title = "Loan Approval Rate by Education Level",
       x = "Education Level",
       y = "Approval Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()




# Load required libraries
if (!require(dplyr)) install.packages("dplyr", dependencies = TRUE)
if (!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
library(dplyr)
library(ggplot2)

# Calculate approval rate by EmploymentStatus, ensuring proper conversion of LoanApproved
employment_approval <- Loan %>%
  group_by(EmploymentStatus) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

print(employment_approval)

# Create the bar graph for Approval Rate by EmploymentStatus
ggplot(employment_approval, aes(x = EmploymentStatus, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Loan Approval Rate by Employment Status",
       x = "Employment Status",
       y = "Approval Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()





# Load required libraries
if (!require(dplyr)) install.packages("dplyr", dependencies = TRUE)
if (!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
library(dplyr)
library(ggplot2)

# Calculate approval rate by MaritalStatus
approval_by_marital <- Loan %>%
  group_by(MaritalStatus) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the computed summary
print(approval_by_marital)

# Create the bar graph for Approval Rate by MaritalStatus
ggplot(approval_by_marital, aes(x = MaritalStatus, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "tomato") +
  labs(title = "Loan Approval Rate by Marital Status",
       x = "Marital Status",
       y = "Approval Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()




# Calculate approval rate by NumberOfDependents
approval_by_dependents <- Loan %>%
  group_by(NumberOfDependents) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the computed summary
print(approval_by_dependents)

# Create the bar graph for Approval Rate by NumberOfDependents
ggplot(approval_by_dependents, aes(x = as.factor(NumberOfDependents), y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Loan Approval Rate by Number of Dependents",
       x = "Number of Dependents",
       y = "Approval Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()





# Load required libraries
if (!require(dplyr)) install.packages("dplyr", dependencies = TRUE)
if (!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
library(dplyr)
library(ggplot2)

# Calculate approval rate by HomeOwnershipStatus
approval_by_homeownership <- Loan %>%
  group_by(HomeOwnershipStatus) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the computed summary
print(approval_by_homeownership)

# Create the bar graph for Approval Rate by HomeOwnershipStatus
ggplot(approval_by_homeownership, aes(x = HomeOwnershipStatus, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Loan Approval Rate by Homeownership Status",
       x = "Homeownership Status",
       y = "Approval Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()




# Load required libraries
if (!require(dplyr)) install.packages("dplyr", dependencies = TRUE)
if (!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
library(dplyr)
library(ggplot2)

# Calculate approval rate by NumberOfCreditInquiries
approval_by_credit_inquiries <- Loan %>%
  group_by(NumberOfCreditInquiries) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the computed summary
print(approval_by_credit_inquiries)

# Create the bar graph for Approval Rate by NumberOfCreditInquiries
ggplot(approval_by_credit_inquiries, aes(x = NumberOfCreditInquiries, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "tomato") +
  labs(title = "Loan Approval Rate by Number of Credit Inquiries",
       x = "Number of Credit Inquiries",
       y = "Approval Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()




# Calculate approval rate by BankruptcyHistory
approval_by_bankruptcy <- Loan %>%
  group_by(BankruptcyHistory) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the computed summary
print(approval_by_bankruptcy)

# Create the bar graph for Approval Rate by BankruptcyHistory
ggplot(approval_by_bankruptcy, aes(x = BankruptcyHistory, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Loan Approval Rate by Bankruptcy History",
       x = "Bankruptcy History",
       y = "Approval Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()





# Load required libraries
library(dplyr)
library(ggplot2)

# Assuming your data is loaded into 'Loan'
# Calculate approval rate by LoanDuration
approval_by_duration <- Loan %>%
  group_by(LoanDuration) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the computed summary
print(approval_by_duration)

# Fit a linear regression model for Approval_Rate as a function of LoanDuration
model <- lm(Approval_Rate ~ LoanDuration, data = approval_by_duration)

# Get the R-squared value
r_squared <- summary(model)$r.squared
cat("R-squared value:", r_squared, "\n")

# Create the plot for Approval Rate by Loan Duration with a fitted regression line
ggplot(approval_by_duration, aes(x = LoanDuration, y = Approval_Rate)) +
  geom_point(color = "tomato", size = 3) +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(title = paste("Loan Approval Probability vs Loan Duration (R² = ", round(r_squared, 3), ")", sep = ""),
       x = "Loan Duration (months)",
       y = "Loan Approval Probability") +
  theme_minimal()



# Load necessary libraries
library(dplyr)
library(ggplot2)

# Assuming 'Loan' is your data frame
# Fit a linear model to estimate Loan Approval Probability vs Loan Amount
model <- lm(LoanApproved ~ LoanAmount, data = Loan)

# Create the plot for Loan Approval Probability vs Loan Amount
ggplot(Loan, aes(x = LoanAmount, y = LoanApproved)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(
    title = "Loan Approval Probability vs Loan Amount",
    x = "Loan Amount",
    y = "Loan Approval Probability"
  ) +
  annotate(
    "text", x = max(Loan$LoanAmount) * 0.7, y = 0.1,
    label = paste("R² =", round(summary(model)$r.squared, 4)),
    size = 5, color = "black"
  ) +
  theme_minimal()


# Load necessary libraries
library(dplyr)
library(ggplot2)

# Assuming 'Loan' is your data frame and BaseInterestRate is a variable in it
# Fit a linear model to estimate Loan Approval Probability vs Base Interest Rate
model_interest_rate <- lm(LoanApproved ~ BaseInterestRate, data = Loan)

# Create the plot for Loan Approval Probability vs Base Interest Rate
ggplot(Loan, aes(x = BaseInterestRate, y = LoanApproved)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(
    title = "Loan Approval Probability vs Base Interest Rate",
    x = "Base Interest Rate",
    y = "Loan Approval Probability"
  ) +
  annotate(
    "text", x = max(Loan$BaseInterestRate) * 0.7, y = 0.1,
    label = paste("R² =", round(summary(model_interest_rate)$r.squared, 4)),
    size = 5, color = "black"
  ) +
  theme_minimal()


# Load necessary libraries
library(dplyr)
library(ggplot2)

# Assuming 'Loan' is your data frame

# Log transform the Loan Amount and Base Interest Rate
Loan$log_LoanAmount <- log(Loan$LoanAmount + 1)  # Add 1 to avoid log(0)
Loan$log_BaseInterestRate <- log(Loan$BaseInterestRate + 1)  # Add 1 to avoid log(0)

# Fit a linear model for loan approval probability vs log-transformed loan amount
model_loan_amount <- lm(LoanApproved ~ log_LoanAmount, data = Loan)

# Fit a linear model for loan approval probability vs log-transformed base interest rate
model_interest_rate <- lm(LoanApproved ~ log_BaseInterestRate, data = Loan)

# Create the plot for Loan Approval Probability vs Log of Loan Amount
ggplot(Loan, aes(x = log_LoanAmount, y = LoanApproved)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(
    title = "Loan Approval Probability vs Log of Loan Amount",
    x = "Log of Loan Amount",
    y = "Loan Approval Probability"
  ) +
  annotate(
    "text", x = max(Loan$log_LoanAmount) * 0.7, y = 0.1,
    label = paste("R² =", round(summary(model_loan_amount)$r.squared, 4)),
    size = 5, color = "black"
  ) +
  theme_minimal()




# Create the plot for Loan Approval Probability vs Log of Base Interest Rate
ggplot(Loan, aes(x = log_BaseInterestRate, y = LoanApproved)) +
  geom_point(alpha = 0.5, color = "green") +
  geom_smooth(method = "lm", se = FALSE, color = "orange") +
  labs(
    title = "Loan Approval Probability vs Log of Base Interest Rate",
    x = "Log of Base Interest Rate",
    y = "Loan Approval Probability"
  ) +
  annotate(
    "text", x = max(Loan$log_BaseInterestRate) * 0.7, y = 0.1,
    label = paste("R² =", round(summary(model_interest_rate)$r.squared, 4)),
    size = 5, color = "black"
  ) +
  theme_minimal()





# Load necessary libraries
library(dplyr)
library(ggplot2)

# Assuming 'Loan' is your dataset
# Find the minimum and maximum LoanAmount
min_loan <- min(Loan$LoanAmount, na.rm = TRUE)
max_loan <- max(Loan$LoanAmount, na.rm = TRUE)

# Create class intervals for Loan Amount (e.g., 0-10, 10-20, ...)
bin_width <- 10  # Define the width of each class interval
breaks <- seq(min_loan, max_loan, by = bin_width)
Loan$LoanAmountBin <- cut(Loan$LoanAmount, breaks = breaks, include.lowest = TRUE, right = FALSE)

# Calculate frequency and mean Loan Approval Probability for each class interval
approval_by_bin <- Loan %>%
  group_by(LoanAmountBin) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the calculated class interval summary
print(approval_by_bin)

# Plot the mean Loan Approval Probability by class interval
ggplot(approval_by_bin, aes(x = LoanAmountBin, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "black") +
  labs(
    title = "Loan Approval Probability by Loan Amount",
    x = "Loan Amount (Class Intervals)",
    y = "Loan Approval Probability"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()








# Load necessary libraries
library(dplyr)
library(ggplot2)

# Assuming 'Loan' is your dataset
# Find the minimum and maximum BaseInterestRate
min_rate <- min(Loan$BaseInterestRate, na.rm = TRUE)
max_rate <- max(Loan$BaseInterestRate, na.rm = TRUE)

# Create class intervals for Base Interest Rate (e.g., 0-5, 5-10, ...)
bin_width <- 5  # Define the width of each class interval (percentage points)
breaks <- seq(min_rate, max_rate, by = bin_width)
Loan$BaseInterestRateBin <- cut(Loan$BaseInterestRate, breaks = breaks, include.lowest = TRUE, right = FALSE)

# Calculate frequency and mean Loan Approval Probability for each class interval
approval_by_rate_bin <- Loan %>%
  group_by(BaseInterestRateBin) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the calculated class interval summary
print(approval_by_rate_bin)

# Plot the mean Loan Approval Probability by class interval
ggplot(approval_by_rate_bin, aes(x = BaseInterestRateBin, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "black") +
  labs(
    title = "Loan Approval Probability by Base Interest Rate",
    x = "Base Interest Rate (%)",
    y = "Loan Approval Probability"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()



# Load necessary libraries
library(dplyr)
library(ggplot2)

# Assuming Loan is your dataset
# Find the minimum and maximum MonthlyIncome
min_income <- min(Loan$MonthlyIncome, na.rm = TRUE)
max_income <- max(Loan$MonthlyIncome, na.rm = TRUE)

# Define the class intervals for MonthlyIncome (adjust the interval size as needed)
breaks_income <- seq(min_income, max_income, by = 1000)  # Here, I set the interval size to 1000 (you can adjust it)
Loan$Income_Bin <- cut(Loan$MonthlyIncome, breaks = breaks_income, include.lowest = TRUE, right = FALSE)

# Calculate the frequency and mean Loan Approval Probability for each class interval
approval_by_income_bin <- Loan %>%
  group_by(Income_Bin) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the calculated class interval summary
print(approval_by_income_bin)

# Plot the Loan Approval Probability by Monthly Income class intervals
ggplot(approval_by_income_bin, aes(x = Income_Bin, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "lightgreen", color = "black") +
  labs(
    title = "Loan Approval Probability by Monthly Income",
    x = "Monthly Income (Class Intervals)",
    y = "Loan Approval Probability"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()

# Fit a curve to the data based on the mean approval probability per class
ggplot(approval_by_income_bin, aes(x = as.numeric(as.character(Income_Bin)), y = Approval_Rate)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = FALSE, color = "red") +
  labs(
    title = "Fitted Curve for Loan Approval Probability vs Monthly Income",
    x = "Monthly Income (Class Midpoint)",
    y = "Loan Approval Probability"
  ) +
  theme_minimal()



# Load necessary libraries
library(dplyr)
library(ggplot2)

# Assuming Loan is your dataset
# Find the minimum and maximum MonthlyIncome
min_income <- min(Loan$MonthlyIncome, na.rm = TRUE)
max_income <- max(Loan$MonthlyIncome, na.rm = TRUE)

# Define the class intervals for MonthlyIncome (adjust the interval size as needed)
breaks_income <- seq(min_income, max_income, by = 1000)  # Set the interval size to 1000 (you can adjust it)
Loan$Income_Bin <- cut(Loan$MonthlyIncome, breaks = breaks_income, include.lowest = TRUE, right = FALSE)

# Calculate the frequency and mean Loan Approval Probability for each class interval
approval_by_income_bin <- Loan %>%
  group_by(Income_Bin) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the calculated class interval summary
print(approval_by_income_bin)

# Create a bar plot to visualize Loan Approval Probability by Monthly Income class intervals
ggplot(approval_by_income_bin, aes(x = Income_Bin, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "black") +
  labs(
    title = "Loan Approval Probability by Monthly Income",
    x = "Monthly Income (Class Intervals)",
    y = "Loan Approval Probability"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()




# Load necessary libraries
library(dplyr)
library(ggplot2)

# Assuming 'Loan' is your dataset

# Find the minimum and maximum TotalDebtToIncomeRatio
min_tdi <- min(Loan$TotalDebtToIncomeRatio, na.rm = TRUE)
max_tdi <- max(Loan$TotalDebtToIncomeRatio, na.rm = TRUE)

# Create class intervals for TotalDebtToIncomeRatio (e.g., 0-0.1, 0.1-0.2, ...)
bin_width_tdi <- 0.1  # Define the width of each class interval
breaks_tdi <- seq(min_tdi, max_tdi, by = bin_width_tdi)
Loan$TDI_Bin <- cut(Loan$TotalDebtToIncomeRatio, breaks = breaks_tdi, include.lowest = TRUE, right = FALSE)

# Calculate frequency and mean Loan Approval Probability for each class interval
approval_by_tdi_bin <- Loan %>%
  group_by(TDI_Bin) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the calculated class interval summary
print(approval_by_tdi_bin)

# Create the bar graph for Loan Approval Probability by Total Debt-to-Income Ratio class intervals
ggplot(approval_by_tdi_bin, aes(x = TDI_Bin, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "black") +
  labs(
    title = "Loan Approval Probability by Total Debt-to-Income Ratio",
    x = "Total Debt-to-Income Ratio (Class Intervals)",
    y = "Loan Approval Probability"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()


# Create class intervals for BaseInterestRate (0.10-0.45)
breaks <- seq(0.10, 0.45, by = 0.05)
Loan$InterestRateBin <- cut(Loan$BaseInterestRate, breaks = breaks, include.lowest = TRUE, right = FALSE)

# Calculate frequency and mean Loan Approval Probability for each class interval
approval_by_bin_interest <- Loan %>%
  group_by(InterestRateBin) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the calculated class interval summary
print(approval_by_bin_interest)

# Plot the mean Loan Approval Probability by class interval
ggplot(approval_by_bin_interest, aes(x = InterestRateBin, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "black") +
  labs(
    title = "Loan Approval Probability by Interest Rate",
    x = "Interest Rate (Class Intervals)",
    y = "Loan Approval Probability"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()



# Load necessary libraries
library(dplyr)
library(ggplot2)

# Assuming 'Loan' is your dataset
# Find the minimum and maximum LoanAmount
min_loan <- min(Loan$LoanAmount, na.rm = TRUE)
max_loan <- max(Loan$LoanAmount, na.rm = TRUE)

# Create class intervals for Loan Amount (e.g., 0-10, 10-20, ...)
bin_width <- 10000  # Define the width of each class interval (adjust as needed)
breaks <- seq(min_loan, max_loan, by = bin_width)

# Create a new column 'LoanAmountBin' representing the loan amount class intervals
Loan$LoanAmountBin <- cut(Loan$LoanAmount, breaks = breaks, include.lowest = TRUE, right = FALSE)

# Calculate frequency, mean Loan Amount, and Loan Approval Probability for each class interval
approval_by_bin <- Loan %>%
  group_by(LoanAmountBin) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications,
    Mean_Loan_Amount = mean(LoanAmount, na.rm = TRUE)
  )

# Print the calculated class interval summary
print(approval_by_bin)

# Plot the Loan Approval Probability by mean Loan Amount for each class interval
ggplot(approval_by_bin, aes(x = Mean_Loan_Amount, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "black") +
  labs(
    title = "Loan Approval Probability by Loan Amount",
    x = "Mean Loan Amount",
    y = "Loan Approval Probability"
  ) +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))






# Load necessary libraries
library(dplyr)
library(ggplot2)

# Assuming 'Loan' is your dataset and NetWorth is the relevant column
# Find the minimum and maximum NetWorth
min_networth <- min(Loan$NetWorth, na.rm = TRUE)
max_networth <- max(Loan$NetWorth, na.rm = TRUE)

# Define the class intervals for NetWorth (e.g., 0-50000, 50000-100000, etc.)
bin_width <- 50000  # Define a suitable bin width
breaks <- seq(min_networth, max_networth, by = bin_width)

# Create class intervals for NetWorth
Loan$NetWorthBin <- cut(Loan$NetWorth, breaks = breaks, include.lowest = TRUE, right = FALSE)

# Calculate frequency and mean Loan Approval Probability for each class interval
approval_by_bin <- Loan %>%
  group_by(NetWorthBin) %>%
  summarize(
    Total_Applications = n(),
    Approved = sum(as.numeric(as.character(LoanApproved))),
    Approval_Rate = Approved / Total_Applications
  )

# Print the calculated class interval summary
print(approval_by_bin)

# Plot the Loan Approval Probability by class interval of NetWorth
ggplot(approval_by_bin, aes(x = NetWorthBin, y = Approval_Rate)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "black") +
  labs(
    title = "Loan Approval Probability by Net Worth",
    x = "Net Worth (Class Intervals)",
    y = "Loan Approval Probability"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()











# Load necessary libraries
library(ggplot2)

# Assuming 'Loan' is your dataset and 'RiskScore' and 'TotalDebtToIncomeRatio' are the relevant columns
# Fit the linear regression model
model <- lm(RiskScore ~ TotalDebtToIncomeRatio, data = Loan)

# Display the model summary to get R-squared value
summary(model)

# Plot the relationship between Total Debt-to-Income Ratio and Risk Score
ggplot(Loan, aes(x = TotalDebtToIncomeRatio, y = RiskScore)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +  # Add regression line
  labs(
    title = "Risk Score vs. Total Debt-to-Income Ratio",
    x = "Total Debt-to-Income Ratio",
    y = "Risk Score"
  ) +
  theme_minimal()





# Load necessary libraries
library(ggplot2)

# Assuming 'Loan' is your dataset and 'RiskScore' and 'MonthlyIncome' are the relevant columns
# Fit the linear regression model
model_monthly_income <- lm(RiskScore ~ MonthlyIncome, data = Loan)

# Display the model summary to get R-squared value
summary(model_monthly_income)

# Plot the relationship between Monthly Income and Risk Score
ggplot(Loan, aes(x = MonthlyIncome, y = RiskScore)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +  # Add regression line
  labs(
    title = "Risk Score vs. Monthly Income",
    x = "Monthly Income",
    y = "Risk Score"
  ) +
  theme_minimal()





# Fit the model for Risk Score vs. Interest Rate
model_interest_rate <- lm(RiskScore ~ BaseInterestRate, data = Loan)

# View the summary of the model
summary(model_interest_rate)

# Plot the relationship between Risk Score and Interest Rate
ggplot(Loan, aes(x = BaseInterestRate, y = RiskScore)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +  # Add regression line
  labs(
    title = "Risk Score vs. Interest Rate",
    x = "Interest Rate",
    y = "Risk Score"
  ) +
  theme_minimal()




library(dplyr)
library(ggplot2)

# Step 1: Group by RiskScore and compute the average BankruptcyHistory (interpreted as a probability)
avg_bankruptcy_by_risk <- Loan %>%
  group_by(RiskScore) %>%
  summarise(AvgBankruptcy = mean(BankruptcyHistory, na.rm = TRUE))

# Step 2: Fit a linear model
model_bankruptcy <- lm(AvgBankruptcy ~ RiskScore, data = avg_bankruptcy_by_risk)

# Step 3: View model summary (to get R-squared)
summary(model_bankruptcy)

# Step 4: Plot
ggplot(avg_bankruptcy_by_risk, aes(x = RiskScore, y = AvgBankruptcy)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(
    title = "Average Bankruptcy History vs. Risk Score",
    x = "Risk Score",
    y = "Average Bankruptcy History (Probability)"
  ) +
  theme_minimal()

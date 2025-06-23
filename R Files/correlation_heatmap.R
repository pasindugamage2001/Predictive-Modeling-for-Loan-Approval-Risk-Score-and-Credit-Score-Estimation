ls()
rm(list = ls())
getwd()
setwd("E:/3rd year/sem2/ST 3082/Final project")
# Load the data set
data=read.csv(file="Loan.csv",header = TRUE,sep = ",")

names(data)
library(dplyr)
# Assuming your data is in train_data and test_data
data <- data %>%
  select(-c(ApplicationDate, AnnualIncome, DebtToIncomeRatio,EducationLevel,Experience,PaymentHistory,
            UtilityBillsPaymentHistory,JobTenure,SavingsAccountBalance,CheckingAccountBalance,TotalAssets))  # Remove LoanApproved column

# Split into training and testing
set.seed(123)
index <- sample(1:nrow(data), 0.2 * nrow(data))  # 20% for test

test_data  <- data[index, ]
train_data <- data[-index, ]

# Install if not already
install.packages("ggcorrplot")

# Load required libraries
library(ggcorrplot)

# Create a dataframe with your selected numeric variables
selected_data <- data.frame(
  riskscore = train_data$RiskScore,
  credit_score = train_data$CreditScore,
  dti = train_data$TotalDebtToIncomeRatio,
  age = train_data$Age,
  loan_amount = train_data$LoanAmount,
  loan_duration = train_data$LoanDuration,
  net_worth = train_data$NetWorth,
  interest_rate = train_data$InterestRate
)

# Compute correlation matrix
cor_matrix <- cor(selected_data, use = "complete.obs")

# Plot the heatmap
ggcorrplot(cor_matrix, 
           method = "circle", 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           colors = c("blue", "white", "red"), 
           title = "Correlation Heatmap of Key Numeric Variables", 
           ggtheme = ggplot2::theme_minimal())

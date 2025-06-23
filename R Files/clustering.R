
ls()
rm(list = ls())
getwd()
setwd("E:/3rd year/sem2/ST 3082/Final project")
# Load the data set
data=read.csv(file="Loan.csv",header = TRUE,sep = ",")
library(dplyr)
# Assuming your data is in train_data and test_data
data <- data %>%
  select(-c(ApplicationDate, AnnualIncome, DebtToIncomeRatio,EducationLevel,TotalAssets,CheckingAccountBalance
            ,SavingsAccountBalance,UtilityBillsPaymentHistory,Experience))  # Remove LoanApproved column


# Split
set.seed(123)
train_index <- createDataPartition(data$LoanApproved, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

names(data)


# Load libraries
library(randomForest)
library(caret)


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


final_rf_model <- randomForest(LoanApproved ~ ., data = train_data, ntree=50,mtry=6)  # Adjust the weights as needed

# Get variable importance from the model
importance_df <- importance(final_rf_model)
importance_df <- data.frame(Variable = rownames(importance_df), Importance = importance_df[, 1])
importance_df <- importance_df %>% arrange(desc(Importance))

important_vars=importance_df$Variable

# Subset the data
data_subset <- train_data %>% select(c(all_of(important_vars),LoanApproved))

#------------------------------ Clean and Convert ------------------------------
# Convert target to factor, others to numeric if needed
data_subset$LoanApproved <- as.factor(data_subset$LoanApproved)

data_numeric <- data_subset %>%
  mutate(across(.fns = ~ if(is.factor(.) || is.character(.)) as.numeric(as.factor(.)) else .))

#------------------------------ Train RDA Model ------------------------------
# gamma and lambda are regularization parameters — tune if needed

library(klaR)
set.seed(123)
rda_model <- rda(LoanApproved ~ ., data = data_numeric, gamma = 0.1, lambda = 0.1)

#------------------------------ View RDA Summary ------------------------------
summary(rda_model)


#------------------------------ Normalize the Data ------------------------------
#------------------------------ Normalize the Data ------------------------------
#------------------------------ Normalize the Data ------------------------------
data_numeric_scaled <- dplyr::select(data_numeric, -LoanApproved) %>%  # Exclude the target variable for clustering
  scale()                    # Normalize the features

#------------------------------ Apply K-means Clustering ------------------------------
set.seed(123)
kmeans_model <- kmeans(data_numeric_scaled , centers = 3, nstart = 20)

#------------------------------ Add Cluster Labels to Data ------------------------------
data_numeric$Cluster <- as.factor(kmeans_model$cluster)

#------------------------------ View Cluster Summary ------------------------------
table(data_numeric$Cluster)

#------------------------------ Visualize Clusters ------------------------------
# If you have more than two variables, use PCA for dimensionality reduction
pca <- prcomp(data_numeric_scaled)
pca_data <- data.frame(pca$x[, 1:2], Cluster = data_numeric$Cluster)

library(ggplot2)
ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point() +
  labs(title = "K-means Clusters (PCA)", x = "PC1", y = "PC2")


library(cluster)
sil <- silhouette(kmeans_model$cluster, dist(data_numeric_scaled))
plot(sil, main = "Silhouette Plot")
mean(sil[, 3])  # average silhouette width

#------------------------------ Elbow Method ------------------------------
set.seed(123)  # for reproducibility
wss <- vector()  # to store within-cluster sum of squares

# Try k from 1 to 10
for (k in 1:10) {
  kmeans_model <- kmeans(data_numeric_scaled, centers = k, nstart = 25)
  wss[k] <- kmeans_model$tot.withinss
}

# Plot the Elbow
plot(1:10, wss, type = "b", pch = 19,
     xlab = "Number of Clusters (k)",
     ylab = "Total Within-Cluster Sum of Squares (WSS)",
     main = "Elbow Method for Choosing Optimal k")

library(cluster)

# Initialize vector to store average silhouette widths
avg_sil <- numeric(5)

# Loop through k = 2 to 5
for (k in 2:5) {
  km <- kmeans(data_numeric_scaled, centers = k, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_numeric_scaled))
  avg_sil[k] <- mean(sil[, 3])
}

# Plot average silhouette width vs. number of clusters
plot(2:5, avg_sil[2:5], type = "b", pch = 19,
     xlab = "Number of Clusters (k)",
     ylab = "Average Silhouette Width",
     main = "Average Silhouette vs Number of Clusters")


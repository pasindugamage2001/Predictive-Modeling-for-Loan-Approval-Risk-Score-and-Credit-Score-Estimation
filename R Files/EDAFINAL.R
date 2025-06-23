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
library(dplyr)
library(ggplot2)
library(readxl)

# Import the cleaned dataset (after preprocessing)
Loan <- read.csv("C:/Users/USER/Desktop/Loan_Cleaned.csv")

# ---- Data Preprocessing for Clustering ----
# Select only numerical variables for clustering
loan_numeric <- Loan %>%
  select(Age, AnnualIncome, CreditScore, LoanAmount, LoanDuration, 
         NumberOfDependents, MonthlyDebtPayments, CreditCardUtilizationRate,
         NumberOfOpenCreditLines, DebtToIncomeRatio, PreviousLoanDefaults, 
         LengthOfCreditHistory, SavingsAccountBalance, CheckingAccountBalance, 
         TotalAssets, TotalLiabilities, JobTenure, NetWorth,
         BaseInterestRate, InterestRate, MonthlyLoanPayment, TotalDebtToIncomeRatio)

# Remove any rows with missing values (if necessary)
loan_numeric <- na.omit(loan_numeric)

# Normalize the data (important for clustering)
loan_scaled <- scale(loan_numeric)

# Perform K-means clustering with 3 clusters (you can adjust the number of clusters as needed)
set.seed(123)  # Set seed for reproducibility
kmeans_result <- kmeans(loan_scaled, centers = 3, nstart = 25)

# Add the cluster assignments to the original dataset
Loan$Cluster <- as.factor(kmeans_result$cluster)

# Print K-means result
print(kmeans_result)

# ---- Visualize the Clusters ----
# Perform PCA (Principal Component Analysis) to reduce data to 2 dimensions for visualization
pca <- prcomp(loan_scaled)
pca_data <- data.frame(pca$x)

# Add cluster information to the PCA data for plotting
pca_data$Cluster <- as.factor(kmeans_result$cluster)

# Plot the first two principal components with cluster coloring
ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point() +
  labs(title = "K-means Clustering Results (PCA)",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()

# ---- Optional: Elbow Method to Determine Optimal Number of Clusters ----
wss <- (nrow(loan_scaled) - 1) * sum(apply(loan_scaled, 2, var))  # Within-cluster sum of squares for k=1
for (i in 2:15) wss[i] <- sum(kmeans(loan_scaled, centers = i, nstart = 25)$withinss)
plot(1:15, wss, type = "b", main = "Elbow Method for Optimal Clusters", 
     xlab = "Number of Clusters", ylab = "Within-cluster Sum of Squares")


# Get summary statistics for each cluster
cluster_summary <- Loan %>%
  group_by(Cluster) %>%
  summarise_all(list(~mean(.), ~sd(.)), na.rm = TRUE)

# Print the summary statistics
print(cluster_summary)



# Exclude non-numeric columns (like dates and factors) from the summary statistics
numeric_vars <- Loan %>% select(where(is.numeric))  # Select only numeric columns

# Now, calculate the summary statistics for the numeric columns only
cluster_summary <- Loan %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), list(mean = ~mean(. , na.rm = TRUE), 
                                           sd = ~sd(. , na.rm = TRUE)), .names = "{.col}_{.fn}"))

# Print the summary statistics
print(cluster_summary)
Loan$Cluster <- kmeans_result$cluster  # Assign cluster labels to the data
library(ggplot2)

ggplot(Loan, aes(x = AnnualIncome, y = CreditScore, color = factor(Cluster))) + 
  geom_point() +
  labs(title = "Clusters based on Annual Income and Credit Score", 
       x = "Annual Income", y = "Credit Score", color = "Cluster")

Loan_scaled <- Loan %>%
  select(AnnualIncome, CreditScore) %>%
  scale()  # Scale the selected variables

# Re-run k-means clustering on the standardized data
kmeans_result <- kmeans(Loan_scaled, centers = 3, nstart = 25)

# Add the cluster labels back to the original dataset
Loan$Cluster <- as.factor(kmeans_result$cluster)

# Plot the clusters again
ggplot(Loan, aes(x = AnnualIncome, y = CreditScore, color = Cluster)) + 
  geom_point() +
  labs(title = "Clusters based on Annual Income and Credit Score", 
       x = "Annual Income", y = "Credit Score", color = "Cluster")

library(dplyr)

# Compute summary statistics for each cluster
cluster_summary <- Loan %>%
  group_by(Cluster) %>%
  summarise(across(c(AnnualIncome, CreditScore, Age, LoanAmount, MonthlyDebtPayments), 
                   list(mean = ~ mean(.), sd = ~ sd(.)), 
                   .names = "{.col}_{.fn}"))

print(cluster_summary)
# Apply PCA to the dataset
pca <- prcomp(Loan %>% select(AnnualIncome, CreditScore, Age, LoanAmount, MonthlyDebtPayments), 
              center = TRUE, scale. = TRUE)

# Get the first two principal components
Loan_pca <- as.data.frame(pca$x)

# Add cluster labels
Loan_pca$Cluster <- as.factor(kmeans_result$cluster)

# Plot PCA results with clusters
ggplot(Loan_pca, aes(x = PC1, y = PC2, color = Cluster)) + 
  geom_point() +
  labs(title = "PCA-based Clustering", 
       x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")





# Apply PCA to the dataset
pca <- prcomp(Loan %>% select(AnnualIncome, CreditScore, Age, LoanAmount, MonthlyDebtPayments), 
              center = TRUE, scale. = TRUE)

# Get the first two principal components
Loan_pca <- as.data.frame(pca$x)

# Add cluster labels
Loan_pca$Cluster <- as.factor(kmeans_result$cluster)

# Plot PCA results with clusters
ggplot(Loan_pca, aes(x = PC1, y = PC2, color = Cluster)) + 
  geom_point() +
  labs(title = "PCA-based Clustering", 
       x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")


# Check the summary of PCA to see the variance explained by each component
summary(pca)

# Add PC3 to Loan_pca
Loan_pca$PC3 <- pca$x[, 3]

# 3D plot for better visualization of the clusters
library(plotly)
plot_ly(Loan_pca, x = ~PC1, y = ~PC2, z = ~PC3, color = ~Cluster, type = "scatter3d", mode = "markers") %>%
  layout(title = "3D PCA-based Clustering", 
         scene = list(xaxis = list(title = 'PC1'), 
                      yaxis = list(title = 'PC2'), 
                      zaxis = list(title = 'PC3')))

# Try a different number of clusters (e.g., 4 clusters)
kmeans_result_4 <- kmeans(Loan_scaled, centers = 4, nstart = 25)
Loan_pca$Cluster_4 <- as.factor(kmeans_result_4$cluster)

# Plot PCA-based clustering with 4 clusters
ggplot(Loan_pca, aes(x = PC1, y = PC2, color = Cluster_4)) + 
  geom_point() +
  labs(title = "PCA-based Clustering with 4 Clusters", 
       x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")

# Install and load the cluster package for silhouette analysis
if (!require(cluster)) install.packages("cluster", dependencies = TRUE)
library(cluster)

# Perform silhouette analysis for 4 clusters
silhouette_score <- silhouette(kmeans_result_4$cluster, dist(Loan_scaled))
plot(silhouette_score, main = "Silhouette Plot for 4 Clusters")




ggplot(Loan_pca, aes(x = PC1, y = PC2, color = Cluster)) + 
  geom_point() +
  labs(title = "PCA-based Clustering", 
       x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")




# Check the proportion of variance explained by each principal component
summary(pca)

# Add PC3 and Cluster information to the PCA result
Loan_pca$PC3 <- pca$x[, 3]

# Plot the first three principal components
ggplot(Loan_pca, aes(x = PC1, y = PC2, color = Cluster)) + 
  geom_point() +
  labs(title = "PCA-based Clustering with First 3 PCs", 
       x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")




# Perform k-means clustering on PCA components
kmeans_result_pca <- kmeans(Loan_pca[, c("PC1", "PC2")], centers = 3, nstart = 25)

# Add the cluster results back to the PCA dataset
Loan_pca$Cluster_pca <- as.factor(kmeans_result_pca$cluster)

# Plot the clustering results in PCA space
ggplot(Loan_pca, aes(x = PC1, y = PC2, color = Cluster_pca)) + 
  geom_point() +
  labs(title = "Clustering based on PCA Components", 
       x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")



# Check cluster centroids on PCA components
data.frame(kmeans_result_pca$centers)
# Calculate WCSS for k-means clustering
wcss <- kmeans_result_pca$tot.withinss
print(paste("WCSS:", wcss))

# Calculate silhouette score (requires 'cluster' library)
library(cluster)
silhouette_score <- silhouette(kmeans_result_pca$cluster, dist(Loan_pca[, c("PC1", "PC2")]))
plot(silhouette_score)

# Elbow method to find optimal number of clusters
wcss_values <- sapply(1:10, function(k) kmeans(Loan_pca[, c("PC1", "PC2")], centers = k, nstart = 25)$tot.withinss)
plot(1:10, wcss_values, type = "b", main = "Elbow Method", xlab = "Number of Clusters", ylab = "WCSS")


# Load necessary library
library(ggplot2)

# Calculate WCSS for different numbers of clusters (1 to 10)
wcss_values <- sapply(1:10, function(k) kmeans(Loan_pca[, c("PC1", "PC2")], centers = k, nstart = 25)$tot.withinss)

# Plot the WCSS values
plot(1:10, wcss_values, type = "b", main = "Elbow Method for Optimal Clusters", 
     xlab = "Number of Clusters", ylab = "WCSS", pch = 19, col = "blue")

# Identify the number of clusters at the "elbow" by inspecting the plot manually
# To automate the elbow detection, you can use a simple heuristic approach:

# Calculate the rate of change in WCSS between consecutive numbers of clusters
diff_wcss <- diff(wcss_values)

# Look for the point where the rate of change starts decreasing sharply
optimal_clusters <- which.min(diff_wcss)

# Print the appropriate number of clusters
cat("The appropriate number of clusters is:", optimal_clusters + 1, "\n")




library(cluster)

# Calculate and plot silhouette scores for different numbers of clusters (1 to 10)
silhouette_scores <- sapply(2:10, function(k) {
  kmeans_result <- kmeans(Loan_pca[, c("PC1", "PC2")], centers = k, nstart = 25)
  silhouette(kmeans_result$cluster, dist(Loan_pca[, c("PC1", "PC2")]))$avg.width
})

# Plot the silhouette scores
plot(2:10, silhouette_scores, type = "b", main = "Silhouette Scores for Different Numbers of Clusters",
     xlab = "Number of Clusters", ylab = "Average Silhouette Score", pch = 19, col = "blue")

# Print the number of clusters with the highest silhouette score
optimal_clusters_silhouette <- which.max(silhouette_scores) + 1
cat("The optimal number of clusters based on the silhouette score is:", optimal_clusters_silhouette, "\n")





library(cluster)

# Calculate and plot silhouette scores for different numbers of clusters (2 to 10)
silhouette_scores <- sapply(2:10, function(k) {
  kmeans_result <- kmeans(Loan_pca[, c("PC1", "PC2")], centers = k, nstart = 25)
  sil_score <- silhouette(kmeans_result$cluster, dist(Loan_pca[, c("PC1", "PC2")]))
  mean(sil_score[, 3])  # Extracting the average silhouette width
})

# Plot the silhouette scores
plot(2:10, silhouette_scores, type = "b", main = "Silhouette Scores for Different Numbers of Clusters",
     xlab = "Number of Clusters", ylab = "Average Silhouette Score", pch = 19, col = "blue")

# Print the number of clusters with the highest silhouette score
optimal_clusters_silhouette <- which.max(silhouette_scores) + 1
cat("The optimal number of clusters based on the silhouette score is:", optimal_clusters_silhouette, "\n")




# Perform k-means clustering with 3 clusters
kmeans_result_3 <- kmeans(Loan_pca[, c("PC1", "PC2")], centers = 3, nstart = 25)

# Add the cluster labels back to the PCA dataset
Loan_pca$Cluster_3 <- as.factor(kmeans_result_3$cluster)

# Plot the clustering results in PCA space with 3 clusters
ggplot(Loan_pca, aes(x = PC1, y = PC2, color = Cluster_3)) + 
  geom_point() +
  labs(title = "Clustering based on PCA Components (3 Clusters)", 
       x = "Principal Component 1", y = "Principal Component 2", color = "Cluster") 




cluster_summary <- Loan %>%
  group_by(Cluster_3) %>%
  summarise(across(everything(), list(mean = mean, sd = sd), .names = "{col}_mean"))

print(cluster_summary)
install.packages("psych")  # if not already installed
library(psych)
Loan_fa <- Loan %>%
  select(AnnualIncome, CreditScore, Age, LoanAmount, MonthlyDebtPayments)
# Kaiser-Meyer-Olkin (KMO) Test
KMO(Loan_fa)

# Bartlett’s Test of Sphericity
cortest.bartlett(cor(Loan_fa), n = nrow(Loan_fa))
fa.parallel(Loan_fa, fa = "fa", n.iter = 100)
fa_result <- fa(Loan_fa, nfactors = 2, rotate = "varimax", fm = "ml")  # You can change to fm = "pa" for principal axis
print(fa_result)
factor_scores <- as.data.frame(fa_result$scores)
Loan <- bind_cols(Loan, factor_scores)
plot(fa_result)
fa.diagram(fa_result)
library(dplyr)

# Step 1: Calculate the mean Age and CreditScore for each cluster
cluster_summary <- Loan %>%
  group_by(Cluster) %>%
  summarise(
    Avg_Age = mean(Age, na.rm = TRUE),
    Avg_CreditScore = mean(CreditScore, na.rm = TRUE)
  )

print(cluster_summary)



# Create a vector of descriptive names matching cluster numbers
cluster_labels <- c(
  "Young - Low CreditScore",     # Cluster 1
  "Old - High CreditScore",      # Cluster 2
  "Middle-Aged - Medium CreditScore"  # Cluster 3
)

# Assign labels to the Loan dataset
Loan$Cluster_Label <- factor(Loan$Cluster,
                             levels = c("1", "2", "3"),
                             labels = cluster_labels)
ggplot(Loan, aes(x = Age, y = CreditScore, color = Cluster_Label)) +
  geom_point() +
  labs(title = "Clusters with Descriptive Names",
       x = "Age", y = "Credit Score", color = "Cluster Description")





Loan_pca$Cluster_pca
# Step 1: Add original data to PCA cluster results
Loan_pca$Age <- Loan$Age
Loan_pca$CreditScore <- Loan$CreditScore

# Step 2: Check average Age and CreditScore by cluster
cluster_summary <- Loan_pca %>%
  group_by(Cluster_pca) %>%
  summarise(
    Mean_Age = mean(Age, na.rm = TRUE),
    Mean_CreditScore = mean(CreditScore, na.rm = TRUE)
  )

print(cluster_summary)

# Map descriptive names (you can change these based on your actual summary)
cluster_labels <- c(
  "1" = "Young, Low Credit Score",
  "2" = "Older, High Credit Score",
  "3" = "Middle-aged, Medium Credit Score"
)

# Assign new labels
Loan_pca$Cluster_Label <- factor(Loan_pca$Cluster_pca,
                                 levels = names(cluster_labels),
                                 labels = cluster_labels)
ggplot(Loan_pca, aes(x = PC1, y = PC2, color = Cluster_Label)) + 
  geom_point() +
  labs(title = "PCA Clustering with Descriptive Labels", 
       x = "Principal Component 1", y = "Principal Component 2", color = "Cluster")
ggplot(Loan_pca, aes(x = PC1, y = PC2)) + 
  geom_point() +
  labs(title = "PCA Scatter Plot without Clustering")

library(clustertend)
set.seed(123)
hopkins_stat <- hopkins(Loan_scaled, n = nrow(Loan_scaled) - 1)
print(hopkins_stat)
install.packages("clustertend")
library(clustertend)
set.seed(123)  # for reproducibility
hopkins_stat <- hopkins(Loan_scaled, n = nrow(Loan_scaled) - 1)
print(hopkins_stat)







#Data Analysis
summary(Loan)
str(Loan)
ggplot(Loan, aes(x = CreditScore)) + 
  geom_histogram(binwidth = 20, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Credit Score")
ggplot(Loan, aes(x = LoanApprovalStatus)) +
  geom_bar(fill = "coral") +
  labs(title = "Loan Approval Status")
colnames(Loan)
# Checking column names to make sure LoanApprovalStatus exists
colnames(Loan)

# If the column is correctly named as LoanApprovalStatus, then the following will work:
ggplot(Loan, aes(x = LoanApprovalStatus)) + 
  geom_bar(fill = "coral") +
  labs(title = "Loan Approval Status")
ggplot(Loan, aes(x = LoanApproved)) + 
  geom_bar(fill = "coral") +
  labs(title = "Loan Approval Status")
ggplot(Loan, aes(x = LoanApproved, y = CreditScore, fill = LoanApproved)) + 
  geom_boxplot() +
  labs(title = "Credit Score by Loan Approval Status", 
       x = "Loan Approval Status", y = "Credit Score")

ggplot(Loan, aes(x = LoanApproved, y = AnnualIncome, fill = LoanApproved)) +
  geom_boxplot() +
  labs(title = "Annual Income by Loan Approval Status")


library(corrplot)
numeric_vars <- Loan %>% select_if(is.numeric)
cor_matrix <- cor(numeric_vars, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)

library(corrplot)

# Select numeric variables from the dataset
numeric_vars <- Loan %>% select_if(is.numeric)

# Calculate the correlation matrix
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Create the correlation plot with cell colors (no values in the cells)
corrplot(cor_matrix, 
         method = "color",           # Color-based correlation plot
         type = "upper",             # Only show the upper triangle
         tl.col = "black",           # Text label color
         tl.cex = 0.8,               # Adjust text size of labels
         addCoef.col = NULL,         # Remove correlation values from cells
         col = colorRampPalette(c("blue", "white", "red"))(200),  # Set color gradient (blue for negative, red for positive)
         diag = FALSE,               # Remove the diagonal
         cl.ratio = 0.2,             # Adjust the legend size ratio
         cl.align = "c",             # Align the legend text in the center
         tl.srt = 45,                # Rotate labels
         number.cex = 0,             # Hide the numbers (correlation values) inside cells
         cl.lim = c(-1, 1)           # Set limits for color scale (-1 to 1)
)

# Add a separate color legend to describe correlation range
legend("topright", 
       legend = c("Strong Negative", "Moderate Negative", "Weak Negative", 
                  "No Correlation", "Weak Positive", "Moderate Positive", "Strong Positive"),
       fill = colorRampPalette(c("blue", "white", "red"))(7), 
       title = "Correlation", 
       cex = 0.8)




library(corrplot)

# Select numerical variables from the dataset
numeric_vars <- Loan %>% select_if(is.numeric)

# Calculate the correlation matrix
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Generate the correlation plot with numbered labels, no correlation values in cells
corrplot(cor_matrix, 
         method = "color",           # Color-based correlation plot
         type = "upper",             # Only show the upper triangle
         tl.col = "black",           # Text label color
         tl.cex = 0.8,               # Adjust text size of labels
         addCoef.col = NULL,         # Remove correlation values from cells
         col = colorRampPalette(c("blue", "white", "red"))(200),  # Set color gradient (blue for negative, red for positive)
         diag = FALSE,               # Remove the diagonal
         cl.ratio = 0.2,             # Adjust the legend size ratio
         cl.align = "c",             # Align the legend text in the center
         tl.srt = 45,                # Rotate labels
         number.cex = 0,             # Hide the numbers (correlation values) inside cells
         cl.lim = c(-1, 1),          # Set limits for color scale (-1 to 1)
         labels = 1:ncol(cor_matrix),  # Use numerical labels for the variables (1, 2, 3, ...)
         mar = c(0, 0, 1, 0)         # Adjust margins to give more space for labels
)

# Add a separate color legend to describe the correlation ranges
legend("topright", 
       legend = c("Strong Negative (-1 to -0.8)", 
                  "Moderate Negative (-0.8 to -0.5)", 
                  "Weak Negative (-0.5 to -0.2)", 
                  "No Correlation (0 to 0)", 
                  "Weak Positive (0 to 0.2)", 
                  "Moderate Positive (0.2 to 0.5)", 
                  "Strong Positive (0.5 to 1)"),
       fill = colorRampPalette(c("blue", "white", "red"))(7), 
       title = "Correlation Range", 
       cex = 0.8)


library(corrplot)
library(dplyr)

# Select numerical variables from the dataset
numeric_vars <- Loan %>% select_if(is.numeric)

# Calculate the correlation matrix
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Set the variable names as numbers for labels
numeric_labels <- as.character(1:ncol(cor_matrix))

# Generate the correlation plot with numbered labels
corrplot(cor_matrix, 
         method = "color",           # Color-based correlation plot
         type = "upper",             # Only show the upper triangle
         tl.col = "black",           # Text label color
         tl.cex = 0.8,               # Adjust text size of labels
         addCoef.col = NULL,         # Remove correlation values from cells
         col = colorRampPalette(c("blue", "white", "red"))(200),  # Set color gradient (blue for negative, red for positive)
         diag = FALSE,               # Remove the diagonal
         cl.ratio = 0.2,             # Adjust the legend size ratio
         cl.align = "c",             # Align the legend text in the center
         tl.srt = 45,                # Rotate labels
         number.cex = 0,             # Hide the numbers (correlation values) inside cells
         cl.lim = c(-1, 1),          # Set limits for color scale (-1 to 1)
         labels = numeric_labels,    # Use numerical labels (1, 2, 3, ...) instead of variable names
         mar = c(4, 4, 2, 1),        # Adjust margins (bottom, left, top, right)
         addgrid.col = "gray",       # Optional: grid lines for clarity
         tl.pos = "lt"               # Adjust label position (left-top)
)

# Add a separate color legend to describe the correlation ranges
legend("topright", 
       legend = c("Strong Negative (-1 to -0.8)", 
                  "Moderate Negative (-0.8 to -0.5)", 
                  "Weak Negative (-0.5 to -0.2)", 
                  "No Correlation (0 to 0)", 
                  "Weak Positive (0 to 0.2)", 
                  "Moderate Positive (0.2 to 0.5)", 
                  "Strong Positive (0.5 to 1)"),
       fill = colorRampPalette(c("blue", "white", "red"))(7), 
       title = "Correlation Range", 
       cex = 0.8)
# Histogram for CreditScore
ggplot(Loan, aes(x = CreditScore)) + 
  geom_histogram(binwidth = 20, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Credit Score")

# Histogram for Age
ggplot(Loan, aes(x = Age)) + 
  geom_histogram(binwidth = 5, fill = "green", color = "white") +
  labs(title = "Distribution of Age")


# Boxplot for CreditScore by Loan Approval Status
ggplot(Loan, aes(x = LoanApproved, y = CreditScore, fill = LoanApproved)) +
  geom_boxplot() +
  labs(title = "Credit Score by Loan Approval Status")

# Boxplot for Age by Loan Approval Status
ggplot(Loan, aes(x = LoanApproved, y = Age, fill = LoanApproved)) +
  geom_boxplot() +
  labs(title = "Age by Loan Approval Status")

# Scatter plot between Age and CreditScore
ggplot(Loan, aes(x = Age, y = CreditScore)) +
  geom_point(color = "blue") +
  labs(title = "Age vs Credit Score")

# Scatter plot between LoanAmount and AnnualIncome
ggplot(Loan, aes(x = AnnualIncome, y = LoanAmount)) +
  geom_point(color = "red") +
  labs(title = "Loan Amount vs Annual Income")
# Check correlations of variables with Loan Approval Status
cor_with_target <- cor(numeric_vars$CreditScore, Loan$LoanApproved)
print(cor_with_target)



summary(Loan$CreditScore)
summary(Loan$Age)
summary(Loan$AnnualIncome)
summary(Loan$LoanAmount)

# Load necessary libraries
library(ggplot2)
library(e1071)  # For skewness calculation

# Make sure your data is loaded correctly
# For example, if you're reading from a CSV file, it should look like this:
# data <- read.csv("your_dataset.csv")

# Replace 'CreditScore' with the actual column name if different
# Check that the column exists
if ("CreditScore" %in% colnames(data)) {
  
  # Plotting the distribution of the Credit Score
  ggplot(data, aes(x = CreditScore)) +
    geom_histogram(binwidth = 5, fill = "blue", color = "black", alpha = 0.7) +
    labs(title = "Distribution of Credit Score", x = "Credit Score", y = "Frequency") +
    theme_minimal()
  
  # Calculate mean and median of Credit Score
  mean_credit_score <- mean(data$CreditScore, na.rm = TRUE)
  median_credit_score <- median(data$CreditScore, na.rm = TRUE)
  
  # Calculate skewness
  skewness_value <- skewness(data$CreditScore, na.rm = TRUE)
  
  # Output the results
  cat("Mean Credit Score: ", mean_credit_score, "\n")
  cat("Median Credit Score: ", median_credit_score, "\n")
  cat("Skewness of Credit Score: ", skewness_value, "\n")
  
  # Interpretation of skewness
  if (skewness_value > 0) {
    cat("The distribution is positively skewed (right-skewed).\n")
  } else if (skewness_value < 0) {
    cat("The distribution is negatively skewed (left-skewed).\n")
  } else {
    cat("The distribution is perfectly symmetrical.\n")
  }
  
} else {
  cat("Error: 'CreditScore' column not found in the dataset.\n")
}

# Load necessary libraries
library(ggplot2)
library(dplyr)
library(e1071)  # For skewness calculation

# Check the column names to confirm the 'CreditScore' column
colnames(Loan)

# Assuming the credit score column is named 'CreditScore' (update if needed)
# Plot the distribution of the credit score
ggplot(Loan, aes(x = CreditScore)) + 
  geom_histogram(binwidth = 5, fill = "blue", color = "black", alpha = 0.7) + 
  labs(title = "Distribution of Credit Score", x = "Credit Score", y = "Frequency") + 
  theme_minimal()

# Calculate mean and median of the credit score
mean_credit_score <- mean(Loan$CreditScore, na.rm = TRUE)
median_credit_score <- median(Loan$CreditScore, na.rm = TRUE)

# Calculate skewness of the credit score
skewness_value <- skewness(Loan$CreditScore, na.rm = TRUE)

# Output the results
cat("Mean Credit Score: ", mean_credit_score, "\n")
cat("Median Credit Score: ", median_credit_score, "\n")
cat("Skewness of Credit Score: ", skewness_value, "\n")

# Interpretation of skewness
if (skewness_value > 0) {
  cat("The distribution is positively skewed (right-skewed).\n")
} else if (skewness_value < 0) {
  cat("The distribution is negatively skewed (left-skewed).\n")
} else {
  cat("The distribution is perfectly symmetrical.\n")
}

library(ggplot2)

# Assuming Loan$LoanApproved is factor (if not, convert using as.factor())
Loan$LoanApproved <- as.factor(Loan$LoanApproved)

ggplot(Loan, aes(x = LoanApproved, y = CreditScore, fill = LoanApproved)) +
  geom_boxplot() +
  labs(
    title = "Credit Score Distribution by Loan Approval Status",
    x = "Loan Approval Status (0 = No, 1 = Yes)",
    y = "Credit Score"
  ) +
  scale_fill_manual(values = c("red", "green")) +
  theme_minimal()


# Load necessary libraries
library(dplyr)

# Group by LoanApproved and summarize CreditScore statistics
credit_stats_by_approval <- Loan %>%
  group_by(LoanApproved) %>%
  summarise(
    Count = n(),
    Mean_CreditScore = mean(CreditScore, na.rm = TRUE),
    Median_CreditScore = median(CreditScore, na.rm = TRUE),
    Min_CreditScore = min(CreditScore, na.rm = TRUE),
    Max_CreditScore = max(CreditScore, na.rm = TRUE),
    IQR_CreditScore = IQR(CreditScore, na.rm = TRUE)
  )

# Print the summary table
print(credit_stats_by_approval)



# Load necessary libraries
library(ggplot2)

# Assuming your dataset is called Loan and columns are named 'LoanAmount' and 'AnnualIncome'
# Create the scatter plot with regression line
ggplot(Loan, aes(x = AnnualIncome, y = LoanAmount)) +
  geom_point(alpha = 0.4, color = "steelblue") +  # scatter points
  geom_smooth(method = "lm", se = TRUE, color = "darkred", linetype = "dashed") +  # regression line
  labs(title = "Loan Amount vs Annual Income with Fitted Regression Line",
       x = "Annual Income",
       y = "Loan Amount") +
  theme_minimal()

# Fit linear model
model <- lm(LoanAmount ~ AnnualIncome, data = Loan)

# Summary of the model
summary(model)


# Load necessary libraries
library(ggplot2)

# Create a scatter plot of Credit Score vs Age
ggplot(Loan, aes(x = Age, y = CreditScore)) +
  geom_point(alpha = 0.5, color = "blue") +   # Scatter plot
  labs(title = "Credit Score vs Age", x = "Age", y = "Credit Score") +
  theme_minimal()


# Load necessary libraries
library(ggplot2)

# Fit a linear regression model to Credit Score vs Age
model <- lm(CreditScore ~ Age, data = Loan)

# Create a scatter plot with the fitted regression line
ggplot(Loan, aes(x = Age, y = CreditScore)) +
  geom_point(alpha = 0.5, color = "blue") +   # Scatter plot
  geom_smooth(method = "lm", se = FALSE, color = "red") +  # Fitted regression line
  labs(title = "Credit Score vs Age with Fitted Regression Line", 
       x = "Age", 
       y = "Credit Score") +
  theme_minimal() +
  annotate("text", x = 50, y = 700, label = paste("R² =", round(summary(model)$r.squared, 3)), color = "red", size = 5)





# Load necessary libraries
library(ggplot2)

# Fit a linear regression model to Credit Score vs Age
model <- lm(CreditScore ~ Age, data = Loan)

# Create a scatter plot with the fitted regression line
ggplot(Loan, aes(x = Age, y = CreditScore)) +
  geom_point(alpha = 0.5, color = "blue") +   # Scatter plot
  geom_smooth(method = "lm", se = FALSE, color = "red") +  # Fitted regression line
  labs(title = "Credit Score vs Age with Fitted Regression Line", 
       x = "Age", 
       y = "Credit Score") +
  theme_minimal()

# Assuming your dataset is called 'Loan' and has columns 'RiskScore' and 'LoanApproved'

# Filter data to include only RiskScores between 40 and 50
Loan_filtered <- Loan %>% filter(RiskScore >= 40 & RiskScore <= 50)

# Calculate the probability of approval
# Assuming 'LoanApproved' is 1 for approved and 0 for rejected
Loan_filtered <- Loan_filtered %>%
  mutate(ApprovalProb = ifelse(LoanApproved == 1, 1, 0)) 

# Fit a logistic regression model to predict the probability of approval based on risk score
logit_model <- glm(ApprovalProb ~ RiskScore, data = Loan_filtered, family = binomial)

# Generate predicted probabilities
Loan_filtered$Predicted_Prob <- predict(logit_model, type = "response")

# Plot probability of loan approval vs RiskScore
ggplot(Loan_filtered, aes(x = RiskScore, y = Predicted_Prob)) +
  geom_point(aes(color = factor(LoanApproved)), size = 2) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE, color = "red") +
  labs(title = "Probability of Loan Approval vs Risk Score (Risk Score 40-50)",
       x = "Risk Score",
       y = "Probability of Approval") +
  scale_color_manual(values = c("blue", "green"), labels = c("Rejected", "Approved")) +
  theme_minimal()
# Fit a logistic regression model to predict the probability of approval based on risk score
logit_model_all <- glm(LoanApproved ~ RiskScore, data = Loan_train, family = binomial)

# Generate predicted probabilities
Loan_train$Predicted_Prob <- predict(logit_model_all, type = "response")

# Plot probability of loan approval vs RiskScore for all loan data in Loan_train
ggplot(Loan_train, aes(x = RiskScore, y = Predicted_Prob)) +
  geom_point(aes(color = factor(LoanApproved)), size = 2, alpha = 0.6) + # Scatter plot of actual values
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE, color = "red") + # Logistic regression curve
  labs(title = "Probability of Loan Approval vs Risk Score (All Risk Scores in Training Data)",
       x = "Risk Score",
       y = "Probability of Approval") +
  scale_color_manual(values = c("blue", "green"), labels = c("Rejected", "Approved")) + # Customize color for Approved/Rejected
  theme_minimal()
# Filter data to only include risk scores between 40 and 50
Loan_train_40_50 <- Loan_train %>%
  filter(RiskScore >= 40 & RiskScore <= 50)

# Fit a logistic regression model for the filtered data
logit_model_40_50 <- glm(LoanApproved ~ RiskScore, data = Loan_train_40_50, family = binomial)

# Generate predicted probabilities for the filtered data
Loan_train_40_50$Predicted_Prob <- predict(logit_model_40_50, type = "response")

# Calculate R-squared for logistic regression model
# McFadden's R-squared for logistic regression
logit_model_40_50_r2 <- 1 - (logLik(logit_model_40_50) / logLik(update(logit_model_40_50, . ~ 1)))
cat("McFadden's R-squared for the model: ", logit_model_40_50_r2, "\n")

# Plot probability of loan approval vs RiskScore for the filtered range (40-50)
ggplot(Loan_train_40_50, aes(x = RiskScore, y = Predicted_Prob)) +
  geom_point(aes(color = factor(LoanApproved)), size = 2, alpha = 0.6) + # Scatter plot of actual values
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE, color = "red") + # Logistic regression curve
  labs(title = "Probability of Loan Approval vs Risk Score (Risk Score 40-50)",
       x = "Risk Score",
       y = "Probability of Approval") +
  scale_color_manual(values = c("blue", "green"), labels = c("Rejected", "Approved")) + # Customize color for Approved/Rejected
  theme_minimal() +
  annotate("text", x = 45, y = 0.8, label = paste("R-squared = ", round(logit_model_40_50_r2, 3)), color = "red", size = 5) +
  annotate("text", x = 45, y = 0.75, label = paste("Logistic regression: y = ", round(coef(logit_model_40_50)[1], 2), " + ", round(coef(logit_model_40_50)[2], 2), " * RiskScore"), color = "red", size = 5)
# Fit a logistic regression model for the full range of RiskScore (0-100)
logit_model_full <- glm(LoanApproved ~ RiskScore, data = Loan_train, family = binomial)

# Fit a logistic regression model for the 40-50 RiskScore range
Loan_train_40_50 <- Loan_train %>%
  filter(RiskScore >= 40 & RiskScore <= 50)
logit_model_40_50 <- glm(LoanApproved ~ RiskScore, data = Loan_train_40_50, family = binomial)

# Generate predicted probabilities for the full range and 40-50 range
Loan_train$Predicted_Prob_full <- predict(logit_model_full, type = "response")
Loan_train_40_50$Predicted_Prob_40_50 <- predict(logit_model_40_50, type = "response")

# Plot for the full range (0-100)
ggplot(Loan_train, aes(x = RiskScore, y = Predicted_Prob_full)) +
  geom_point(aes(color = factor(LoanApproved)), size = 2, alpha = 0.6) + 
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE, color = "blue") +
  labs(title = "Probability of Loan Approval vs Risk Score (Full Range 0-100)",
       x = "Risk Score",
       y = "Probability of Loan Approval") +
  theme_minimal()

# Plot for the 40-50 range
ggplot(Loan_train_40_50, aes(x = RiskScore, y = Predicted_Prob_40_50)) +
  geom_point(aes(color = factor(LoanApproved)), size = 2, alpha = 0.6) + 
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE, color = "red") +
  labs(title = "Probability of Loan Approval vs Risk Score (Risk Score 40-50)",
       x = "Risk Score",
       y = "Probability of Loan Approval") +
  theme_minimal()

# Display R-squared values for both models
cat("R-squared for full model (0-100): ", 1 - (logLik(logit_model_full) / logLik(update(logit_model_full, . ~ 1))), "\n")
cat("R-squared for

##Data set source: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
## Reference: https://www.kaggle.com/code/rohitamalnerkar/adult-dataset-income-prediction/notebook
## Reference: https://www.kaggle.com/code/esmaeil391/ibm-hr-analysis-with-90-3-acc-and-89-auc#New-Random-Forest

#==========================================================================
# 1. data cleaning

#install libraries
install.packages("tidyverse")     
library(tidyverse)

# Read file
data <- read.csv(
  file = "adult_income.csv",
  header = TRUE,
  sep = ",",
  dec = ".",
  na.strings = c("?","NA"),      # convert "?" to "NA"
  stringsAsFactors = FALSE
)

summary(data)
glimpse(data)
# Count the number of people in each income group
income_counts <- table(data$income)

# check income distribution with par plot
barplot(income_counts, main = "Income Classification", col = "green", 
        ylab = "# of people", xlab = "Income", ylim = c(0, max(income_counts) * 1.1))

# Define colors for each race
race_colors <- c("White" = "blue", 
                         "Black" = "red", 
                         "Asian-Pac-Islander" = "green", 
                         "Amer-Indian-Eskimo" = "purple",
                         "Other" = "orange")

# Count the number of occurrences of each race in the dataset
race_counts <- table(data$race)

# Create a bar graph with the race categories along the x-axis and the count of people in each category along the y-axis
barplot(race_counts, 
        xlab = "Race", 
        ylab = "Number of People", 
        main = "Distribution of Race in the Adult Income Dataset",
        col = race_colors)

# Add a legend to the graph to explain the color scheme
#legend("topright", legend = names(race_colors), fill = race_colors)

# check "NA" values in each column
colSums(is.na(data)) 

# check the structure of the data
glimpse(data)

#Remove irrelevant columns "education.num, relationship, fnlwgt, capital.gain and capital.loss" from data set
data$capital.gain<-NULL
data$capital.loss<-NULL
data$fnlwgt<-NULL
data$education.num<-NULL
data$relationship<-NULL

# check columns have "NA" values
colSums(is.na(data)) 

#To replace "NA" values on columns "workclass, occupation, native.country", use kNN imputation method
install.packages("VIM")
library(VIM)

data_new <- kNN(data, variable = c("workclass", "occupation", "native.country"), k = sqrt(nrow(data)))

head(data_new)

# save cleaned data set 
write.csv(data_new, "ds_clean_dataset.csv", row.names=FALSE, fileEncoding = "UTF-8", na = ' ')

df <- read.csv(                      # Read csv file to perform following tasks
  file = "ds_clean_dataset.csv",
  header = TRUE,
  sep = ",",
  dec = ".",
  stringsAsFactors = FALSE
)

summary(df)
glimpse(df)

df1 <- df
colSums(df1 == 0)

df1 <- df1 %>%                     # Drop unwanted columns ("index", "salary" and "salary_currency")         
  select(-c(workclass_imp, occupation_imp, native.country_imp)) 
colSums(df1 == 0)

#################################################################

# 2. predict overall accuracy on original cleaned data set
df2 <- df1
glimpse(df2)
# convert all character columns to factor columns
for (col in names(df2)[sapply(df2, is.character)]) {
  df2[[col]] <- as.factor(df2[[col]])
}
glimpse(df2)

# convert all integer columns to numeric columns
for (col in names(df2)[sapply(df2, is.integer)]) {
  df2[[col]] <- as.numeric(df2[[col]])
}
glimpse(df2)

# make sample index
set.seed(123)
idx <- sample(1:nrow(df2), size = nrow(df2) * 0.75)
idx_y <- which(colnames(df2) == 'income')

#split train and test data set
## Train / Test Split
X_train <- df2[idx, -idx_y] 
X_test <- df2[-idx, -idx_y]
Y_test <- df2[-idx, idx_y]  #store target column
Y_train <- df2[idx, idx_y]  #store target column

summary(X_train)
glimpse(X_train)

install.packages("randomForest")
library(randomForest)
# Random forest model
rf_model <- randomForest(Y_train ~ ., data = X_train)

#predict model with test data set
rf.prd <- predict(rf_model, newdata = X_test)

actual <- Y_test

# confusion matrix
cm <- table(actual, rf.prd)
cm

#evaluate accuracy
accuracy <- mean(rf.prd == actual)
print(paste("Accuracy of Decision Forest model: ", round(accuracy*100,2))) 

#########################################################################

# 3. Predict accuracy on split test data set of original data set

# Choose a protected variable (race in this example)
protected_var <- "race"

# Split the test set into subsets by the protected variable
test_groups <- split(X_test, X_test[[protected_var]])


glimpse(Y_train)   # make sure "income" column is "factor"
# Build a Random forest model
rf_model <- randomForest(Y_train ~ ., data = X_train)

# Predict income on each subset of the test set
rf_prediction <- lapply(test_groups, function(x) predict(rf_model, newdata = x))

# Calculate prediction accuracy on each subset
rf_accuracy <- sapply(rf_prediction, function(x) sum(x == x[1])/length(x))
print(paste("Accuracy on each subset as protected variable is race :"))
rf_accuracy

# convert accuracy into percentage
accuracy_percentage <- rf_accuracy*100
print(paste("Accuracy on each subset as protected variable is race :"))
accuracy_percentage


# create bar plot with colors
barplot(accuracy_percentage, main = "Accuracy w.r.t Race", 
        col = c("pink", "red", "yellow", "orange", "green"), ylab = "Accuracy (%)", ylim = c(0, 100))



#########################################################################

# 4. solving unbalanced bias problem using Stratified method and make balanced data

df3 <- df1
glimpse(df3)

# Calculate class count
class_count <- table(df3$income)

# Calculate over and under sampling rates
over <- ((0.6 * max(class_count)) - min(class_count)) / min(class_count)
under <- (0.4 * max(class_count)) / (min(class_count) * over)

# Round over and under sampling rates and multiply by 100
over <- round(over, 1) * 100
under <- round(under, 1) * 100

# Calculate weights for differential sampling based on race
race_counts <- table(df3$race)
race_weights <- 1 / race_counts

library(caret)

# Define the sampling ratios for each protected group
sampling_ratio_white <- 0.5  # Sampling ratio for "White" group
sampling_ratio_black <- 0.5  # Sampling ratio for "Black" group
sampling_ratio_Amer_Indian <- 0.5  # Sampling ratio for "Amer_Indian_Eskimo" group
sampling_ratio_Asian_Pac <- 0.8  # Sampling ratio for "Asian_Pac_Islander" group
sampling_ratio_other <- 0.8  # Sampling ratio for "other" group

sampling_ratio <- ifelse(df3$race == "White", sampling_ratio_white,
                         ifelse(df3$race == "Black", sampling_ratio_black,
                                ifelse(df3$race == "Amer_Indian_Eskimo", sampling_ratio_Amer_Indian,
                                       ifelse(df3$race == "Asian_Pac_Islander", sampling_ratio_Asian_Pac, 
                                              ifelse(df3$race == "other", sampling_ratio_other, 0.2)))))


df_balanced <- upSample(x = df3[, -1], y = df3$income, yname = "income",
                        groupVars = df3$race, 
                        sampsize = floor(sampling_ratio * nrow(df3)))

# Check the balance of the output data set
table(df_balanced$income)
table(df_balanced$race)

########################################################################

# 5. use balanced data and split train and test and built Decision Forest model

df4 <- BalancedData
glimpse(df4)

# make sample index
set.seed(123)
idx <- sample(1:nrow(df4), size = nrow(df4) * 0.75)
idx_y <- which(colnames(df4) == 'income')

#split train and test data set
## Train / Test Split
X_train <- df4[idx, -idx_y] 
X_test <- df4[-idx, -idx_y]
Y_test <- df4[-idx, idx_y]  #store target column
Y_train <- df4[idx, idx_y]  #store target column

summary(X_train)
glimpse(X_train)

install.packages("randomForest")
library(randomForest)
# Random forest model
rf_model <- randomForest(Y_train ~ ., data = X_train)

#predict model with test data set
rf.prd <- predict(rf_model, newdata = X_test)

actual <- Y_test

# confusion matrix
cm <- table(actual, rf.prd)
cm

#evaluate accuracy
accuracy <- mean(rf.prd == actual)
print(paste("Accuracy of Decision Forest model with Balanced Data set: ", round(accuracy*100,2))) 


#########################################################################

# 6. Predict accuracy on split test data set using balanced data set

# Choose a protected variable (race in this example)
protected_var <- "race"

# Split the test set into subsets by the protected variable
test_groups <- split(X_test, X_test[[protected_var]])


glimpse(Y_train)   # make sure "income" column is "factor"
# Build a Random forest model
rf_model <- randomForest(Y_train ~ ., data = X_train)

# Predict income on each subset of the test set
rf_prediction <- lapply(test_groups, function(x) predict(rf_model, newdata = x))

# Calculate prediction accuracy on each subset
rf_accuracy <- sapply(rf_prediction, function(x) sum(x == x[1])/length(x))
print(paste("Accuracy on each subset as protected variable is race :"))
rf_accuracy

# convert accuracy into percentage
accuracy_percentage <- rf_accuracy*100
print(paste("Accuracy on each subset as protected variable is race :"))
accuracy_percentage


# create bar plot with colors
barplot(accuracy_percentage, main = "Accuracy on Balanced Dataset w.r.t Race", 
        col = c("pink", "red", "yellow", "orange", "green"), ylab = "Accuracy (%)", ylim = c(0, 100))


#========================================================================================================













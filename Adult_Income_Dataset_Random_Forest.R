##Data set source: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
## Reference: https://www.kaggle.com/code/rohitamalnerkar/adult-dataset-income-prediction/notebook

#======================================================================================================================
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

#=====================================================
# Assign column names
colnames(data) <- c("age", "workclass", "fnlwgt", "education", "education.num", "marital.status", 
                    "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",
                    "hours.per.week", "native.country", "income")
#====================================================

# Count the number of people in each income group
income_counts <- table(data$income)

# check income distribution with par plot
barplot(income_counts, main = "Income Classification", col = "green", 
        ylab = "# of people", xlab = "Income", ylim = c(0, max(income_counts) * 1.1))

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

summary(data_new)
glimpse(data_new)

# Convert to a factor variable
data_new$income <- factor(data_new$income)
glimpse(data_new)
summary(data_new)

#above code creates three irrelevant columns in the end of data set, now drop the columns
data_new <- data_new %>%                     # Drop unwanted columns ("index", "salary" and "salary_currency")         
  select(-c(workclass_imp, occupation_imp, native.country_imp))  

# check number of rows and columns in data set
df <- data_new
dim(df) 

# check number of people based on "workcalss"
# bar plot shows that "people work in private sector earn more money" than others
barplot(table(df$workclass),main = 'Income Classification w.r.t workclass',
        col='yellow',ylab ='# of people')


# Convert income variable to binary
#df$income <- ifelse(df$income == "<=50K", 0, 1)


# Choose a protected variable (race in this example)
protected_var <- "race"

# Split data into training (75%) and testing sets (25%)
set.seed(123)
train_idx <- sample(1:nrow(df), size = 0.7*nrow(df), replace = FALSE)
train <- df[train_idx, ]
test <- df[-train_idx, ]

# Split the test set into subsets by the protected variable
test_groups <- split(test, test[[protected_var]])


glimpse(train)   # make sure "income" column is "factor"
# Build a Random forest model
rf_model <- randomForest(income ~ ., data = train)

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



#======================================================================
# Build a neural network model
library(neuralnet)

# save train and test data set into different name to perform neural network model
nn_train <- train
nn_test <- test
nn_test_groups <- test_groups
#=============================
# convert all variables in train to numeric format
nn_train <- sapply(nn_train, as.numeric)

sapply(nn_train, is.numeric)
non_numeric_cols <- names(nn_train)[!sapply(nn_train, is.numeric)]  # give column names which are not numeric
#==============================
glimpse(train)
glimpse(nn_train)

#covert categorical variable to numeric variable

unique(nn_train$race)
nn_train$race <- as.numeric(factor(nn_train$race, labels = c(0, 1, 2, 3, 4)))
nn_train$education <- as.numeric(factor(nn_train$education, labels = c(0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15)))
nn_train$native.country <- as.numeric(factor(nn_train$native.country, labels = c(0, 1,2, 3,4,5,6,7,8,9,10, 11, 12, 13,14,15,16,17,18,19,20,
                                                                                 21,22, 23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39)))
nn_train$educatio<-NULL

str(nn_train)

# Check current factor levels
levels(nn_train$income)

# Set correct factor levels
nn_train$income <- factor(nn_train$income, levels = c("<=50K", ">50K"))

#======================== upto this its working ==================

# train neural network with one-hot encoded variables
nn_model <- neuralnet(income ~ ., data = nn_train, hidden = 5)

# Predict income on each subset of the test set
nn_prediction <- lapply(nn_test_groups, function(x) predict(nn_model, newdata = x)$net.result)

# Convert predicted probabilities to binary predictions
nn_prediction <- lapply(nn_preds, function(x) ifelse(x >= 0.5, 1, 0))

# Calculate prediction accuracy on each subset
nn_accs <- sapply(nn_preds, function(x) sum(x == x[1])/length(x))

# Plot the accuracy of both models
barplot(rf_accs, main = "Random Forest Accuracy by Race", xlab = protected_var, ylab = "Accuracy")
barplot(nn_accs, main = "Neural Network Accuracy by Race", xlab = protected_var, ylab = "Accuracy")
















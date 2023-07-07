##Data set source: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
## Reference: https://www.kaggle.com/code/yipfafa/r-income-prediction-with-knn-nn-decision-tree


#===============================================================================================
# 1. data cleaning 

#install libraries
install.packages("tidyverse")     
library(tidyverse)

dat <- read.csv(                      # Read csv file to perform following tasks
  file = "adult_income.csv",
  header = TRUE,
  sep = ",",
  dec = ".",
  stringsAsFactors = FALSE
)

# count number of values of "<= 50K" and ">50K"
count <- sum(dat$income == "<=50K")
print(count)

count1 <- sum(dat$income == ">50K")
print(count1)

summary(dat)
glimpse(dat)

# convert all character columns to factor columns
for (col in names(dat)[sapply(dat, is.character)]) {
  dat[[col]] <- as.factor(dat[[col]])
}
glimpse(dat)

# clean the data

# 1. Remove redundant columns: fnlwgt, education, and relationship
dat = dat[,-c(3,4,8)] 

############################################################

# 2. Clean the Outliers by using IQR method
## Cleaning outliers in age ##
summary(dat$age)
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##   17.00   28.00   37.00   38.58   48.00   90.00
Q1_age = 28
Q3_age = 48
IQR_age = Q3_age - Q1_age #IQR = Q3 - Q1
IQR_age  ## 20

# Find lowest value (LowerWhisker = Q1 - 1.5 * IQR_age) 
LowerW_age = Q1_age - (1.5*IQR_age)
LowerW_age ## -2

# Find upper value (UpperWhisker = Q3 + 1.5 * IQR_age)
UpperW_age = Q3_age + 1.5 * IQR_age
UpperW_age  ## 78


# Find observations above 78 (as UpperW_age =78)
dat = subset(dat, age <= 78)

## Cleaning outliers in education.num ##
summary(dat$education.num)
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##    1.00    9.00   10.00   10.08   12.00   16.00
Q1_education.num  = 9
Q3_education.num  = 12
IQR_education.num = Q3_education.num  - Q1_education.num
IQR_education.num   ## 3


# Find lowest value (LowerWhisker = Q1 - 1.5 * IQR_education.num) 
LowerW_education.num  = Q1_education.num - 1.5*IQR_education.num
LowerW_education.num 
## [1] 4.5

# Find upper value: (UpperWhisker = Q3 + 1.5 * IQR_education.num)
UpperW_education.num  = Q3_education.num  + 1.5*IQR_education.num
UpperW_education.num 
## [1] 16.5

# Find observations below 4.5
dat = subset(dat, education.num >= 4.5)

## Cleaning outliers in capital.gain ##
library(ggplot2)
summary(dat$capital.gain)
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##       0       0       0    1109       0   99999
box_plot = ggplot(dat, aes(x=capital.gain))+ geom_boxplot()
box_plot

# remove outlier 99999
dat = subset(dat, capital.gain < 99999)

#################################################

# 3. Reclassifying Categorical Variables on "workclass, marital.status, native.country and occupation
## Change the "?" to Unknown ##
dat$occupation = gsub("?", "Unknown", dat$occupation, fixed = T )
dat$occupation = as.factor(dat$occupation)

dat$workclass = gsub("?", "Unknown", dat$workclass, fixed = T )
dat$workclass = as.factor(dat$workclass)

## Reclassify field values ##
## For marital.status ##
levels(dat$marital.status)
## [1] "Divorced"              "Married-AF-spouse"     "Married-civ-spouse"   
## [4] "Married-spouse-absent" "Never-married"         "Separated"            
## [7] "Widowed"
levels(dat$marital.status)[c(2,3,4)] = 'Married'

## For workclass ##
# Grouping "Federal-gov" "Local-gov", and "State-gov" into "Gov"
levels(dat$workclass)

levels(dat$workclass)[c(1,2,7)] = 'Gov'

levels(dat$workclass)

levels(dat$workclass)[4:5] = 'Self-emp'
levels(dat$workclass)


## For native.country ##
t1 = table(dat$native.country) 
prop.table(t1) 

# Since 90% records are from the US, we group the variable native.country into "non-US" and "US"
levels(dat$native.country)[c(28)] = 'United-States'
levels(dat$native.country)[c(1:27,29:41)] = 'Non-U.S.'
levels(dat$native.country)
## [1] "Non-U.S."      "United-States"

## For occupation 
levels(dat$occupation)

levels(dat$occupation)[c(6,8,9)] = 'Service'
levels(dat$occupation)[c(4,8)] = 'Professional/Managerial'
levels(dat$occupation)[c(1,7)] = 'Administration'

######################################################################

datnorm <- dat

# 4. min-max normalization
for (i in c(1, 3, 8, 9, 10)){
  mindf = min(datnorm[,i])
  maxdf = max(datnorm[,i])
  datnorm[,i] =(datnorm[,i] - mindf)/(maxdf - mindf)
}

View(datnorm)


# save cleaned data set 
write.csv(datnorm, "cleaned_dataset.csv", row.names=FALSE, fileEncoding = "UTF-8", na = ' ')

df <- read.csv(                      # Read csv file to perform following tasks
  file = "cleaned_dataset.csv",
  header = TRUE,
  sep = ",",
  dec = ".",
  stringsAsFactors = FALSE
)

######################################################################################

# split data set 

# copy data set
df2 <- df

# make sample index
set.seed(123)
idx <- sample(1:nrow(df2), size = nrow(df2) * 0.75)
idx_y <- which(colnames(df2) == 'income')

#split train and test data set
## Train / Test Split
X_train <- df2[idx, -idx_y] 
X_test <- df2[-idx, -idx_y]
Y_test <- ifelse(df2[-idx, idx_y] == ">50K", 1, 0)  # convert to binary factor
Y_train <- ifelse(df2[idx, idx_y] == ">50K", 1, 0)  # convert to binary factor

summary(X_train)
glimpse(X_train)

#########################################################################################
# 2. Original cleaned whole data set: Predict overall accuracy of whole test set

library(nnet)

# Neural network model
model_nn <- nnet(Y_train ~ ., data = X_train, size=10, maxit = 500)

# Predict model with test data set
prd_nn <- ifelse(predict(model_nn, newdata = X_test, type="raw") > 0.5, 1, 0)

actual <- Y_test

# Confusion matrix
cm <- table(actual, prd_nn)
cm

# Evaluate accuracy
accuracy <- sum(diag(cm))/sum(cm)
print(paste("Accuracy of Neural Network model: ", round(accuracy*100,2)))   # 85.02 %

########################################################################################

# 3. Split whole test data set into sub groups based on "race": Predict accuracy of subsets

# Choose a protected variable (race in this example)
protected_var <- "race"

# Split the test set into subsets by the protected variable
test_groups <- split(X_test, X_test[[protected_var]])

glimpse(Y_train)   # make sure "income" column is "factor"

# Neural network model
model_nn2 <- nnet(Y_train ~ ., data = X_train, size=10, maxit = 500)

# Predict income on each subset of the test set
nn_prediction <- lapply(test_groups, function(x) predict(model_nn2, newdata = x, type = "raw"))

# Calculate prediction accuracy on each subset
nn_accuracy <- sapply(nn_prediction, function(x) mean(x == x[1]))

# Display accuracy for each subset in the console
for (i in 1:length(test_groups)) {
  cat("Accuracy of subset", i, ":", round(nn_accuracy[i]*100, 2), "%\n")
}

# Display accuracy for each subset in a bar graph
# Display accuracy for each subset in a bar graph with colors
barplot(nn_accuracy*100, xlab = "Subsets", ylab = "Accuracy (%)", main = "Accuracy w.r.t race", col = c("red", "green", "blue", "orange", "purple"))

###############################################################################
# Accuracy of subsets shows bias 

# 4. solving bias problem using ROSE method and make balanced data

# copy original cleaned data set
df3 <- df
glimpse(df3)

# check unique values of column
unique(df3$income)

# Convert binary variables into factors
df3$sex <- as.factor(df3$sex)
df3$native.country <- as.factor(df3$native.country)
df3$income <- as.factor(df3$income)

# convert all character columns to factor columns
for (col in names(df3)[sapply(df3, is.character)]) {
  df3[[col]] <- as.factor(df3[[col]])
}

glimpse(df3)

Classcount = table(df3$income)

# Over Sampling
over = ( (0.6 * max(Classcount)) - min(Classcount) ) / min(Classcount)

# Under Sampling
under = (0.4 * max(Classcount)) / (min(Classcount) * over)

over = round(over, 1) * 100
under = round(under, 1) * 100

install.packages("ROSE")  # install the package
library(ROSE)  # load the package


# Generate the balanced data set
BalancedData <- ROSE(income~., data = df3, seed = 1)$data

#Generate the balanced data set
BalancedData = ROSE(income~., data = df3, seed = 1)$data

# let check the output of the Balancing
BalancedData %>%
  group_by(income) %>%
  tally() %>%
  ggplot(aes(x = income, y = n,fill=income)) +
  geom_bar(stat = "identity") +
  theme_minimal()+
  labs(x="Income", y="Count of Income")+
  ggtitle("Income")+
  geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.9))


###########################################################################################

# 5. use balanced data and split train and test and built the Neural Network model, find overall accuracy

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

library(nnet)
# Neural network model
model_nn4 <- nnet(Y_train ~ ., data = X_train, size=10, maxit = 500)

# Predict model with test data set
prd_nn4 <- ifelse(predict(model_nn4, newdata = X_test, type="raw") > 0.5, 1, 0)

actual <- Y_test

# Confusion matrix
cm <- table(actual, prd_nn4)
cm

# Evaluate accuracy
accuracy <- sum(diag(cm))/sum(cm)
print(paste("Accuracy of Neural Network model with balanced data set: ", round(accuracy*100,2), "%"))

##########################################################################################################

# 6. Predict accuracy on split test data set using balanced data set

# Choose a protected variable (race in this example)
protected_var <- "race"

# Split the test set into subsets by the protected variable
test_groups <- split(X_test, X_test[[protected_var]])

glimpse(Y_train)   # make sure "income" column is "factor"

# Neural network model
model_nn5 <- nnet(Y_train ~ ., data = X_train, size=10, maxit = 500)

# Predict income on each subset of the test set
nn_prediction <- lapply(test_groups, function(x) predict(model_nn5, newdata = x, type = "raw"))

# Calculate prediction accuracy on each subset
nn_accuracy <- sapply(nn_prediction, function(x) mean(x == x[1]))

# Display accuracy for each subset in the console
for (i in 1:length(test_groups)) {
  cat("Accuracy of subset", i, ":", round(nn_accuracy[i]*100, 2), "%\n")
}

# Display accuracy for each subset in a bar graph
# Display accuracy for each subset in a bar graph with colors
barplot(nn_accuracy*100, xlab = "Subsets", ylab = "Accuracy (%)", main = "Accuracy w.r.t race", col = c("red", "green", "blue", "orange", "purple"))



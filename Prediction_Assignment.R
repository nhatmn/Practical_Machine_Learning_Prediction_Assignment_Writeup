# Remove everything in current working library
rm(list = ls())

# Load neccessary library
library("caret")
library("tree")
library("rattle")
library("randomForest")
library("rpart")
library("rpart.plot")

# Read data from "pml-training.csv" and "pml-testing.csv".
trainingOrg = read.csv("pml-training.csv", na.strings=c("", "NA", "NULL"))
testingOrg = read.csv("pml-testing.csv", na.strings=c("", "NA", "NULL"))

# Now see dimension of "pml-training.csv" and "pml-testing.csv".
dim(trainingOrg)
dim(testingOrg)

# Remove variables that have too many NA values.
training.dena <- trainingOrg[ , colSums(is.na(trainingOrg)) == 0]
# Now see dimension of training.dena
dim(training.dena)
# Remove unrelevant variables.
remove = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
training.dere <- training.dena[, -which(names(training.dena) %in% remove)]
dim(training.dere)

# Check the variables that have extremely low variance
zeroVar= nearZeroVar(training.dere[sapply(training.dere, is.numeric)], saveMetrics = TRUE)
training.nonzerovar = training.dere[, zeroVar[, 'nzv']==0]
dim(training.nonzerovar)

# Remove highly correlated variables 90%
corrMatrix <- cor(na.omit(training.nonzerovar[sapply(training.nonzerovar, is.numeric)]))
dim(corrMatrix)
corrDF <- expand.grid(row = 1:52, col = 1:52)
corrDF$correlation <- as.vector(corrMatrix)
levelplot(correlation ~ row+ col, corrDF)
# Remove high correlation variables.
removecor = findCorrelation(corrMatrix, cutoff = .90, verbose = TRUE)
training.decor = training.nonzerovar[, -removecor]
dim(training.decor)

# Split data to training and testing for cross validation.
inTrain <- createDataPartition(y=training.decor$classe, p=0.7, list=FALSE)
training <- training.decor[inTrain,]; testing <- training.decor[-inTrain,]
dim(training)
dim(testing)

# Fit a tree to these data, and summarize and plot it
set.seed(2125) #Birthday date of me and my girlfriend :)
tree.training=tree(classe~., data=training)
summary(tree.training)
plot(tree.training)
text(tree.training, pretty=0, cex =.8)

# Running rpart for the form Caret
modFit <- train(classe ~ ., method="rpart", data=training)
print(modFit$finalModel)

# Prettier plots
fancyRpartPlot(modFit$finalModel)

# Check the performance of the tree on the testing data by cross validation.
tree.pred=predict(tree.training, testing,type="class")
predMatrix = with(testing, table(tree.pred, classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix))
tree.pred=predict(modFit, testing)
predMatrix = with(testing,table(tree.pred, classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix))

# Use Cross Validation to prune the tree
cv.training=cv.tree(tree.training, FUN=prune.misclass)
cv.training
plot(cv.training)

# Suppose that the size of nodes is 19
prune.training=prune.misclass(tree.training, best=19)
# Evaluate this pruned tree on the test data
tree.pred=predict(prune.training, testing, type="class")
predMatrix = with(testing, table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix))

# Random Forests
set.seed(2125)
rf.training=randomForest(classe~., data=training, ntree=100, importance=TRUE)
rf.training
varImpPlot(rf.training,)

# Evaluate this tree on the test data.
tree.pred=predict(rf.training, testing, type="class")
predMatrix = with(testing, table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix))

# Predict the testing data from the website
answers <- predict(rf.training, testingOrg)
# See answers
answers

# Function to write "answers" vector to files
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_", i ,".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
    }
}
# Call the function to write "answers" vector to files
pml_write_files(answers)









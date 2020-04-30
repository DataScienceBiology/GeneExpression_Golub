library(dplyr)
library(caret)
library(pheatmap)
library(caretEnsemble)
library(doParallel)
library(xgboost)

# read data 

training <- read.csv("D:/DataScience/Profile/Kaggle_GeneExpression_Golub/data_set_ALL_AML_train.csv", stringsAsFactors = FALSE, quote = "")
testing <- read.csv("D:/DataScience/Profile/Kaggle_GeneExpression_Golub/data_set_ALL_AML_independent.csv", stringsAsFactors = FALSE, quote = "")
colData <- read.csv("D:/DataScience/Profile/Kaggle_GeneExpression_Golub/actual.csv", stringsAsFactors = FALSE, quote = "")
colData$patient <- paste("X", colData$patient, sep = "")
colData_train <- colData[1:38,]
colData_test <- colData[39:72,]

# clean up the call variables and gene description
trainset <- training %>% select(-contains("call"), -Gene.Description)
trainset <- trainset[,c(1:28,34:39,29:33)]
rownames(trainset) <- trainset$Gene.Accession.Number
trainset <- trainset[,-1]

testset <- testing %>% select(-contains("call"), - Gene.Description)
testset <- testset[,order(names(testset))]
rownames(testset) <- testset$Gene.Accession.Number
testset <- testset[,-1]

# check fro missing values
sum(is.na(trainset))
sum(is.na(testset))# no missing values

# check for normalization
boxplot(trainset, outline = F, col = "green") # there was some normalization done. expression values per sample looks to have same scale. Long tails are due to outliers.

# data transformation
# compare to log for one of the samples

par(mfrow = c(1,2))

hist(trainset[,5], xlab = "gene expression", border = "blue4", col = "green")

hist(log10(trainset+1)[,5], xlab = "gene expression log scale", border = "blue4", col = "red")

# data processing
trainset <- as.data.frame(t(trainset))
testset <- as.data.frame(t(testset))

# select top 100 variable genes
SDs = apply(trainset, 2, sd)
topPreds = order(SDs, decreasing = TRUE)[1:100]
trainset = trainset[,topPreds]

# centering and scaling
Centered <- preProcess(trainset, method = c("center"))
trainset <- predict(Centered, trainset)

# create filter for gighly correlated variables

# corrVar <- preProcess(trainset, method = "corr", cutoff = 0.9)
# remove one of the pair correlated variables
# trainset <- predict(corrVar, trainset)

hist(trainset[,5], xlab = "gene expression", border = "blue4", col = "green") # looks promissing

# add cancer type to dataframe
trainset <- cbind(trainset, colData_train$cancer)
colnames(trainset)[101] <- "Cancer"


# ensemble learning
grid.xgb <- expand.grid(.nrounds = 200, .eta = c(0.01, 0.001, 0.0001),
                        .max_depth = c(2,4), .gamma = 0, .colsample_bytree = 0.05, .subsample = 0.05,
                        .min_child_weight = 1)

grid.rf <- expand.grid(.mtry=50,
                       .min.node.size = 5,
                       .splitrule="gini")
grid.enet <- expand.grid(.alpha = 0.5, .lambda=seq(0.1,0.7,0.05))

registerDoParallel(cores = 3)

set.seed(428)
my_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
                           savePredictions = "final",classProbs = TRUE, allowParallel = TRUE)

set.seed(430)
model_list <- caretList(Cancer~., data = trainset,
                        trControl = my_control,
                        tuneList = list(
                          xgb = caretModelSpec(method = "xgbTree", tuneGrid = grid.xgb),
                          rf = caretModelSpec(method = "ranger", tuneGrid = grid.rf),
                          enet = caretModelSpec(method = "glmnet", tuneGrid = grid.enet)
                        ))


stopImplicitCluster()

# check individual models
model_list$xgb # accuracy 0.711
model_list$rf # accuracy 0.91
model_list$enet # accuracy 0.93

resample <- resamples(model_list)
modelCor(resample) # models are not highly correlated, thus we can go for ensembling the models

dotplot(resample, metric = "Accuracy")# enet provides better accuracy

# ensamble with caretEnsamble(), which performs a linear combination of all models
registerDoSEQ()
ensemble_1 <- caretEnsemble(model_list,
                            metric = "Accuracy",
                            trControl = my_control)

summary(ensemble_1)# the accuracy of ensemble: 0.94

# plot the ensamble
plot(ensemble_1) 

# more specific ensamble using caretStack
ensemble_2 <- caretStack(model_list,
                         method = "glmnet",
                         metric = "Accuracy",
                         trControl = my_control)

print(ensemble_2)#  accuracy = 0.94

plot(varImp(ensemble_2$models$enet), top = 10)




# preparing testset
col_names <- names(trainset)
col_ID <- which(names(testset) %in% col_names)

testset <- testset[,col_ID]
Centered <- preProcess(testset, method = c("center"))
testset <- predict(Centered, testset)




# predict on testset

pred_ensemble1 <- predict(ensemble_1, newdata =  testset)
pred_ensemble2 <- predict(ensemble_2, newdata = testset)

pred_accuracy <- data.frame(ensemble_1 = confusionMatrix(as.factor(colData_test$cancer), pred_ensemble1)$overall[1],
                            ensemble_2 = confusionMatrix(as.factor(colData_test$cancer), pred_ensemble2)$overall[1])
pred_accuracy # the performance of ensemble_1 is 0.91 and ensemble_2 is 0.94 on testset


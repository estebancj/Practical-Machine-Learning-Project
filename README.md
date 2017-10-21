Practical Machine Learning Final Course Project Report
================================================

One thing that people regularly do is quantify how much of a particular activity they do, but 
they rarely quantify how well they do it. In this project your goal will be to use data from 
accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

Background
----------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large 
amount of data about personal activity relatively inexpensively. These type of devices are part 
of the quantified self movement - a group of enthusiasts who take measurements about themselves 
regularly to improve their health, to find patterns in their behavior, or because they are tech 
geeks. One thing that people regularly do is quantify how much of a particular activity they do, 
but they rarely quantify how well they do it. In this project, our goal will be to use data from 
accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to 
perform barbell lifts correctly and incorrectly in 5 different ways. More information is 
available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on 
the Weight Lifting Exercise Dataset).

Data Sources
------------

The training data for this project is available here:  
[<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)  
The test data is available here:  
[<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)  
The data for this project comes from this original source: [<http://groupware.les.inf.puc-rio.br/har>](http://groupware.les.inf.puc-rio.br/har). If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.  

Intended Results
----------------

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.  
1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to \< 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).  
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details.  

Report
----------------

In order to classify the test dataset the following was implemented in R (project.r):

1. Use of the caret library (Machine learning library).
``` r
library(caret)
```

2. Load training and test datasets. 

``` r
training <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")
```
3. Preprocessing steps which includes the elimination of NA values, all the variables that have near zero 
variance and all the variables that do not contribute to the classifier performance. This step is applied 
to the whole training and test datasets.

``` r
## Remove variables that are almost always NA
mostlyNA <- sapply(training, function(x) mean(is.na(x))) > 0.95
training <- training[, mostlyNA==F]
test <- test[, mostlyNA==F]

## Remove variables with nearly zero variance
nzv <- nearZeroVar(training)
training <- training[, -nzv]
test <- test[, -nzv]

## Remove variables that do not contribute to the overall classification process
training <- training[, -(1:5)]
test <- test[, -(1:5)]
```

4. Initialize the classifier based on a Random Forest model using a three cross validation 
process over training subset.

``` r
Parameters <- trainControl(method="cv", number=3, verboseIter=F)
fit <- train(classe ~ ., data=train1, method="rf", trControl=Parameters )
```

5. Print the obtained model to see the overall classifier accuracy.

``` r
print(fit$finalModel)
```

``` r
## Call:
##   randomForest(x = x, y = y, mtry = param$mtry) 
## Type of random forest: classification
## Number of trees: 500
## No. of variables tried at each split: 27
## 
## OOB estimate of  error rate: 0.18%
## Confusion matrix:
##   A    B    C    D    E  class.error
## A 4462    2    0    0    0 0.0004480287
## B    5 3030    2    1    0 0.0026333114
## C    0    5 2733    0    0 0.0018261505
## D    0    0    9 2563    1 0.0038865138
## E    0    1    0    2 2883 0.0010395010
```

From the model creation it can be observed thta the algorithm decid to create 500 
trees, each one with 27 variables associated.

6. Using the previously created model, the unpredicted instances are classified in order to 
obtain the labels for each of the elements in the test dataset.

``` r
## Predict elements on the  test dataset
predictions <- predict(fit, newdata=test)

##To evaluate accuracy
pedictionsEvaluation<-predictions

## Convert predictions to character vector
predictions <- as.character(predictions)
```

7. Save the predictons to a file (predictions.txt).

``` r
## Number of predictions made
n <- length(predictions)

##Save all predictions in a single file
pred <- c()
for(i in 1:n) {
  pred <- c(pred, paste0("problem_id_", i," ", predictions[i],"\n"))
}
write.table(pred, file="predictions.txt", quote=F, row.names=F, col.names=F)
```

8. After use the results (predictions.txt) on the final quiz, extract goldStandar to
evaluate the accuracy.

``` r
goldStandar<-c("B","A","B","A","A","E","D","B","A","A","B","C","B","A","E","E","A","B","B","B")

# show confusion matrix to get estimate of out-of-sample error
print(confusionMatrix(goldStandar, pedictionsEvaluation))
```

``` r
## Confusion Matrix and Statistics
## 
## Reference
## Prediction A B C D E
## A 7 0 0 0 0
## B 0 8 0 0 0
## C 0 0 1 0 0
## D 0 0 0 1 0
## E 0 0 0 0 3
## 
## Overall Statistics
## 
## Accuracy : 1          
## 95% CI : (0.8316, 1)
## No Information Rate : 0.4        
## P-Value [Acc > NIR] : 1.1e-08    
## 
## Kappa : 1          
## Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
## Class: A Class: B Class: C Class: D Class: E
## Sensitivity              1.00      1.0     1.00     1.00     1.00
## Specificity              1.00      1.0     1.00     1.00     1.00
## Pos Pred Value           1.00      1.0     1.00     1.00     1.00
## Neg Pred Value           1.00      1.0     1.00     1.00     1.00
## Prevalence               0.35      0.4     0.05     0.05     0.15
## Detection Rate           0.35      0.4     0.05     0.05     0.15
## Detection Prevalence     0.35      0.4     0.05     0.05     0.15
## Balanced Accuracy        1.00      1.0     1.00     1.00     1.00
```

According to the confusion matrix, the accuracy of the model is one which highlights that the features selected as well
as the classifier are the correct ones to predict this kind of information, however more experiments need to be performed 
in order  to test the model using more test samples from distinct  types of accelerometers.
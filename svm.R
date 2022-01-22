library(e1071)
library(caret)
library(MetricsWeighted)
library(pROC)
library(dplyr)

## wczytanie danych 
doc2vec_downloaded <- read.table(
  file = "doc2vec_word2vec_downloaded.txt"
)

doc2vec <- read.table(
  file = "word2vec_doc2vec.txt"
)

glove <- read.delim("Glove_Nasze_Dane.txt", header = TRUE, sep = ",")
glove_downloaded <- read.delim("GLOVE_WCZYTANE_DANE.txt", header = TRUE, sep = ",")
glove <- subset(glove, select = -X)
glove_downloaded <- subset(glove_downloaded, select = -X)
tfidf <- read.csv("TF-IDF_0.2_0.001.txt")


## przygotowanie danych do treningu

dataset <- read.csv('winemag-data-130k-v2.csv')
classes_distribution <- as.data.frame(table(dataset['points']))
classes <- length(unique(dataset[["points"]]))



data_with_labels <- transform(dataset, group=cut(points,  breaks=c(0,86, 88, 90, 100),
                                      labels=c(1, 2, 3, 4))) # '80 - 86', '87 - 88', '89 - 90', '91 - 100'



labels_count <- do.call(data.frame,aggregate(X~group, data_with_labels, 
                                       FUN=function(x) Count=length(x)))


x <- glove #zmienic nazwe w zaleznosci od rodzaju embedingu
y <- data_with_labels$group
data <- x
data$class <- y

## przygotowanie podzbioru danych 
data_s <- data %>% group_by(class) %>% slice_sample(n=3250)



smp_size <- floor(0.85 * nrow(data_s))
set.seed(123)
train_ind <- sample(seq_len(nrow(data_s)), size = smp_size)

train_data <- data_s[train_ind, ]
test_data <- data_s[-train_ind, ]

smp_size <- floor(0.85 * nrow(train_data))
set.seed(123)
val_ind <- sample(seq_len(nrow(train_data)), size = smp_size)

train_data2 <- train_data[val_ind, ]
val_data <- train_data[-val_ind, ]


## reczne dostrajanie hiperparametrow

# trening
svm_model <- svm(class ~ ., data=train_data2, kernel = 'linear', C = 0.25, scale = TRUE)

# walidacja

predicted_classes <- predict(svm_model, val_data)
cm <- confusionMatrix(val_data$class, predicted_classes)
results <- cm[["byClass"]]

## obliczanie dodatkowych metryk

class_weights = labels_count$X

res_df <- as.data.frame(results)
F1_classes_scores <- res_df$F1
precision_classes_scores <- res_df$Precision
recall_classes_scores <- res_df$Recall
weighted_F1 <- sum(class_weights * F1_classes_scores)/nrow(data)
weighted_precision <- sum(class_weights * precision_classes_scores)/nrow(data)
weighted_recall <- sum(class_weights * recall_classes_scores)/nrow(data)

# test

summary(svm_model)
predicted_classes <- predict(svm_model, test_data)
cm <- confusionMatrix(test_data$class, predicted_classes)
results <- cm[["byClass"]]

## obliczanie dodatkowych metryk 

class_weights = labels_count$X

res_df <- as.data.frame(results)
F1_classes_scores <- res_df$F1
precision_classes_scores <- res_df$Precision
recall_classes_scores <- res_df$Recall
weighted_F1 <- sum(class_weights * F1_classes_scores)/nrow(data)
weighted_precision <- sum(class_weights * precision_classes_scores)/nrow(data)
weighted_recall <- sum(class_weights * recall_classes_scores)/nrow(data)



multiclass.roc(test_data$class, as.integer(predicted_classes)) #https://rdrr.io/cran/pROC/man/multiclass.html

#https://cran.r-project.org/web/packages/multiROC/multiROC.pdf


## automatyczne dostrajanie parametrów SVM 
tc <- tune.control(cross = 5)
# Cs = c(.1,.25,.5,1,1.5,2)
# gammas = c(.1,.25,.5,1,1.5,2)
Cs = c(2^-2, 2^-1, 2^0, 2^1, 2^5, 2^8)
gammas = c(2^-3, 2^-1, 2^0, 2^1, 2^3)

svm_tune <- tune(svm, train.x=train_data[,1:ncol(train_data)-1], train.y=train_data$class, 
                 kernel="radial", ranges=list(cost=Cs, gamma=gammas), tunecontrol = tc)


svm_model <- svm(class ~ ., data=train_data2, cost = 1, gamma = 0.1)

predicted_classes_tuned <- predict(svm_model, test_data)

cm <- confusionMatrix(data_s$class, predict(svm_model))
results <- cm[["byClass"]]

class_weights = labels_count_s$V1

res_df <- as.data.frame(results)
F1_classes_scores <- res_df$F1
precision_classes_scores <- res_df$Precision
recall_classes_scores <- res_df$Recall
weighted_F1 <- sum(class_weights * F1_classes_scores)/sum(labels_count_s$V1)
weighted_precision <- sum(class_weights * precision_classes_scores)/sum(labels_count_s$V1)
weighted_recall <- sum(class_weights * recall_classes_scores)/sum(labels_count_s$V1)


svm_tune <- best.tune(svm, train.x=data_s[,1:ncol(data_s)-1], train.y=data_s$class, 
                 kernel="radial", ranges=list(cost=Cs, gamma=gammas), tunecontrol = tc)


## R dataframe to csv
unclass(data$class)
data$class = as.double(data$class)

write.csv(data,"R_glove.csv", row.names = FALSE)


  

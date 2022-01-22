library(e1071)
library(caret)
library(MetricsWeighted)
library(pROC)
doc2vec_downloaded <- read.table(
  file = "doc2vec_word2vec_downloaded.txt"
)
head(doc2vec_downloaded)

doc2vec <- read.table(
  file = "word2vec_doc2vec.txt"
)
head(doc2vec)

#preparing data

dataset <- read.csv('winemag-data-130k-v2.csv')
classes_distribution <- as.data.frame(table(dataset['points']))
classes <- length(unique(dataset[["points"]]))



data_with_labels <- transform(dataset, group=cut(points,  breaks=c(0,86, 88, 90, 100),
                                                 labels=c(1, 2, 3, 4))) # '80 - 86', '87 - 88', '89 - 90', '91 - 100'



labels_count <- do.call(data.frame,aggregate(X~group, data_with_labels, 
                                             FUN=function(x) Count=length(x)))


x <- doc2vec
y <- data_with_labels$group
data <- x
data$class <- y

# split data to train and test
split_idx = as.integer(nrow(data)*0.85)
train_data <- data[1:split_idx,]
test_data <- data[(split_idx + 1):nrow(data),]


data_s <- data[1:1000,]

labels_count_s <- do.call(data.frame,aggregate(V1~class, data_s, 
                                               FUN=function(x) Count=length(x)))


nb_0 <- naiveBayes(data_s[,-ncol(data_s)], data_s$class, laplace = 0)
pred_nb_0 <- predict(nb_0, data_s[,-ncol(data_s)])
cm <- confusionMatrix(data_s$class, pred_nb_0)
results <- cm[["byClass"]]


##wlasciwy trening

nb_0 <- naiveBayes(train_data[,-ncol(train_data)], train_data$class, laplace = 0)
pred_nb_0 <- predict(nb_0, test_data[,-ncol(test_data)])
cm <- confusionMatrix(test_data$class, pred_nb_0)
results0 <- cm[["byClass"]]


nb_10 <- naiveBayes(train_data[,-ncol(train_data)], train_data$class, laplace = 10)
pred_nb_10 <- predict(nb_10, test_data[,-ncol(test_data)])
cm <- confusionMatrix(test_data$class, pred_nb_10)
results10 <- cm[["byClass"]]

nb_1000 <- naiveBayes(train_data[,-ncol(train_data)], train_data$class, laplace = 100000)
pred_nb_1000 <- predict(nb_1000, test_data[,-ncol(test_data)])
cm <- confusionMatrix(test_data$class, pred_nb_1000)
results1000 <- cm[["byClass"]]

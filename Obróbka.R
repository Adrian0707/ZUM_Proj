df <- read.csv(
  file = 'winemag-data-130k-v2.csv',
  sep = ','
)

df2 <- df[,colSums(is.na(df["description"]))<nrow(df)]
df2[["description"]] <- tolower(df2[["description"]])

library(stringr)
#df2[["description"]] <-str_replace(gsub("([^a-z1-9.%'])", " ", df2[["description"]]), "B", "b") # Dziwne znaczki
df2[["description"]] <-str_replace(gsub("([^a-z'])", " ", df2[["description"]]), "B", "b") # Dziwne znaczki
#df2[["description"]] <-str_replace(gsub("\\.+(\\s+|$|[a-z])", " ", df2[["description"]]), "B", "b") # "." bez wycinania w liczbach
df2[["description"]] <-str_replace(gsub("\\s+", " ", df2[["description"]]), "B", "b") # powielone Spacje
df2[["description"]] <-str_replace(gsub("\\s$", "", df2[["description"]]), "B", "b") # koñcowe Spacje

#Sprawdzanie
df2[29996,"description"]
df[29996,"description"]
df2[502,"description"]
df[502,"description"]

#wektory

library(text2vec)
library(data.table)
library(magrittr)

# ***Wczytanie danych przyk³adowych wiki
text8_file = "~/text8"
if (!file.exists(text8_file)) {
  download.file("http://mattmahoney.net/dc/text8.zip", "~/text8.zip")
  unzip ("~/text8.zip", files = "text8", exdir = "~/")
}
wiki = readLines(text8_file, n = 1, warn = FALSE)

# # ***podzia³ danych (Przy uczeniu na naszych danych)
# all_ids = movie_review$id
# train_ids = sample(all_ids, 4000)
# test_ids = setdiff(all_ids, train_ids)
# train = movie_review[J(train_ids)]
# test = movie_review[J(test_ids)]

#*******************************************************************
# Robienie s³ownika
#*******************************************************************
# #***robienie s³ownika funkcje (Przy naszych danych)
# prep_fun = tolower # tego nie potrzebujemy
# tok_fun = word_tokenizer

# # ***robienie s³ownika klasycznie
# 
# it_train = itoken(train$review, 
#                   preprocessor = prep_fun, 
#                   tokenizer = tok_fun, 
#                   ids = train$id, 
#                   progressbar = FALSE)
# vocab = create_vocabulary(it_train)

# # ***robienie s³ownika z wykorzystaniem tokenóW (bardziej u¿yteczne, tokeny s¹ spoko- matrix w kolumnie 'value' ma listê s³óW) 
# 
# train_tokens = tok_fun(prep_fun(train$review))# prep_fun(...) wywalony z wnetrza tok_fun
# it_train = itoken(train_tokens, 
#                   ids = train$id,
#                   # turn off progressbar because it won't look nice in rmd
#                   progressbar = FALSE)
# vocab = create_vocabulary(it_train)

# ***robienie s³ownika tokeny + Gotowy s³ownik

# Create iterator over tokens
tokens = space_tokenizer(wiki)
# Create vocabulary. Terms will be unigrams (simple words).
it = itoken(tokens, progressbar = FALSE)
vocab = create_vocabulary(it)

# # Robienie s³ownika z oczyszczaniem klasycznie
# 
# stop_words = c("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours")
# t1 = Sys.time()
# vocab = create_vocabulary(it_train, stopwords = stop_words)
# print(difftime(Sys.time(), t1, units = 'sec'))
#************************************************************************
#Oczyszczanie
#************************************************************************
#Oczyszczanie s³ownika automatyczne 1 (Minimalistyczne)

vocab = prune_vocabulary(vocab, term_count_min = 5L)

#oczyszczanie s³ownika automatycznie 2 (bardziej restrykcyjne <- potrzebujemy tego bo mamy uszkodzone s³owa)

pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 10, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
vectorizer = vocab_vectorizer(pruned_vocab)
#************************************************************************
#Robimy DTM
#***********************************************************************
# # ***robimy dtm przy wykorzystaniu nieoczyszczonego s³ownika klasycznie
# 
# vectorizer = vocab_vectorizer(vocab)
# t1 = Sys.time()
# dtm_train = create_dtm(it_train, vectorizer)
# print(difftime(Sys.time(), t1, units = 'sec'))

# ****robimy dtm z oczyszczonym s³ownikiem

t1 = Sys.time()
dtm_train  = create_dtm(it_train, vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))

# Normalizacja

dtm_train_l1_norm = normalize(dtm_train, "l1")

#*******************************************************************************
# TF-IDF (Advanced Bag of Words)
#*******************************************************************************
vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

# definicja tfidf modelu
tfidf = TfIdf$new()
# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
# tfidf modified by fit_transform() call!
# apply pre-trained tf-idf transformation to test data
dtm_test_tfidf = create_dtm(it_test, vectorizer)
dtm_test_tfidf = transform(dtm_test_tfidf, tfidf)


#******************************************************************************
# Robimy DCT
#******************************************************************************
# Use our filtered vocabulary
vectorizer = vocab_vectorizer(vocab)
# use window of 5 for context words
tcm = create_tcm(it, vectorizer, skip_grams_window = 5L)


#******************************************************************************
# GLOVE 
# https://datascience-enthusiast.com/DL/Operations_on_word_vectors.html
# (jakieœ zabawy przegl¹dn¹æ tylko) https://mmuratarat.github.io/2020-03-20/glove
#******************************************************************************
glove = GlobalVectors$new(rank = 50, x_max = 10)
wv_main = glove$fit_transform(tcm, n_iter = 10, convergence_tol = 0.01, n_threads = 8)
dim(wv_main)
wv_context = glove$components
dim(wv_context)
word_vectors = wv_main + t(wv_context)


berlin = word_vectors["paris", , drop = FALSE] - 
  word_vectors["france", , drop = FALSE] + 
  word_vectors["germany", , drop = FALSE]
cos_sim = sim2(x = word_vectors, y = berlin, method = "cosine", norm = "l2") # odleg³oœæ cosinusowa https://cran.r-project.org/web/packages/text2vec/text2vec.pdf str 29
head(sort(cos_sim[,1], decreasing = TRUE), 5)
#******************************************************************************
# Problem do rozwi¹zania !!!!!!!!!!!!!!!!!111
# Tworzenie wektorów - Doc2Vec ?
# (Obiecj¹ce) https://www.bnosac.be/index.php/blog/103-doc2vec-in-r
# https://stats.stackexchange.com/questions/221715/apply-word-embeddings-to-entire-document-to-get-a-feature-vector
# (to ju¿ chyba by³o podobne) https://towardsdatascience.com/understanding-word-embeddings-with-tf-idf-and-glove-8acb63892032
# (Nikt nie odpowiedzia³) https://stackoverflow.com/questions/56439449/quick-way-to-get-document-vector-using-glove
# https://stackoverflow.com/questions/50225323/document-classification-using-word-vectors
# https://stackoverflow.com/questions/47615799/from-word-vector-to-document-vector-text2vec
# https://radimrehurek.com/gensim/models/doc2vec.html
# https://xplordat.com/2018/09/27/word-embeddings-and-document-vectors-part-1-similarity/
# (ca³e ? hmmm) https://lena-voita.github.io/nlp_course/word_embeddings.html
# (ciekawe) https://medium.com/analytics-vidhya/word-embeddings-in-nlp-word2vec-glove-fasttext-24d4d4286a73
# https://www.rama.mahidol.ac.th/ceb/sites/default/files/public/pdf/journal_club/2021/JC_Presenter_WordEmbeddings.pdf
# https://blogs.sap.com/2019/07/03/glove-and-fasttext-two-popular-word-vector-models-in-nlp/
# (hmmm) https://smltar.com/embeddings.html
# (hmmm dyskusja) https://www.quora.com/How-can-I-use-word2vec-or-GLOVE-for-document-classification-in-to-predefined-categories
# (Naukowe) https://aclanthology.org/N18-1043.pdf
# (Naukowe) https://www.researchgate.net/publication/284576917_Glove_Global_Vectors_for_Word_Representation
# (Naukowe) https://www.researchgate.net/publication/351685149_WOVe_Incorporating_Word_Order_in_GloVe_Word_Embeddings
# (Naukowe- poœrednio zwi¹zane) https://www.sciencedirect.com/science/article/pii/S2590177X19300563
# (Glove-emb) https://www.mygreatlearning.com/blog/word-embedding/
# (Smieciowe raczej) https://machinelearningmastery.com/what-are-word-embeddings/
# https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
# https://stackoverflow.com/questions/27470670/how-to-use-gensim-doc2vec-with-pre-trained-word-vectors
# https://stackoverflow.com/questions/36815038/how-to-load-pre-trained-model-with-in-gensim-and-train-doc2vec-with-it?rq=1
# https://stackoverflow.com/questions/45037860/gensim1-0-1-doc2vec-with-google-pretrained-vectors?noredirect=1&lq=1
# https://www.frontiersin.org/articles/10.3389/fdata.2020.00009/full
# https://analyticsindiamag.com/hands-on-guide-to-word-embeddings-using-glove/
# https://towardsdatascience.com/word-embeddings-and-document-vectors-when-in-doubt-simplify-8c9aaeec244e
# 
# Inne
# https://cran.r-project.org/web/packages/fastText/index.html
# https://cran.r-project.org/web/packages/fastTextR/index.html
# https://cran.r-project.org/web/packages/udpipe/index.html
#* https://cran.r-project.org/web/packages/ruimtehol/index.html

# Filmiki
# (hmm tu chyba nie ma tego czego szukamy) https://www.youtube.com/watch?v=1kHvEKc8ikw&list=PLhWB2ZsrULv-wEM8JDKA1zk8_2Lc88I-s&index=4
# Bag Of words ? https://www.youtube.com/watch?v=NFd0ZQk5bR4

#pytonowe 
# https://www.youtube.com/watch?v=YNK-pBXDLzk
# https://www.youtube.com/watch?v=Qsmn9pL5kcU

#*******************************************************************************?????????????

# # Ucz¹ to ju¿ bezpoœrednio na dtm... Nie robi¹ z tego ju¿ nic nowego ... Zwyk³e BAG of words?
# library(glmnet)
# NFOLDS = 4
# t1 = Sys.time()
# glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['sentiment']], 
#                               family = 'binomial', 
#                               # L1 penalty
#                               alpha = 1,
#                               # interested in the area under ROC curve
#                               type.measure = "auc",
#                               # 5-fold cross-validation
#                               nfolds = NFOLDS,
#                               # high value is less accurate, but has faster training
#                               thresh = 1e-3,
#                               # again lower number of iterations for faster training
#                               maxit = 1e3)
# print(difftime(Sys.time(), t1, units = 'sec'))


# Do rozwagi:

# n-gramy analizujemy wiêcej ni¿ jedno s³owo np "Nie lubiê" jako ca³oœæ iloœc s³ów po³¹czonych zale¿y od iloœci przyk³adów
# w s³owniku chyba?

# Feature hashing szybkoœæ
# prosto: https://en.wikipedia.org/wiki/Feature_hashing
# naukowo: http://alex.smola.org/papers/2009/Weinbergeretal09.pdf



## Amazon Fine Food Reviews Analysis using various Machine Learning Models

- Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews
- EDA: https://nycdatascience.com/blog/student-works/amazon-fine-foods-visualization/

The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.

* Number of reviews: 568,454
* Number of users: 256,059
* Number of products: 74,258
* Timespan: Oct 1999 - Oct 2012
* Number of Attributes/Columns in data: 10

## Attribute Information:  

1. Id
2. ProductId - unique identifier for the product
3. UserId - unqiue identifier for the user
4. ProfileName
5. HelpfulnessNumerator - number of users who found the review helpful
6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
7. Score - rating between 1 and 5
8. Time - timestamp for the review
9. Summary - brief summary of the review
10. Text - text of the review

## Objective:
Given a review, determine whether the review is positive (Rating of 4 or 5) or negative (rating of 1 or 2).

**[Q] How to determine if a review is positive or negative?**

**[Ans]** We could use the Score/Rating. A rating of 4 or 5 could be cosnidered a positive review. A review of 1 or 2 could be considered negative. A review of 3 is nuetral and ignored. This is an approximate and proxy way of determining the polarity (positivity/negativity) of a review.

## Data Visualization using PCA and t-SNE

### Reading the data from SQLITE
1. We are going to classify our data using the attribute "SCORE" from our dataframe
2. SCORE > 3 is classified as Positive review, SCORE < 3 is classified as Negative review
3. Since SCORE = 3 is Neutral, we will not consider those reviews in our classiication

### Exploratory Data Analysis

**Data Cleaning: 1. Removing duplicate values**

1. It's neccessary to remove the duplicate values from our data points to get unbiased results.
2. In out dataset it is observed that multiple reviews are with the same values for UserId, ProfileName, Score, Time, Summary and Text.
3. When an user gives a review for a particular flavour of some product, the same review is getting added to all the flavours of that product.
4. In order to avoid redundancy, we are eliminating the duplicate values.

**Data Cleaning: 2. Removing values for which HelpfulnessNumerator is greater than HelpfulnessDenominator**

1. It has been observed that in the below two rows, the value of HelpfulnessNumerator is greater than HelpfulnessDenominator which is not practically possible hence removing them from our data

**Data Cleaning: 3. Removing miscategorized data points**

1. The products "B00004CI84" and "B00004CXX9" belong to movie category. Since the product name contains the word "Juice", it has been miscategorized under food.
2. "B0002YW26E" is a Pest control product. Miscategorized as food.
3. The products "6641040" and "2841233731" are CookBooks.

**Text Preprocessing**: Stemming, stop-word removal and Lemmatization

**Preprocessing Review Text**

Now that we have finished deduplication our data requires some preprocessing before we go on further with analysis and making the prediction model.

Hence in the Preprocessing phase we do the following in the order below:-

1. Begin by removing the html tags
2. Remove any punctuations or limited set of special characters like , or . or # etc.
3. Check if the word is made up of english letters and is not alpha-numeric
4. Check to see if the length of the word is greater than 2 (as it was researched that there is no adjective in 2-letters)
5. Convert the word to lowercase
6. Remove Stopwords
7. Finally Snowball Stemming the word (it was obsereved to be better than Porter Stemming)

After which we collect the words used to describe positive and negative reviews

## 2D Visualization using PCA and t-SNE:
BoW unigram, BoW BIGrams, TF-IDF: Unigrams, TF-IDF: Bigrams, Average Word2vec, TF-IDF Word2vec

1. When comparing all the plots, 2D visualization using T-SNE BOW and TF-IDF(Unigram) is better compared to others.
2. However none of the plots linearly seperates both positive and negative points using a plane

## Applying KNN brute force and kd-tree: 
BOW, TFIDF, Average Word2vec, TF-IDF Word2vec


| Vectorizer |  Model  | Hyper parameter | Train AUC | Test AUC |
|------------|---------|-----------------|-----------|----------|
|    BOW     |  Brute  |        11       |    0.96   |   0.7    |
|   TFIDF    |  Brute  |        3        |    0.99   |   0.61   |
|    W2V     |  Brute  |        13       |    0.98   |   0.66   |
|  TFIDFW2V  |  Brute  |        11       |    0.98   |   0.61   |
|    BOW     | kd_tree |        9        |    0.97   |   0.7    |
|   TFIDF    | kd_tree |        11       |    0.96   |   0.56   |
|    W2V     | kd_tree |        9        |    0.98   |   0.56   |
|  TFIDFW2V  | kd_tree |        9        |    0.98   |   0.54   |


## Applying Naive Bayes:
BOW, TFIDF


| Vectorizer | Hyper parameter | Train AUC | Test AUC |
|------------|-----------------|-----------|----------|
|    BOW     |      10000      |    0.99   |   0.91   |
|   TFIDF    |      10000      |    0.99   |   0.89   |

## Applying Logistic Regression:
BOW, TFIDF, Average Word2vec, TF-IDF Word2vec


| Vectorizer | Regularization | Hyper parameter | Train AUC | Test AUC |
|------------|----------------|-----------------|-----------|----------|
|    BOW     |       L1       |        10       |   0.999   |  0.946   |
|   TFIDF    |       L1       |        1        |   0.999   |  0.953   |
|    W2V     |       L1       |       0.1       |   0.916   |  0.806   |
|  TFIDFW2V  |       L1       |       0.1       |   0.892   |  0.746   |
|    BOW     |       L2       |      0.0001     |   0.999   |  0.924   |
|   TFIDF    |       L2       |      0.0001     |   0.999   |  0.935   |
|    W2V     |       L2       |       0.01      |   0.916   |  0.808   |
|  TFIDFW2V  |       L2       |       0.01      |   0.892   |  0.747   |

## Applying SVM:
BOW, TFIDF, Average Word2vec, TF-IDF Word2vec


|      SVM      | Vectorizer | Regularization | Hyper parameter | Train AUC | Test AUC |
|---------------|------------|----------------|-----------------|-----------|----------|
| Linear Kernal |    BOW     |       L1       |       0.01      |   0.852   |  0.739   |
| Linear Kernal |    BOW     |       L2       |      10000      |   0.999   |  0.909   |
| Linear Kernal |   TFIDF    |       L1       |      0.001      |   0.997   |  0.875   |
| Linear Kernal |   TFIDF    |       L2       |        10       |   0.999   |  0.909   |
| Linear Kernal |  AVG W2V   |       L1       |       0.01      |   0.938   |  0.908   |
| Linear Kernal |  AVG W2V   |       L2       |        1        |   0.929   |  0.911   |
| Linear Kernal | TFTDF W2V  |       L1       |      0.001      |   0.919   |  0.909   |
| Linear Kernal | TFTDF W2V  |       L2       |      0.001      |   0.919   |   0.91   |
|   RBF Kernal  |    BOW     |       -        |        1        |   0.992   |  0.883   |
|   RBF Kernal  |   TFIDF    |       -        |        1        |   0.993   |  0.888   |
|   RBF Kernal  |  AVG W2V   |       -        |        1        |   0.956   |  0.888   |
|   RBF Kernal  | TFIDF W2V  |       -        |        1        |   0.945   |  0.869   |

## Applying Decision Trees:
BOW, TFIDF, Average Word2vec, TF-IDF Word2vec


| Vectorizer | min_samples_split | max_depth | Train AUC | Test AUC |
|------------|-------------------|-----------|-----------|----------|
|    BOW     |        500        |     10    |   0.911   |  0.821   |
|   TFIDF    |        500        |     10    |   0.955   |  0.778   |
|  AVG W2V   |        100        |    100    |   0.972   |  0.818   |
| TFIDF W2V  |        100        |     50    |   0.965   |  0.789   |


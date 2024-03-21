Sentiment Analysis on Amazon product reviews
First Name	Last Name	Email address
Sanjana	Surendra	ssurendra@hawk.iit.edu

Table of Contents
1. Introduction	2
2. Data	2
3. Problems and Solutions	3
4. KDD	3
4.1. Data Processing	3
4.2. Text Processing	4
4.3. Data Mining Methods and Processes	4
5. Evaluations and Results	4
5.1. Evaluation Methods	4
5.2. Results and Findings	5
6. Conclusions and Future Work	6
6.1. Conclusions	6
6.2. Limitations	6
6.3. Potential Improvements or Future Work	6


 
1.	Introduction
Sentiment analysis is the study of subjective information in a statement, such as opinions, appraisals, feelings, or attitudes about a topic, person, or entity. Text analytics, natural language processing, and machine learning approaches are used in sentiment analysis to assign sentiment scores to topics or categories.
Sentiment analysis is quickly becoming a crucial tool for monitoring and understanding sentiment in all forms of data, as humans communicate their thoughts and feelings more openly than ever before. Machines learn how to detect sentiment without human intervention by training machine learning models with samples of text input and the target variable. Classifying the polarity of a given text, phrase, or feature, whether the conveyed perspective is positive, negative, or neutral is the basic task of sentiment analysis, which can also be referred as Graded Sentiment Analysis.

2.	Data

The dataset Amazon Product Reviews was found on Kaggle. https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews.
The dataset contains consumer reviews on different amazon products. There were 568454 records and 10 features in all. The dataset's domain is amazon.com which is in Excel spreadsheet format and is 300.9 MB in size. The features are as follows:
•	Id – Unique identity for every record.
•	Product Id – Every product has its own unique identifier. It has 74258 values in it.
•	User Id – Every user has its own unique identifier. It has 256059 values in it. 
•	Profile Name - Profile name is users’ identification.
•	Helpfulness Numerator – The Helpfulness of a rating refers to how the review has benefited other users. The number of users who have offered favorable comments is represented by the Helpfulness Numerator.
•	Helpfulness Denominator - The Helpfulness of a rating refers to how the review has benefited other users. The number of users who have offered both positive and negative feedback is represented by the Helpfulness Denominator.
•	Score – ranking of product based on users’ assessment on a scale of 1 through 5
•	Time – The timeline when the rating was delivered.
•	Summary – The gist of the review.
•	Text – The detailed description on why the user likes or dislikes the product.

3.	Problems and Solutions

Today every organization collect feedback from the brands to learn what makes customers happy or frustrated, so that they can tailor products and services to meet their customers’ needs.  Analyzing customer feedback manually is a waste of resources. Using sentiment analysis to automatically analyze open-ended responses could help discover why customers are happy or unhappy. Implementing sentiment analysis by using text data mining on user reviews to draw insights for upselling opportunities in an organization is a solution.  	 
•	To deal with unprocessed strings, clean and prepare data using Text Processing.
•	To deal with text data, use feature extraction by CountVectorizer and TFIDF Vectorizer.
•	To learn about sentiments from vast quantities of data using different classifiers.
•	To learn about the efficiency of the model, evalute the models in different settings.

4.	KDD

4.1. Data Processing
•	Working with such a huge dataset was impossible due to computational constraints, thus it was limited to 50,000 rows.
•	Dealing with missing values in the column Profile Name and Summary. Unknow User is used as a global constant to fill the missing values in the column Profile Name. Remove the rows in column Summary that have missing values.
•	The column Score having five different values was mapped to sentiment positive, neutral, and negative. Score 1 and 2 was assigned as negative sentiment (-1), score 3 was assigned as neutral sentiment (0) and score 4 and 5 was assigned as positive sentiment (1). This column was named as Class Label which is the target variable.
•	Columns that had no meaning to perform sentiment analysis were eliminated. Only the columns Text, Summary and Class Label were retained.
•	Columns Summary and Text were concatenated. They don't convey various meanings because Summary is only a condensed interpretation of Text. This column was named as Reviews.

4.2. Text Processing
•	HTML tags, hyperlink, markup, encoded characters, numbers, special characters were removed from the column Reviews.
•	Additional white spaces were stripped.
•	All characters were converted to lowercase.
•	Stopwords which don’t add much meaning to a sentence were removed.
•	Stemming where suffixes or prefixes of a word is reduced while retaining its original meaning was carried out.
•	Lemmatization where the word is returned to its base word by use of vocabulary was done.

4.3. Data Mining Methods and Processes
The following Data Mining methods were used with differnet parameters and results were evaluated for best settings.
•	K-nearest neighbors
•	Naive Bayes
•	Decision Tree
•	Random Forest
•	Logistic Regression
•	Support Vector
•	Multi-Layer Perceptron
•	Gradient Descent
•	eXtreme Gradient Boosting
5.	Evaluations and Results

5.1. Evaluation Methods
As classification models cannot process text data, feature extraction using the below methods convert text data into word vectors.
•	Count Vectorizer only counts the number of times a word appears in the document which results in biasing in favor of most frequent words. This ends up in ignoring rare words which could have helped is in processing our data more efficiently.
•	TF-IDF Vectorizer considers the overall document weightage of a word. It helps in dealing with most frequent words and rare words. TF-IDF Vectorizer weights the word counts by a measure of how often they appear in the documents. 
Both methods contain many features. Performing dimensionality reduction using PCA requires dense matrix and TF-IDF produces sparse matrix. So, the converison from sparse to dense requires 25GB memory. So, carrying out dimenionality reduction on TF-IDF vetors using TruncatedSVD solves the problem.
Since there was an imbalance issue making the dataset more biased towards positive sentiment, Over Sampling and Under Sampling was carried out, Random Over Sampling gave better results.
Due to computational reasons, only hold-out evaluation was performed. The evaluation metrics Accuracy, Precision, Recall and F1-score is calculated.

5.2.	Results and Findings
These are the results for all nine supervised learning models for Count Vectorizer and TFIDF-Vectorizer.
Classifiers	Count Vectorizer	TFIDF Vectorizer
	TFIDF Vectorizer
(With TruncatedSVD)
K-nearest neighbors 	K: 1 
Accuracy: 0.768 
F1 score: 0.752	K: 1 
Accuracy: 0.753 
F1 score: 0.747	K: 1 
Accuracy: 0.752
F1 score: 0.740
Naive Bayes	Accuracy: 0.831
F1 score: 0.796	Accuracy: 0.779
F1 score: 0.683 	Accuracy: 0.782
F1 score: 0.687
Decision Tree	Accuracy: 0.777
F1 score: 0.680	Accuracy: 0.779 
F1 score: 0.683 	Accuracy: 0.782 
F1 score: 0.687 
Random Forest	Accuracy: 0.827 
F1 score: 0.786 	Accuracy: 0.826 
F1 score: 0.784 	Accuracy: 0.815 
F1 score: 0.778 
Logistic Regression	Accuracy: 0.867 
F1 score: 0.854	Accuracy: 0.856 
F1 score: 0.827 	Accuracy: 0.824 
F1 score: 0.785
Support Vector	Accuracy: 0.862 
F1 score: 0.846	Accuracy: 0.816 
F1 score: 0.757 	Accuracy: 0.814 
F1 score: 0.758
Multi-Layer Perceptron	Accuracy: 0.859 
F1 score: 0.829	Accuracy: 0.833 
F1 score: 0.786	Accuracy: 0.782
F1 score: 0.687
Gradient Descent	Accuracy: 0.865
F1 score: 0.851 	Accuracy: 0.846 
F1 score: 0.805	Accuracy: 0.821
F1 score: 0.787
eXtreme Gradient Boosting	Accuracy: 0.771 
F1 score: 0.729	Accuracy: 0.777 
F1 score: 0.735	Accuracy: 0.768 
F1 score: 0.728

Results after balancing the dataset and dimensionality reduction: The total number of samples initially before balancing was {1: 39079, -1: 7164, 0: 3752}. After UnderSampling, the number of samples for each sentiment was {1: 3752, -1: 3752, 0: 3752} for which SVM using TFIDFVectorizer gave best results with an accuracy of 0.64. After OverSampling, the number of samples for each sentiment was {1: 39079, -1: 39079, 0: 39079} for which Random Forest using TFIDFVectorizer gave best results with an accuracy of 0.97. 

6.	Conclusions and Future Work

6.1. Conclusions
Using the Amazon Product Review dataset to train a variety of machine learning models, and based on performance measures such as accuracy, precision, recall, and F1 score, it was discovered that the Logistic regression model was effective in classifying review sentiments into one of three categories: positive (1), negative (-1), or neutral (0). Furthermore, if the imbalance issue was addresed using Oversampling, Random Forest model was the most effective in classifying the sentiments.

6.2.	Limitations
•	Sentiment analysis can only be done in English, which creates a linguistic barrier.
•	Because the models only learn from words, detecting sarcasm in sentiment analysis is extremely difficult without a thorough understanding of the circumstance.
•	Negation, which are a way of reversing the polarity of words, phrases, and even sentences, are incompatible with sentiment analysis.
•	Sentiment polarity causes key information to be omitted, causing the overall result to be misleading since it conceals valuable information.

6.3. Potential Improvements or Future Work
•	Perform product specific sentiment analysis and remove the vendor with products with only negative feedback.
•	Recommendation systems could be built to fulfil the following criteria
o	If a product has positive sentiment, it could be recommended to other users.
o	Products that fall in the same categories can be recommended to the customer, if one of the products has positive feedback by same customer.
•	Furthermore, Feature Engineering can be performed to know if a review is fake.
![image](https://github.com/SanjanaSurendra1997/SentimentAnalysisNLP/assets/113851128/7c16b53f-8431-4332-aa1c-5b9156e7b696)

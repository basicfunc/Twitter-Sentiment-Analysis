# Twitter-Sentiment-Analysis
This project performs sentiment analysis on text data using logistic regression. The goal of sentiment analysis is to classify text as having positive or negative sentiment. In this project, we will use the Sentiment140 dataset, which consists of 1.6 million tweets, each labeled with either positive or negative sentiment.

# Dataset
The Sentiment140 dataset is used for sentiment analysis tasks, especially for binary classification of positive and negative sentiment. The dataset consists of 1.6 million tweets, each labeled with either 'positive' or 'negative' sentiment. The tweets were collected using the Twitter API and cover the period from April to June 2009.

Each tweet in the dataset is represented as a single line of text, with the following fields separated by commas:

- `target`: the sentiment of the tweet (0 = negative, 2 = neutral, 4 = positive)
- `id`: the ID of the tweet
- `date`: the date and time the tweet was created
- `flag`: a query label used to filter out spam tweets
- `user`: the username of the Twitter user who posted the tweet
- `text`: the text content of the tweet

For sentiment analysis tasks, the `target` field is typically used as the target variable, with 0 indicating negative sentiment and 4 indicating positive sentiment. The text field is used as the input data, with machine learning models trained to predict the sentiment label based on the text content of the tweets.

The dataset can be found here: https://www.kaggle.com/datasets/kazanova/sentiment140

# Preprocessing
The first step in this project is to preprocess the text data. The preprocessing steps include removing URLs and usernames, removing punctuation, converting text to lowercase, removing stopwords, and tokenizing the text using regular-expression and removing stopwords. 
We also tokenize the text into a list of words using the `word_tokenize()` function from the `nltk` library.

# Model Building
We use logistic regression to build a sentiment analysis model. Logistic regression is a classification algorithm that is used to predict the probability of a binary outcome (i.e. positive or negative sentiment in our case). The logistic regression algorithm is trained on a training set and then tested on a separate testing set.

Before building the model, we convert the text data into a matrix of token counts using the CountVectorizer class from the `scikit-learn` library. We then convert this matrix of token counts into a matrix of term frequencies times inverse document frequency (tf-idf) values using the `TfidfTransformer` class from `scikit-learn`.

We then split the dataset into training and testing sets using the `train_test_split()` function from `scikit-learn`. The training set is used to train the logistic regression model, and the testing set is used to evaluate the performance of the model.

Once the logistic regression model is trained, we use it to make predictions on the testing set. We then compute the accuracy of the model using the `accuracy_score()` function from `scikit-learn`.

# Conclusion
In this project, we demonstrated how to perform sentiment analysis using logistic regression. We used the Sentiment140 dataset, preprocessed the text data, built a logistic regression model, and evaluated its performance. Logistic regression is a simple and effective algorithm for sentiment analysis tasks, and it can be used to classify text as having positive or negative sentiment.
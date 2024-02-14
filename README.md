# Naive-Bayes-Based-Spam-Detection-Classifier
The dataset for this can be obtained from the UCI Machine Learning Repository. Specifically, use the “SMS Spam Collection Data Sets”, which can be found at: 
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

Implementation Basis: Modelling the class conditional probabilities as independent Bernoulli distributions.

**Algorithm**
The algorithm has been implemented using Naïve-Bayes Classification theory. The likelihood or the class conditional probability is calculated from the training data for each word independent of other words.
The likelihood of a certain word is given by the ratio of number of times that particular word was found in the dataset that was labelled as one of the classes to the total no of instances of that particular class.
The posterior probability is then calculated for each sentence as an independent Bernoulli distribution. This posterior probability is then compared for both the classes –“ham” & “spam” and the classified according to greater posterior probability.
The testing data is then used to test the data. The posterior probability is then calculated for the testing dataset based in the new vector and the likelihood the model learned from the training.Again, a comparison of the posterior probabilities for the two classes is done to predict the classification of the given vector.

In general , the algorithm used can be summarized as the following steps :
  1. Load the SMSSpamCollection dataset
  2. Preprocess the dataset by removing all characters other than A-Z, a-z, 0-9, $, and
  3. Randomly split the data into train and test sets, with an 80/20 split
  4. Create a vocabulary of the most frequent words in the training set
  5. Convert each sentence in the dataset into a binary vector whose length is the size of
  the vocabulary, with each dimension in this vector being 1 if and only if the
  corresponding word is present in the sentence being considered
  6. Calculate the likelihood of each word for each class (spam or ham) in the training set
  7. Calculate the posterior probability of each class for each sample in the training set
  using the Naive Bayes formula
  8. Classify the samples in the test set using the Naive Bayes formula and the posterior
  probabilities calculated in step 7

**Assumptions
**Only the words that appear in the vocabulary are used for classification
• The words are independent of each other
• The prior probabilities for each class are equal
• There are no missing values in the dataset
• There is no overfitting

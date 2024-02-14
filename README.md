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

**Limitations**
  • The algorithm is only effective for short messages such as SMS, and may not perform as well for longer texts
  • The algorithm relies on the quality of the pre-processing step. Incorrect pre- processing could lead to poor performance
  • The algorithm assumes that words are independent of each other, which may not always be the case in real-world scenarios
  • The algorithm may not work well for new words that are not in the vocabulary
  • The algorithm may not work well if the prior probabilities for each class are not equal
  • The algorithm may suffer from overfitting if the training set is too small or if the vocabulary is too large

**Implementation details
**

    1. Firstly, the data is read into a dataframe called "df", and the column names are set to ['Ham_or_Spam', 'SMS'].
    2. The 'SMS' column of the dataframe is then preprocessed by splitting it on the regex pattern nacc_char = '[^A-Za-z0-9$\s]', removing any unwanted characters, and joining it again as a string.
    3. The data is then split into training and testing datasets using an 80-20 split ratio. To ensure that both the spam and ham classes are equally represented in the datasets, the df dataframe is first split into two datasets, df_spam and df_ham, representing spam and ham classifications, respectively. These are concatenated using pandas index slicing to form the training and testing dataframes, training_data and testing_data. Finally, the index for both dataframes is reset to start from 0 to make manipulation easier.
    4. To choose the top 'k' most frequently occurring words to create a vocabulary, the SMS column of the training_data dataframe is converted to lowercase, split on whitespace, and counted for each unique word using a for loop. The resulting data is stored in a dictionary called "vocabulary", where the keys are the unique words and the values are their respective counts. This dictionary is then sorted in descending order of count values and stored in another dictionary called "vocabulary_sorted". A function called "topk" is used throughout the program to fetch the top 'k' words according to their count values.
    5. For the multi-hot spot vector, a matrix called "bin_vec" is defined with all zeros initially, and a length equal to the number of rows in the training_data dataframe. A loop is then run over the training_data['SMS'] column to check for the presence of any of the top 'k' vocabulary words. If a match is found, the bin_vec matrix is updated with a 1 in the corresponding position, indicating the presence of one of the top 'k' vocabulary words in that sentence. The result is a binary vector of length 'k' for each sub-list in the bin_vec matrix.
    6. Naive Bayes classifier:
            a.Firstly, the prior probabilities are calculated as the ratio of the count of 'ham' or 'spam' instances in the dataset to the total number of instances in the dataset. These prior                  probabilities are stored in a dictionary called "prior", with the class names 'ham' and 'spam' as keys and their respective probabilities as values.
            b. The likelihood or class conditional probabilities are then calculated by looping over the entire bin_vec matrix and counting the number of occurrences of a specific feature index                 (denoted by its 'w' position in the topk function returned list). This is done for each position of the sub-list inside the bin_vec matrix for each class. The ratio of this count 
               to the count of total class counts for each class is calculated as the likelihood of that word given the 'ham' or the 'spam' class.
            c. For posterior probability, a logarithmic function and independent Bernoulli distribution are used, with the probability being the likelihood value calculated as above for each                     vector.
            d. For training prediction, the posterior probability of each class is compared, and the class with the greater probability is chosen as the predicted class.
            e. The code for testing is similar, except that the likelihood is not calculated, as only references from the training set are used for posterior probability calculation.

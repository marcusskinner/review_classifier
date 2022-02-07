# Part 3
The program classifies hotel reviews as real or fake using a Bayes Net classifier. It does this by calculating 
the probabilities P(class|words) = P(class|word_1) * P(class|word_2) * ... P(class|word_n) for each class
and classifying a review to the class with the highest probability. 

## Word Frequency Dictionary and Probabilities
The program starts by going through each review in the training data and adding up the number of times each word
appears in a document for each of the classes, deceptive and truthful. A word is only added at most once for a 
single document. For example, if the word "the" appeared 12 times in document A, 6 times in Document B, and 
23 times in Document 6, the total frequency is 3. This is so that P(word|class) = the probabilty a word appears
at least once in a document. Another dictionary is then made that holds the probability a word appears at least
once in a document for each class, along with the probability the word appears at least once in any document.
The entry for a word might look like {word: [0.3, 0.24, 0.6]}. Another list is made for P(class). In this case,
the probabilities for each class is 0.5, but I wrote the program so that it could scale if needed (that is, we
can add more classes or samples if needed).

## Prediction
Next, the classify method calls the prediction method for each sample in the test data. The prediction method 
computes P(class|words) for each class and returns the argmax of the classes. The probability is computed using
logs so that longer reviews don't cause underflow. It also throws out any word with a probability lower than sigma,
in this case 1/1200 (a word appeared 0 or only 1 time). 

## Challenges
I tried removing punctuation, removing numbers, switching everthing to lowercase, and counting repeated words. Using the
probability of the exsistence of a word (appears at least once) led to the best results. Also, removing punctuation led
the slightly better results. Any other change actually lowered accuracy.

Using the log probability didn't actually lead to any increase in accuracy, but if longer reviews were introduced to the dataset,
we would have to multiply by even more probabilities, potentially causing underflow in the future. So that my model could scale to
any problem of this type, I used log probability.

## Results
The program classified the testing data with an accuracy of <b>95.58%</b>. 

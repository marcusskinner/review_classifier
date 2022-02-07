import sys
from math import log

def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

def word_freq(data):
    """
    Builds a word frequency dictionary from a dataset of reviews. The frequency 
    represents the number of individual documents the word appeared. If a word appeared
    twice in a single document, it is only counted once for the frequency. 
    
    Input: a dictionary of reviews
    Output: a dictinary of the word frequencies
    """
    num_reviews = len(data['objects'])
    classes = data['classes']
    freq_dict = {}

    for i in range(num_reviews):
        text = data['objects'][i]
        words = list(set(text.split(" ")))
        label = data['labels'][i]

        for word in words:
            if word not in freq_dict: freq_dict[word] = [0 for k in range(len(classes))]
            freq_dict[word][classes.index(label)] += 1
            
    return freq_dict

def posterior_probabilities(freq_dict, data):
    """
    Calculates the probabilities P(class), P(word|class) for each word and each class.
    
    Inputs
    ----------------
    freq_dict : dict
        a dictionary of the word frequencies
        
    data      : dict
        the data for all the hotel reviews
    
    Outputs
    ---------------
    word_probs  : dict {word: [P(word|class1), P(word|class2), ... P(word|classn)]}
        a dictinary for all probabilities P(word|class)
        
    class_probs : list [P(class1), P(class2), ... , P(classn)]
        a list of all the class probabilities
    """
    total_samples = len(data['labels'])
    class_samples = [data['labels'].count(label) for label in data['classes']]
    word_probs = {}
    
    for word in freq_dict:
        word_probs[word] = [freq_dict[word][i]/class_samples[i] for i in range(len(class_samples))] + [sum(freq_dict[word])/total_samples]
        
    class_probs = [class_samples[i]/total_samples for i in range(len(class_samples))]
    
    return word_probs, class_probs


def predict(text, word_probs, class_probs, sigma):
    """
    Predicts the class of a single given review. It calculates the entire distribution
    and returns the index of the class with the highest probability.
    
    Inputs
    ----------------
    text : str
        the text for the review to be classified
        
    word_probs : dict dict {word: [P(word|class1), P(word|class2), ... P(word|classn)]}
        a dictionary for all word probabilities
        
    class_probs : list [P(class1), P(class2), ... , P(classn)]
        a list of all class probabilities
        
    Outputs
    ----------------
    class index : int
        the index of the predicted class
    """
    n_classes = len(class_probs)
    words = list(set(text.split(" ")))
    distribution = [1 for i in range(n_classes)]
    
    for word in words:
        if word in word_probs:
            for c in range(n_classes):
                if word_probs[word][c] > sigma:
                    distribution[c] += log((word_probs[word][c] * class_probs[c]) / word_probs[word][-1])
                          
    return distribution.index(max(distribution))
    
# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)

def classifier(train_data, test_data):
    freq_dict = word_freq(train_data)
    word_probs, class_probs = posterior_probabilities(freq_dict, train_data)
    
    pred = []
    for review in test_data['objects']:
        pred += [test_data['classes'][predict(review, word_probs, class_probs, 1/1200)]]
    return pred

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    results = classifier(train_data, test_data)

    # calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))

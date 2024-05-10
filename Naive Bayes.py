import pandas as pd
from collections import defaultdict
import numpy as np

class NaiveBayesClassifier:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.word_counts = defaultdict(lambda: [0, 0])  # [spam_count, not_spam_count]
        self.total_messages = [0, 0]  # [total_spam_messages, total_not_spam_messages]
        self.total_words = [0, 0]  # [total_spam_words, total_not_spam_words]

    def train(self, X_train, y_train):
        for message, label in zip(X_train, y_train):
            words = message.split()
            for word in words:
                label_index = 1 if label == 'not spam' else 0
                self.word_counts[word][label_index] += 1
                self.total_words[label_index] += 1
            self.total_messages[label_index] += 1

    def predict(self, X_test):
        predictions = []
        for message in X_test:
            words = message.split()
            spam_score = np.log(self.total_messages[0] / sum(self.total_messages))
            not_spam_score = np.log(self.total_messages[1] / sum(self.total_messages))
            for word in words:
                if word in self.word_counts:
                    spam_score += np.log((self.word_counts[word][0] + self.alpha) / (self.total_words[0] + self.alpha * len(self.word_counts)))
                    not_spam_score += np.log((self.word_counts[word][1] + self.alpha) / (self.total_words[1] + self.alpha * len(self.word_counts)))
                else:
                    spam_score += np.log(self.alpha / (self.total_words[0] + self.alpha * len(self.word_counts)))
                    not_spam_score += np.log(self.alpha / (self.total_words[1] + self.alpha * len(self.word_counts)))
            if spam_score > not_spam_score:
                predictions.append('spam')
            else:
                predictions.append('not spam')
        return predictions

# Sample dataset
data = {
    'Message': [
        "Hey, how's it going?",
        "Reminder: Your appointment is tomorrow at 2 PM.",
        "Congratulations! You've won a free vacation! Click here to claim your prize.",
        # Add more messages here...
    ],
    'Label': [
        'not spam',
        'not spam',
        'spam',
        # Add corresponding labels here...
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split data into X and y
X = df['Message']
y = df['Label']

# Create and train classifier
classifier = NaiveBayesClassifier(alpha=1)  # Set alpha for Laplace smoothing
classifier.train(X, y)

# Function to classify input message
def classify_message(message):
    preprocessed_message = message.lower()  # Convert to lowercase
    prediction = classifier.predict([preprocessed_message])[0]
    return prediction

# Input message to classify
input_message = "Congratulations! You've won a trip to paradise. Click here to claim your prize."

# Classify input message
result = classify_message(input_message)
print("Input message classified as:", result)

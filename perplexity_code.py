import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK punkt resource
nltk.download('punkt')

# Load dataset
data_path = '/content/Tax_Kams_Europe.csv'  # Update with your dataset path
df = pd.read_csv(data_path)

# Fill missing values with empty strings
df.fillna('', inplace=True)

# Combine text columns
df['concatenated_text'] = df[['TITLE', 'DESCRIPTION', 'RESPONSE', 'CONCLUSION']].apply(lambda x: ' '.join(x), axis=1)

# Define stop words
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Remove stop words and tokenize
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# Preprocess text
df['processed_text'] = df['concatenated_text'].apply(preprocess_text)

# Vectorize text
vectorizer = CountVectorizer(max_features=1000, lowercase=True, stop_words='english', max_df=0.5, min_df=10)
X = vectorizer.fit_transform(df['processed_text'])

# Define range of number of topics to evaluate
num_topics_range = range(2, 11)

# Initialize lists to store perplexity values
perplexity_values = []

# Calculate perplexity for each number of topics
for num_topics in num_topics_range:
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    perplexity = lda.perplexity(X)
    perplexity_values.append(perplexity)

# Plot perplexity graph
plt.plot(num_topics_range, perplexity_values, marker='o')
plt.title('Perplexity vs. Number of Topics')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.xticks(num_topics_range)
plt.grid(True)
plt.show()

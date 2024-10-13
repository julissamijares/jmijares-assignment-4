from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the 20 Newsgroups dataset
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(max_features=5000)
term_doc_matrix = vectorizer.fit_transform(documents)

# Apply SVD for dimensionality reduction
svd = TruncatedSVD(n_components=100)
lsa_matrix = svd.fit_transform(term_doc_matrix)

# Function to process query and calculate cosine similarity
def process_query(query):
    query_vec = vectorizer.transform([query])
    query_lsa = svd.transform(query_vec)
    return query_lsa

def get_top_documents(query, lsa_matrix=lsa_matrix, top_n=5):
    # Convert query to the same LSA space
    query_vector = vectorizer.transform([query])  # TF-IDF transform
    query_lsa = svd.transform(query_vector)       # Project into LSA space
    
    # Calculate cosine similarity between query and all documents
    similarities = cosine_similarity(query_lsa, lsa_matrix)[0]  # 1D array
    
    # Get the indices of the top N documents
    top_indices = np.argsort(similarities)[-top_n:][::-1]  # Sort and reverse to get top scores
    
    # Return the indices of the top documents and their similarity scores
    top_similarities = similarities[top_indices]
    
    print("Top document indices:", top_indices)
    print("Top similarities:", top_similarities)
    
    return top_indices, top_similarities

import requests        # For making HTTP requests to the local embedding API
import os              # For file system operations
import json            # For loading JSON content
import numpy as np     # For numerical operations (e.g., array manipulation)
import pandas as pd    # For working with tabular data
from sklearn.metrics.pairwise import cosine_similarity  # For similarity comparison
import joblib          # For saving/loading data (like DataFrames) efficiently

# Function to create embeddings from a list of text inputs using the local embedding API
def create_embedding(text_list):
    # Send a POST request to the local embedding server (e.g., Ollama with bge-m3 model)
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    # Extract the embedding vectors from the response
    embedding = r.json()["embeddings"]
    return embedding

# Get a list of all JSON files in the 'jsons' directory
jsons = os.listdir("jsons")
print(jsons)  # Print filenames for verification

my_dicts = []     # List to hold all processed chunk dictionaries
chunk_id = 0      # Unique identifier for each chunk (across all files)

# Loop through each JSON file
for json_file in jsons:
    # Open and load the JSON file
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)

    print(f"Creating Embeddings for {json_file}")

    # Generate embeddings for all text chunks in the file
    embeddings = create_embedding([c['text'] for c in content['chunks']])

    # Assign metadata and embedding to each chunk
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id                # Add unique chunk ID
        chunk['embedding'] = embeddings[i]          # Add embedding vector
        chunk_id += 1
        my_dicts.append(chunk)                      # Append to master list

# Convert the list of dictionaries into a Pandas DataFrame
df = pd.DataFrame.from_records(my_dicts)

# Save the DataFrame to a file for later use (e.g., searching or retrieval)
joblib.dump(df, 'embeddings.joblib')

# The following block is commented out — it's for running similarity search using a question embedding

# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0]

# Compute cosine similarity between the query embedding and all chunk embeddings
# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

# Get the indices of the top N most similar chunks
# top_results = 3
# max_indx = similarities.argsort()[::-1][0:top_results]

# Retrieve the most relevant chunks and print their info
# new_df = df.loc[max_indx]
# print(new_df[["title", "number", "text"]])

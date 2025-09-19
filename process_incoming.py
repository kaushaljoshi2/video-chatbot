import requests        # For sending HTTP requests to the local embedding and LLM APIs
import os              # For interacting with the file system (not used here but imported)
import json            # For parsing and generating JSON
import numpy as np     # For numerical operations and array manipulation
import pandas as pd    # For handling structured data
from sklearn.metrics.pairwise import cosine_similarity  # To compute similarity between embeddings
import joblib          # For loading the saved DataFrame with embeddings

# Function to generate embeddings from a list of text inputs
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",      # Use the bge-m3 embedding model
        "input": text_list      # List of strings to embed
    })
    embedding = r.json()["embeddings"]
    return embedding

# Function to run inference using a local LLM with a given prompt
def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",  # Alternative model 
        "model": "llama3.2",      # Currently used LLM model (for faster processing use models like gpt 4o)
        "prompt": prompt,         # Input prompt to generate a response
        "stream": False           # Disable streaming response
    })

    response = r.json()
    print(response)  # Print the raw response from the LLM
    return response

# Load precomputed embeddings and subtitle metadata from file
df = joblib.load('embeddings.joblib')

# Prompt the user for a question
incoming_query = input("Ask a Question: ")

# Generate the embedding for the user question
question_embedding = create_embedding([incoming_query])[0]

# Compute cosine similarity between the question embedding and all subtitle chunk embeddings
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

# Get indices of top N most relevant subtitle chunks
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]

# Retrieve the top matching subtitle chunks
new_df = df.loc[max_indx]

# Construct the prompt for the language model using top-matching subtitle chunks
prompt = f''' I am teaching web development in my Sigma Web Development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:  

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

----------------------------------------
"{incoming_query}"

User asked this question related to the video chunks, you have to answer in a human way (don't mention the above format, it's just for you) where and how much content is taught where (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course.
'''

# Save the generated prompt to a file (useful for debugging or review)
with open("prompt.txt", "w") as f:
    f.write(prompt)

# Send the prompt to the LLM and extract the generated answer
response = inference(prompt)["response"]
print(response)

# Save the LLM response to a file
with open("response.txt", "w") as f:
    f.write(response)

# (Optional) Print detailed info about top chunks — currently commented out
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])

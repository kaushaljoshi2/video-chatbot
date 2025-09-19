# RAG based AI video chatbot

### Video Chatbot: Ask Questions About Your Course Videos Using AI

## Description

This project turns a folder of **Educational Videos** into an intelligent **Video Q&A Chatbot** using:

- Whisper (for transcription + translation)
- ffmpeg (for audio extraction)
- Ollama (for embeddings + LLM)
- Pandas, Scikit-learn (for processing + retrieval)

Users can ask questions like:
> _"Where do you teach (...) ?"_

And get intelligent answers with **video number + timestamps** like:  
> _"(...) is introduced in Video #01 from 0:12 to 4:30. Please refer to that section for more details."_

## Features

- Converts `.mp4` videos into `.mp3`
- Transcribes & translates Hindi speech using Whisper
- Embeds transcripts with `bge-m3` via Ollama
- Finds relevant video chunks using cosine similarity
- Generates answers using local LLMs like `llama3.2` or `gpt-4o`

## Prerequisites

### 1. Python Libraries

- Install using `pip`: `pip install openai-whisper numpy pandas scikit-learn joblib requests`

### 2. Whisper by OpenAI

- Clone into your project directory: `git clone https://github.com/openai/whisper.git`

### 3. ffmpeg

- Required for video-to-audio conversion. Download and install:
Windows: `https://ffmpeg.org/download.html`
macOS (via Homebrew): `brew install ffmpeg`
Linux: `sudo apt install ffmpeg`

### 4. Ollama

- Used for both embeddings and LLM inference: `https://ollama.com/download`
- Consider this link for selecting your LLM model: `https://ollama.com/library`
(Consider model based on your system specs/processing speed/internet connection)

## Usage

### Step 1: Place Your Videos

- Put all your `.mp4` (or other like `webm`) video files in the `videos/` folder.

### Step 2: Extract Audio from Videos

- Run the script to convert videos to `.mp3`:
`python video_to_mp3.py`
- This stores output in `audios/`.

### Step 3: Transcribe & Translate Audio

- Use Whisper to generate English transcripts from Hindi audio:
`python mp3_to_json.py`
- This will generate `output.json` or multiple JSON files (if extended) in the `jsons/` folder.
- Note: Code assumes input audio is in Hindi. You can modify the `language="hi"` parameter if using a different language.

### Step 4: Generate Embeddings

- Create vector embeddings for each transcript chunk:
`python preprocess_json.py`
- This creates embeddings.joblib.
- Requires Ollama running locally with bge-m3 model.

### Step 5: Ask Questions via Chatbot

- Run the interactive search and response generator:
`python process_incoming.py`

Enter your question when prompted, and the script will:

- Find the most relevant video segments
- Show where and when content is covered
- Save the prompt and response

## System Requirements

For smoother experience, this project requires a mid to high-spec system:

- On low-spec systems, performance may degrade or models may fail to run entirely.

## Model Suggestions

- Embedding: bge-m3 via Ollama
- LLM (Local): llama3.2, deepseek-r1 (very slow not recommended), etc.
- LLM (Cloud): Use gpt-4o via OpenAI API for better responses (requires code adjustments & it also may charge based on use)

## Acknowledgments

- This project is created as part of the [Data Science Course](https://www.codewithharry.com/courses/the-ultimate-job-ready-data-science-course) Created by [CodeWithHarry](https://www.youtube.com/@CodeWithHarry)
- Special thanks to:

    [CodeWithHarry](https://www.youtube.com/@CodeWithHarry) - for the course inspiration

    [OpenAI Whisper](https://github.com/openai/whisper) - for transcription

    [Ollama](https://ollama.com/) - for local LLM + embedding models

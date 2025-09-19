import whisper  # Import the Whisper speech recognition library
import json     # For saving the transcription results in JSON format
import os       # For interacting with the file system

# Load the Whisper model (large-v2 for best accuracy and multilingual support)
model = whisper.load_model("large-v2")

# Get a list of audio files from the 'audios' directory
audios = os.listdir("audios")

# Process each audio file
for audio in audios:
    # Ensure the filename has an underscore separating number and title
    if "_" in audio:
        # Extract the tutorial number from the filename (before the underscore)
        number = audio.split("_")[0]
        
        # Extract the title from the filename (after the underscore), removing the .mp3 extension
        title = audio.split("_")[1][:-4]

        print(number, title)  # Print the number and title for tracking progress

        # Transcribe and translate the audio using Whisper
        result = model.transcribe(
            # To process each file in the loop, replace the line below with:
            # audio = f"audios/{audio}",
            # Taking sample.mp3 for faster processing. You can take files as per your use (or you can bring all mp3 files in a folder and then convert them to json as described in above commented code)
            audio = f"audios/sample.mp3", 
            language="hi",          # Input language is Hindi
            task="translate",       # Translates to English during transcription
            word_timestamps=False   # We only need sentence-level timestamps
        )

        # Prepare a list of chunks with metadata for each segment
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],   # Start time of the segment
                "end": segment["end"],       # End time of the segment
                "text": segment["text"]      # Translated text
            })
        
        # Combine chunks and full transcription text
        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }

        # Save the result to an output file (currently overwrites for each audio)
        with open(f"output.json", "w") as f:
            json.dump(chunks_with_metadata, f)

import os  # For interacting with the file system
import subprocess  # For running external commands like ffmpeg

# Get a list of all files in the "videos" directory
files = os.listdir("videos")

# Loop through each video file
for file in files:
    # Extract the tutorial number from the filename
    # Assumes format like "Some Title #01 [info] ｜ More Info.mp4"
    tutorial_number = file.split(" [")[0].split(" #")[1]
    
    # Extract the main part of the filename before the "｜" symbol
    file_name = file.split(" ｜ ")[0]
    
    # Print extracted info for tracking
    print(tutorial_number, file_name)
    
    # Convert the video to MP3 using ffmpeg
    # Output filename format: "<tutorial_number>_<file_name>.mp3"
    subprocess.run([
        "ffmpeg", 
        "-i", f"videos/{file}", 
        f"audios/{tutorial_number}_{file_name}.mp3"
    ])

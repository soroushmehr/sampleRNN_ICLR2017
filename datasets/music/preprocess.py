import os, sys
import subprocess

RAW_DATA_DIR=str(sys.argv[1])
OUTPUT_DIR=os.path.join(RAW_DATA_DIR, "parts")
os.makedirs(OUTPUT_DIR)
print RAW_DATA_DIR
print OUTPUT_DIR

# Step 1: write all filenames to a list
with open(os.path.join(OUTPUT_DIR, 'preprocess_file_list.txt'), 'w') as f:
    for dirpath, dirnames, filenames in os.walk(RAW_DATA_DIR):
        for filename in filenames:
            f.write("file '" + dirpath + '/'+ filename + "'\n")

# Step 2: concatenate everything into one massive wav file
os.system("ffmpeg -f concat -safe 0 -i {}/preprocess_file_list.txt {}/preprocess_all_audio.wav".format(OUTPUT_DIR, OUTPUT_DIR))

# # get the length of the resulting file
length = float(subprocess.check_output('ffprobe -i {}/preprocess_all_audio.wav -show_entries format=duration -v quiet -of csv="p=0"'.format(OUTPUT_DIR), shell=True))

# # Step 3: split the big file into 8-second chunks
for i in xrange(int(length)//8 - 1):
    os.system('ffmpeg -ss {} -t 8 -i {}/preprocess_all_audio.wav -ac 1 -ab 16k -ar 16000 {}/p{}.flac'.format(i, OUTPUT_DIR, OUTPUT_DIR, i))

# # Step 4: clean up temp files
os.system('rm {}/preprocess_all_audio.wav'.format(OUTPUT_DIR))
os.system('rm {}/preprocess_file_list.txt'.format(OUTPUT_DIR))

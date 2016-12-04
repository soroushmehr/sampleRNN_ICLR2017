# Requires 2GB of free disk space at most.
# Change the following address accordingly. Ends with /
DL_PATH=/Tmp/mehris/music/download/ 
mkdir -p $DL_PATH
echo "Downloading files to "$DL_PATH""
# See: https://blog.archive.org/2012/04/26/downloading-in-bulk-using-wget/
wget -r -H -nc -nH --cut-dir=1 -A .ogg -R *_vbr.mp3 -e robots=off -P "$DL_PATH" -l1 -i ./itemlist.txt -B 'http://archive.org/download/'
echo "Organizing files and folders"
mv "$DL_PATH"*/*.ogg "$DL_PATH"
rmdir "$DL_PATH"*/
echo "Converting from OGG to 16Khz, 16bit mono-channel WAV"
# Next line with & executes in a forked shell in the background. That's parallel and not recommended.
# Remove if causing problem
#for file in "$DL_PATH"*_64kb.mp3; do ffmpeg -i "$file" -ar 16000 -ac 1 "$DL_PATH""`basename "$file" _64kb.mp3`.wav" & done 
for file in "$DL_PATH"*.ogg; do
	ffmpeg -i "$file" -ar 16000 -ac 1 "$DL_PATH""`basename "$file" .ogg`.wav"
done 
echo "Cleaning up"
rm "$DL_PATH"*.ogg

echo "Preprocessing"
python preprocess.py "$DL_PATH"
echo "Done!"

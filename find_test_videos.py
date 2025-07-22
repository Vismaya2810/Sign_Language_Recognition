import json
import os

WLASL_JSON = 'resource/WLASL_v0.3.json'
VIDEO_DIR = 'resource/videos'
# WORDS = ['before', 'bowling', 'go', 'trade', 'candy']
WORDS = ['now', 'drink','water']

# Get set of available video filenames (without .mp4)
available_videos = {f[:-4] for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')}

def main():
    with open(WLASL_JSON, 'r') as f:
        data = json.load(f)
    print('Available test videos for each word:')
    for entry in data:
        gloss = entry['gloss']
        if gloss in WORDS:
            vids = [inst['video_id'] for inst in entry['instances'] if inst['video_id'] in available_videos]
            if vids:
                print(f"{gloss}: {[v + '.mp4' for v in vids]}")
            else:
                print(f"{gloss}: No available videos!")

if __name__ == '__main__':
    main() 
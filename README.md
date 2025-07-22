# Sign Language to Text (LSTM Demo)

This project demonstrates a pipeline for converting sign language videos to text using MediaPipe, an LSTM neural network, and Streamlit.

## Features
- Upload a video of a sign (mp4)
- Extracts hand landmarks using MediaPipe
- Uses an LSTM model to classify the sign (sequence of landmarks)
- Displays the detected word as text and shows a probability bar chart with word labels
- **Upload multiple videos to form a sentence** (with advanced grammar correction)

---

## Step-by-Step: Training and Using the Model

### 1. Prepare Your Data
- Place your sign language video files in the `resource/videos/` directory.
- Ensure your dataset label file (`resource/WLASL_v0.3.json`) is present.
- Edit `resource/top_words.txt` to contain the list of words you want to train on (one per line, e.g., the top 20 words).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
PYTHONPATH=. python src/train.py
```
- This will:
  - Parse the JSON and find videos for the words in `top_words.txt`.
  - Extract hand landmark sequences from each video.
  - Augment the data for better generalization.
  - Train an LSTM model on the sequences.
  - Save the trained model to `src/sign_word_lstm.h5` and the label encoder to `src/label_encoder.joblib`.

### 4. Run the Streamlit App
```bash
streamlit run src/app.py
```
- This will launch a web app at [http://localhost:8501](http://localhost:8501) (or a similar port).

### 5. Test the Model & Form Sentences
1. **Upload one or more videos** (from `resource/videos/`) for your trained words, in the order you want them to appear in the sentence.
2. The app will:
   - Predict a word for each video.
   - Show the predicted word and its confidence.
   - Display a bar chart with probabilities for each word (with actual word labels).
   - **Form a sentence from the sequence of words and apply advanced grammar correction.**

#### Example
- Upload videos for: `before`, `bowling`, `go`, `trade`, `candy` (in order)
- The app will output: `Before bowling, go trade candy.` (with grammar correction if possible)
- **Example video files for this sentence:**
  - `05731.mp4` → before
  - `07397.mp4` → bowling
  - `24962.mp4` → go
  - `59207.mp4` → trade
  - `08918.mp4` → candy
- Upload these files in order to form the sentence above.

#### How to Find Video Files for Each Word
- Use the script:
  ```bash
  PYTHONPATH=. python3 src/find_test_videos.py
  ```
- This will print available video filenames for each word in your sentence (e.g., `before`, `bowling`, `go`, `trade`, `candy`).
- Example output:
  ```
  before: ['05728.mp4', ...]
  bowling: ['07389.mp4', ...]
  go: ['69345.mp4', ...]
  trade: ['59206.mp4', ...]
  candy: ['08929.mp4', ...]
  ```
- Upload one file for each word in the order you want them to appear in the sentence.

---

## Notes
- To change the set of words, edit `resource/top_words.txt` and retrain.
- To add more data, place more videos in `resource/videos/` and retrain.
- The model is optimized for the words in `top_words.txt`—using videos for other words will not work as expected.
- For best results, ensure you have several videos per word.

---

## Next Steps
- Add more words or videos for better accuracy.
- Add webcam support or sentence formation if desired.
- For any issues or feature requests, just ask! 
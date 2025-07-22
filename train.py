import json
import os
import numpy as np
from src.data_prep import extract_hand_landmarks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Conv1D, BatchNormalization, GlobalAveragePooling1D, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

WLASL_JSON = 'resource/WLASL_v0.3.json'
VIDEO_DIR = 'resource/videos'
TOP_WORDS_FILE = 'resource/top_words.txt'
SAMPLES_PER_WORD = 10  # Use up to 10 samples per word if available
SEQ_LEN = 15  # Number of frames per sequence

# Read top words from file
def get_selected_words():
    with open(TOP_WORDS_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# 1. Parse JSON and collect video paths for selected words
def get_video_paths(selected_words):
    with open(WLASL_JSON, 'r') as f:
        data = json.load(f)
    samples = []
    for entry in data:
        gloss = entry['gloss']
        if gloss in selected_words:
            count = 0
            for inst in entry['instances']:
                if count >= SAMPLES_PER_WORD:
                    break
                video_id = inst['video_id']
                video_path = os.path.join(VIDEO_DIR, f'{video_id}.mp4')
                if os.path.exists(video_path):
                    samples.append((gloss, video_path))
                    count += 1
    return samples

def mirror_landmarks(seq):
    # Mirror x-coordinates (assuming normalized [0,1])
    mirrored = seq.copy()
    mirrored[..., 0] = 1.0 - mirrored[..., 0]
    return mirrored

def augment_sequence(seq, num_aug=5, noise_std=0.01, scale_range=0.05, rot_range=10, trans_range=0.02):
    augmented = []
    for _ in range(num_aug):
        # Gaussian noise
        noise = np.random.normal(0, noise_std, size=seq.shape)
        # Random scale
        scale = 1 + np.random.uniform(-scale_range, scale_range)
        # Random rotation (2D, for x/y only)
        theta = np.radians(np.random.uniform(-rot_range, rot_range))
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        lm = seq.copy()
        lm_xy = lm[..., :2] @ rot_matrix.T
        # Random translation (2D)
        trans = np.random.uniform(-trans_range, trans_range, size=(1, 2))
        lm_xy += trans
        lm_aug = np.concatenate([lm_xy, lm[..., 2:]], axis=-1) * scale + noise
        augmented.append(lm_aug)
        # Add mirrored version
        augmented.append(mirror_landmarks(lm_aug))
    return augmented

def extract_sequence(video_path, seq_len=SEQ_LEN):
    landmarks_list = extract_hand_landmarks(video_path, max_frames=seq_len)
    if not landmarks_list:
        return None
    # Pad or truncate to seq_len
    seq = [l for _, l in landmarks_list]
    if len(seq) < seq_len:
        # Pad with zeros
        pad = [np.zeros_like(seq[0])] * (seq_len - len(seq))
        seq.extend(pad)
    else:
        seq = seq[:seq_len]
    return np.stack(seq)

def build_dataset(samples):
    X, y = [], []
    for label, video_path in samples:
        try:
            seq = extract_sequence(video_path)
            if seq is not None:
                X.append(seq)
                y.append(label)
                # Augmentation
                for aug_seq in augment_sequence(seq, num_aug=10):
                    X.append(aug_seq)
                    y.append(label)
        except Exception as e:
            print(f'Error processing {video_path}: {e}')
    X = np.array(X)
    y = np.array(y)
    # Remove samples with NaN or Inf
    mask = np.all(np.isfinite(X), axis=(1,2))
    if not np.all(mask):
        print(f'Warning: Removed {np.sum(~mask)} samples with NaN or Inf values')
        X = X[mask]
        y = y[mask]
    return X, y

def train_cnn_classifier(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(SEQ_LEN, 63)),
        BatchNormalization(),
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(y_cat.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {acc*100:.2f}%')
    # Save model and label encoder
    model.save('src/sign_word_cnn.h5')
    joblib.dump(le, 'src/label_encoder.joblib')
    print('CNN model and label encoder saved.')
    return model, le

def main():
    selected_words = get_selected_words()
    print(f'Training on words: {selected_words}')
    samples = get_video_paths(selected_words)
    print(f'Found {len(samples)} video samples.')
    X, y = build_dataset(samples)
    print(f'Dataset shape: {X.shape}, Labels: {set(y)}')
    train_cnn_classifier(X, y)

if __name__ == '__main__':
    main() 
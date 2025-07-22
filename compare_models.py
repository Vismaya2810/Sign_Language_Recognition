print('Script started')

try:
    import numpy as np
    from tensorflow.keras.models import load_model
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    import joblib
    import time
    from train import get_selected_words, get_video_paths, build_dataset
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load label encoder
    le = joblib.load('src/label_encoder.joblib')

    # Prepare data
    selected_words = get_selected_words()
    samples = get_video_paths(selected_words)
    X, y = build_dataset(samples)
    y_enc = le.transform(y)

    # Split into train/test (same as in train.py)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    # Load models
    cnn_model = load_model('src/sign_word_cnn.h5')
    lstm_model = load_model('src/sign_word_lstm.h5')

    # Predict with CNN
    start = time.time()
    cnn_preds = cnn_model.predict(X_test)
    cnn_pred_labels = np.argmax(cnn_preds, axis=1)
    cnn_time = time.time() - start

    # Predict with LSTM
    start = time.time()
    lstm_preds = lstm_model.predict(X_test)
    lstm_pred_labels = np.argmax(lstm_preds, axis=1)
    lstm_time = time.time() - start

    # Metrics
    cnn_acc = accuracy_score(y_test, cnn_pred_labels)
    lstm_acc = accuracy_score(y_test, lstm_pred_labels)
    cnn_f1 = f1_score(y_test, cnn_pred_labels, average='macro')
    lstm_f1 = f1_score(y_test, lstm_pred_labels, average='macro')

    print("CNN Accuracy:", cnn_acc)
    print("LSTM Accuracy:", lstm_acc)

    print("\nCNN Classification Report:")
    print(classification_report(y_test, cnn_pred_labels, target_names=le.classes_))

    print("\nLSTM Classification Report:")
    print(classification_report(y_test, lstm_pred_labels, target_names=le.classes_))

    print("\nCNN Confusion Matrix:")
    print(confusion_matrix(y_test, cnn_pred_labels))

    print("\nLSTM Confusion Matrix:")
    print(confusion_matrix(y_test, lstm_pred_labels))

    print(f"\nCNN Inference Time: {cnn_time:.2f} seconds")
    print(f"LSTM Inference Time: {lstm_time:.2f} seconds")

    # Visualization 1: Bar chart for accuracy and macro F1
    plt.figure(figsize=(6,4))
    metrics = ['Accuracy', 'Macro F1']
    cnn_scores = [cnn_acc, cnn_f1]
    lstm_scores = [lstm_acc, lstm_f1]
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, cnn_scores, width, label='CNN')
    plt.bar(x + width/2, lstm_scores, width, label='LSTM')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('CNN vs LSTM: Accuracy and Macro F1')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison_bar.png')
    plt.show()

    # Visualization 2: Side-by-side confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(confusion_matrix(y_test, cnn_pred_labels), ax=axes[0], cmap='Blues', cbar=False)
    axes[0].set_title('CNN Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    sns.heatmap(confusion_matrix(y_test, lstm_pred_labels), ax=axes[1], cmap='Greens', cbar=False)
    axes[1].set_title('LSTM Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    plt.tight_layout()
    plt.savefig('model_comparison_confusion.png')
    plt.show()

    # Print which model is better
    if cnn_acc > lstm_acc and cnn_f1 > lstm_f1:
        print("\nCNN is better than LSTM in both accuracy and macro F1.")
    elif lstm_acc > cnn_acc and lstm_f1 > cnn_f1:
        print("\nLSTM is better than CNN in both accuracy and macro F1.")
    else:
        print("\nThe models have mixed results. See the bar chart and reports above.")
except Exception as e:
    print("Error:", e)
    import traceback; traceback.print_exc() 
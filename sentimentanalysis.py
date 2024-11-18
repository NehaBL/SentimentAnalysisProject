# Basic libraries
import numpy as np
import pandas as pd

# NLP libraries
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data (stopwords, etc.)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


data = pd.read_csv('twitter_training.csv')
data = pd.read_csv('/content/twitter_training.csv')
# Example preprocessing function
def preprocess_text(text):
    # Lowercase text
    text = text.lower()
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing to text data
data['processed_text'] = data['review_text'].apply(preprocess_text)


data.columns


# Reload the dataset with specified column names
data = pd.read_csv('twitter_training.csv', names=['id', 'text', 'label'])
data.head()
# Apply preprocessing to the text column
data['processed_text'] = data['text'].apply(preprocess_text)


import nltk
nltk.download('punkt')


!pip install tensorflow


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Example parameters - adjust as needed
vocab_size = 5000  # Number of unique words in the vocabulary
embedding_dim = 64
max_length = 100  # Maximum length of input sequences

# Build the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(units=64, return_sequences=True),
    LSTM(units=32),
    Dense(1, activation='sigmoid')  # Binary classification (positive/negative)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the tokenizer and fit it on your text data
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(data['text'])  # Replace 'text' with your text column name

# Convert the texts to sequences
X_sequences = tokenizer.texts_to_sequences(data['text'])

# Pad the sequences to ensure uniform length
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')

# Prepare the labels
y = data['label'].apply(lambda x: 1 if x == 'Positive' else 0).values  # Convert labels to binary (1/0)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)


history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


def predict_sentiment(text):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Predict sentiment
    prediction = model.predict(padded_sequence)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

# Test the function
print(predict_sentiment("I love this product!"))


history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


data['label'].value_counts()


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")


from tensorflow.keras.layers import Dropout

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(units=64, return_sequences=True),
    Dropout(0.5),
    LSTM(units=32),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])


# Check if there are any duplicates between training and test sets
train_texts = set(data.iloc[X_train.index]['text'])
test_texts = set(data.iloc[X_test.index]['text'])

print(f"Number of overlaps between train and test sets: {len(train_texts & test_texts)}")


# Check if there are any duplicates between training and test sets
train_texts = set(data.loc[y_train.index, 'text'])
test_texts = set(data.loc[y_test.index, 'text'])

print(f"Number of overlaps between train and test sets: {len(train_texts & test_texts)}")


# Assuming X_train and X_test contain the text data directly (as lists or arrays)
train_texts = set(X_train)  # Convert training texts to a set
test_texts = set(X_test)    # Convert test texts to a set

# Find the intersection (overlapping texts) between the train and test sets
overlap_count = len(train_texts & test_texts)
print(f"Number of overlaps between train and test sets: {overlap_count}")


# Convert each numpy array in X_train and X_test to strings
train_texts = set([str(text) for text in X_train])  # Convert training texts to a set
test_texts = set([str(text) for text in X_test])    # Convert test texts to a set

# Find the intersection (overlapping texts) between the train and test sets
overlap_count = len(train_texts & test_texts)
print(f"Number of overlaps between train and test sets: {overlap_count}")


# Filter out overlapping texts from the test set
X_test_filtered = [text for text in X_test if str(text) not in train_texts]
y_test_filtered = [y_test[i] for i, text in enumerate(X_test) if str(text) not in train_texts]

# Convert back to numpy arrays if needed
X_test_filtered = np.array(X_test_filtered)
y_test_filtered = np.array(y_test_filtered)

# Check the new test set size
print(f"Original test set size: {len(X_test)}")
print(f"Filtered test set size: {len(X_test_filtered)}")


# Re-evaluate the model on the filtered test set
test_loss, test_accuracy = model.evaluate(X_test_filtered, y_test_filtered)
print(f"Test Accuracy after removing overlaps: {test_accuracy:.2f}")


from tensorflow.keras.preprocessing.sequence import pad_sequences

# Pad the filtered test set to match the input shape required by the model
X_test_filtered_padded = pad_sequences(X_test_filtered, maxlen=max_length, padding='post', truncating='post')


test_loss, test_accuracy = model.evaluate(X_test_filtered_padded, y_test_filtered)
print(f"Test Accuracy after removing overlaps: {test_accuracy:.2f}")


print(f"Filtered test set size (X): {len(X_test_filtered_padded)}")
print(f"Filtered test set size (y): {len(y_test_filtered)}")


if len(X_test_filtered_padded) > 0 and len(y_test_filtered) > 0:
    test_loss, test_accuracy = model.evaluate(X_test_filtered_padded, y_test_filtered)
    print(f"Test Accuracy after removing overlaps: {test_accuracy:.2f}")
else:
    print("Filtered test set is empty; unable to evaluate model.")




# Evaluate on the original test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy on original test set: {test_accuracy:.2f}")


from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import clone_model

# Initialize stratified K-fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for train_index, test_index in kf.split(X, y):
    # Split data
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Clone model to get a fresh copy for each fold
    model_fold = clone_model(model)
    model_fold.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model on the fold
    model_fold.fit(X_train_fold, y_train_fold, epochs=5, batch_size=32, verbose=0)

    # Evaluate on test fold
    _, accuracy = model_fold.evaluate(X_test_fold, y_test_fold, verbose=0)
    fold_accuracies.append(accuracy)

# Calculate average accuracy across folds
average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
print(f"Cross-Validation Accuracy: {average_accuracy:.2f}")


# Assuming X_train and y_train represent your full dataset
X = np.concatenate([X_train, X_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)


from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import clone_model
import numpy as np

# Assuming X and y are your complete feature and label sets
# Replace X and y here with your actual data variables if they are different
X = np.concatenate([X_train, X_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# Initialize stratified K-fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for train_index, test_index in kf.split(X, y):
    # Split data
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Clone model to get a fresh copy for each fold
    model_fold = clone_model(model)
    model_fold.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model on the fold
    model_fold.fit(X_train_fold, y_train_fold, epochs=5, batch_size=32, verbose=0)

    # Evaluate on test fold
    _, accuracy = model_fold.evaluate(X_test_fold, y_test_fold, verbose=0)
    fold_accuracies.append(accuracy)

# Calculate average accuracy across folds
average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
print(f"Cross-Validation Accuracy: {average_accuracy:.2f}")


from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Define a simpler model with dropout
simpler_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(units=64),
    Dropout(0.5),  # Dropout layer to prevent overfitting
    Dense(1, activation='sigmoid')
])

# Compile the model
simpler_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the simpler model
history = simpler_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


def predict_sentiment(text):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Predict sentiment
    prediction = simpler_model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return sentiment, f"Confidence: {confidence:.2f}"

# Test with new examples
print(predict_sentiment("This is an amazing product!"))
print(predict_sentiment("The service was disappointing and slow."))


# Check class distribution
import numpy as np
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class Distribution in Training Set:", class_distribution)


# Example: Increase number of LSTM units and adjust dropout
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

tuned_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(units=128),  # Increase LSTM units
    Dropout(0.3),     # Adjust dropout
    Dense(1, activation='sigmoid')
])

# Compile and train the model
tuned_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = tuned_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))


from sklearn.metrics import classification_report

# Predict on test set
y_pred = (tuned_model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))


print("Unique classes in y_test:", np.unique(y_test))
print("Unique classes in y_pred:", np.unique(y_pred))


from sklearn.metrics import classification_report

# Predict on test set
y_pred = (tuned_model.predict(X_test) > 0.5).astype("int32")

# Check if both classes are present, if not, specify labels manually
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"], labels=[0, 1]))


# Check the distribution of classes in the training set
import numpy as np
unique, counts = np.unique(y_train, return_counts=True)
print("Class distribution in training set:", dict(zip(unique, counts)))


# Calculate class weights (if necessary)
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)

# Use class weights in model training
history = tuned_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights_dict)


from tensorflow.keras.layers import Dropout

# Modify the model to include dropout
model_with_dropout = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(units=128, return_sequences=True),
    Dropout(0.4),  # Dropout layer to reduce overfitting
    LSTM(units=64),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model_with_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model_with_dropout.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights_dict)


from sklearn.metrics import classification_report

# Predict on test set
y_pred = (model_with_dropout.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"], labels=[0, 1]))


from sklearn.metrics import classification_report, roc_auc_score, f1_score

# Try a different threshold for predicting Positive class
threshold = 0.3  # Adjust this value as needed
y_pred = (model_with_dropout.predict(X_test) > threshold).astype("int32")

# Print classification report
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"], labels=[0, 1]))

# Calculate additional metrics
roc_auc = roc_auc_score(y_test, model_with_dropout.predict(X_test))
f1_positive = f1_score(y_test, y_pred, pos_label=1)

print(f"ROC-AUC Score: {roc_auc:.2f}")
print(f"F1 Score for Positive class: {f1_positive:.2f}")


from imblearn.over_sampling import SMOTE

# Only use this on the training data
smote = SMOTE(sampling_strategy='minority')
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Retrain the model with balanced data
history = model_with_dropout.fit(
    X_train_balanced, y_train_balanced,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict  # Optional if still needed
)


# Check for unique classes in y_train and y_test
if len(np.unique(y_train)) > 1:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy='minority')
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
else:
    X_train_balanced, y_train_balanced = X_train, y_train  # No resampling needed

# Train model with balanced or original data
history = model_with_dropout.fit(
    X_train_balanced, y_train_balanced,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate on test set
y_pred = (model_with_dropout.predict(X_test) > 0.5).astype("int32")

# Classification report (adjusted for single class)
if len(np.unique(y_test)) > 1:
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"], labels=[0, 1]))
else:
    print("Classification Report (Single Class):")
    print(classification_report(y_test, y_pred, target_names=["Negative"]))

# Skip ROC AUC if only one class is present
if len(np.unique(y_test)) > 1:
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", roc_auc)
else:
    print("ROC AUC Score not applicable for single class.")


# Check the class distribution in the training and test sets
print("Training set class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Test set class distribution:", dict(zip(*np.unique(y_test, return_counts=True))))


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np

# Stratified K-Folds Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for train_index, test_index in kf.split(X, y):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Train the model on this fold
    history = model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=32, validation_data=(X_test_fold, y_test_fold))

    # Evaluate on test fold
    y_pred_fold = (model.predict(X_test_fold) > 0.5).astype("int32")
    report = classification_report(y_test_fold, y_pred_fold, target_names=["Negative", "Positive"], zero_division=1)
    print(report)
    fold_accuracies.append(report)

# Calculate average across folds if needed


# Evaluate on test fold
y_pred_fold = (model.predict(X_test_fold) > 0.5).astype("int32")

# Determine the unique classes in y_test_fold and y_pred_fold
unique_classes = np.unique(y_test_fold.tolist() + y_pred_fold.tolist())

# Generate classification report based on the available classes
report = classification_report(y_test_fold, y_pred_fold, target_names=["Negative", "Positive"], labels=unique_classes, zero_division=1)
print(report)
fold_accuracies.append(report)


# Determine the unique classes in y_test_fold and y_pred_fold
unique_classes = np.unique(np.concatenate((y_test_fold, y_pred_fold)))

# Generate classification report based on the available classes
report = classification_report(y_test_fold, y_pred_fold, target_names=["Negative", "Positive"], labels=unique_classes, zero_division=1)
print(report)
fold_accuracies.append(report)


# Flatten y_test_fold and y_pred_fold to ensure they are one-dimensional
y_test_fold_flat = y_test_fold.ravel()
y_pred_fold_flat = y_pred_fold.ravel()

# Determine the unique classes in y_test_fold and y_pred_fold
unique_classes = np.unique(np.concatenate((y_test_fold_flat, y_pred_fold_flat)))

# Generate classification report based on the available classes
report = classification_report(y_test_fold_flat, y_pred_fold_flat, target_names=["Negative", "Positive"], labels=unique_classes, zero_division=1)
print(report)
fold_accuracies.append(report)


# Flatten y_test_fold and y_pred_fold to ensure they are one-dimensional
y_test_fold_flat = y_test_fold.ravel()
y_pred_fold_flat = y_pred_fold.ravel()

# Determine the unique classes in y_test_fold and y_pred_fold
unique_classes = np.unique(np.concatenate((y_test_fold_flat, y_pred_fold_flat)))

# Check if both classes are present, otherwise generate the report for available classes only
if len(unique_classes) == 2:
    report = classification_report(
        y_test_fold_flat, y_pred_fold_flat,
        target_names=["Negative", "Positive"],
        labels=unique_classes, zero_division=1
    )
else:
    # Generate report for single available class
    report = classification_report(
        y_test_fold_flat, y_pred_fold_flat,
        target_names=["Negative"] if unique_classes[0] == 0 else ["Positive"],
        labels=unique_classes, zero_division=1
    )

print(report)
fold_accuracies.append(report)


from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


print("Class distribution in y_train:", dict(zip(*np.unique(y_train, return_counts=True))))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


unique, counts = np.unique(y, return_counts=True)
print("Class distribution in the original dataset:", dict(zip(unique, counts)))


unique, counts = np.unique(y_train, return_counts=True)
print("Class distribution in y_train after splitting:", dict(zip(unique, counts)))


from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler(sampling_strategy='majority')
X_resampled, y_resampled = undersample.fit_resample(X, y)

# Now, perform the train-test split and apply SMOTE on y_train after checking class distribution.
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)


unique, counts = np.unique(y, return_counts=True)
print("Class distribution in y before resampling:", dict(zip(unique, counts)))


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Ensure balanced dataset
undersample = RandomUnderSampler(sampling_strategy='majority')
X_resampled, y_resampled = undersample.fit_resample(X, y)

# Split the resampled data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Apply SMOTE to the training set
smote = SMOTE(sampling_strategy='minority')
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Use class weights directly in model training
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=32,
    class_weight=class_weights_dict  # Include class weights
)


from sklearn.metrics import f1_score

# Predict on the test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculate F1-score
f1 = f1_score(y_test, y_pred, pos_label=1)
print("F1 Score:", f1)


from sklearn.metrics import precision_recall_curve

# Get predicted probabilities
y_probs = model.predict(X_test)

# Calculate precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# Find threshold that maximizes F1-score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print("Optimal threshold:", optimal_threshold)

# Apply the threshold to classify
y_pred_optimal = (y_probs >= optimal_threshold).astype("int32")

# Calculate F1-score with the new threshold
f1_optimal = f1_score(y_test, y_pred_optimal)
print("F1 Score with optimal threshold:", f1_optimal)


from sklearn.utils import resample

# Downsample majority class
X_balanced, y_balanced = resample(
    X[y == 0],
    y[y == 0],
    replace=False,
    n_samples=y[y == 1].shape[0],  # Downsample to the minority class size
    random_state=42
)
X_balanced = np.concatenate([X[y == 1], X_balanced], axis=0)
y_balanced = np.concatenate([y[y == 1], y_balanced], axis=0)


from sklearn.utils import resample

# Downsample majority class
# Check if minority class has samples before resampling
if y[y == 1].shape[0] > 0:  # Add this condition
    X_balanced, y_balanced = resample(
        X[y == 0],
        y[y == 0],
        replace=False,
        n_samples=y[y == 1].shape[0],  # Downsample to the minority class size
        random_state=42
    )
    X_balanced = np.concatenate([X[y == 1], X_balanced], axis=0)
    y_balanced = np.concatenate([y[y == 1], y_balanced], axis=0)
else:
    # Handle case where minority class has no samples
    print("Warning: Minority class has no samples. Skipping downsampling.")
    X_balanced, y_balanced = X, y # Or raise an exception, depending on your desired behavior

from collections import Counter

# Check distribution in the training set
print("Class distribution in y_train:", Counter(y_train))


from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

# Baseline model that predicts the most frequent class
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_baseline = baseline_model.predict(X_test)
print(classification_report(y_test, y_pred_baseline, target_names=["Negative", "Positive"], zero_division=1))


from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

# Baseline model that predicts the most frequent class
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_baseline = baseline_model.predict(X_test)

# Determine unique classes in y_test and set target names accordingly
unique_classes = np.unique(y_test)
target_names = ["Negative", "Positive"] if len(unique_classes) > 1 else ["Negative"]

# Print classification report
print(classification_report(y_test, y_pred_baseline, target_names=target_names, zero_division=1))


# Example prompt to generate positive sentences using GPT-3 or ChatGPT
from openai import ChatCompletion

prompt = "Generate a list of 10 positive product reviews."
response = ChatCompletion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=100
)
positive_examples = response.choices[0].text.strip().split("\n")


import pandas as pd

# Load initial dataset (assuming it has a column 'text' and 'label' where 0=Negative, 1=Positive)
initial_data = pd.read_csv('/path/to/your/initial_dataset.csv')

# Load new positive reviews dataset
positive_reviews = pd.read_csv('/path/to/your/positive_reviews.csv')

# Add a label column to the positive reviews (assuming 1 indicates Positive sentiment)
positive_reviews['label'] = 1

# Combine initial data with positive reviews
balanced_data = pd.concat([initial_data, positive_reviews], ignore_index=True)


# Load initial dataset
initial_data = pd.read_csv('/content/initial_dataset.csv')  # Replace with the actual file name

# Load new positive reviews dataset
positive_reviews = pd.read_csv('/content/positive_reviews.csv')  # Replace with the actual file name


# Load initial dataset
initial_data = pd.read_csv('/content/twitter_validation.csv')  # Adjust based on the file's purpose

# Load new positive reviews dataset
positive_reviews = pd.read_csv('/content/positive_reviews.csv')  # Use the path for the positive reviews file


# Load initial dataset
initial_data = pd.read_csv('/content/twitter_validation.csv')  # Adjust based on the file's purpose

# Load new positive reviews dataset
positive_reviews = pd.read_csv('/content/positive_reviews.csv')  # Use the correct path for the positive reviews file


# Load initial dataset
initial_data = pd.read_csv('/content/twitter_validation.csv')  # Adjust based on the file's purpose

# Load new positive reviews dataset
# Check if the file 'positive_examples.txt' exists. If not, create it.
import os
# Load initial dataset
initial_data = pd.read_csv('/content/twitter_validation.csv')  # Adjust based on the file's purpose

# Load new positive reviews dataset
# Check if the file 'positive_examples.txt' exists. If not, create it.
import os
if not os.path.exists('positive_examples.txt'):
    with open('positive_examples.txt', 'w') as f:
        f.write('\n')

# Import necessary libraries
import pandas as pd

# Load the initial dataset
initial_data = pd.read_csv('/content/twitter_validation.csv')  # Adjust the path if needed
initial_data['label'] = 0  # Assuming 0 indicates Negative sentiment

# Load the positive reviews dataset
positive_reviews = pd.read_csv('/content/positive_examples.txt', header=None, names=['text'])
positive_reviews['label'] = 1  # Assuming 1 indicates Positive sentiment

# Combine initial data with positive reviews
balanced_data = pd.concat([initial_data, positive_reviews], ignore_index=True)

# Check the distribution to ensure it's balanced
print("Class distribution in balanced data:", balanced_data['label'].value_counts())

# Proceed with preprocessing (e.g., tokenization, removing stop words)
# Example: Using sklearn's train_test_split to split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_t


# Import necessary libraries
import pandas as pd

# Load the initial dataset
initial_data = pd.read_csv('/content/twitter_validation.csv')  # Adjust the path if needed
initial_data['label'] = 0  # Assuming 0 indicates Negative sentiment

# Load the positive reviews dataset
positive_reviews = pd.read_csv('/content/positive_examples.txt', header=None, names=['text'])
positive_reviews['label'] = 1  # Assuming 1 indicates Positive sentiment

# Combine initial data with positive reviews
balanced_data = pd.concat([initial_data, positive_reviews], ignore_index=True)

# Check the distribution to ensure it's balanced
print("Class distribution in balanced data:", balanced_data['label'].value_counts())

# Proceed with preprocessing (e.g., tokenization, removing stop words)
# Example: Using sklearn's train_test_split to split the data
from sklearn.model_selection import train_test_split
# Include target variable 'y' when splitting the data
X_train, X_test, y_train, y_test = train_test_split(balanced_data['text'], balanced_data['label'], test_size=0.2, random_state=42) # Assuming 'text' column contains the review text

# Now you have X_train, X_test, y_train, y_test for further processing

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Define a function to preprocess text
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

# Apply the preprocessing to training and test data
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)


import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Define a function to preprocess text
def preprocess_text(text):
    # Check if the input is a string
    if isinstance(text, str):
        # Remove special characters and digits
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)
        # Convert text to lowercase
        text = text.lower()
        # Remove stop words
        text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
        return text
    else:
        # Handle non-string inputs (e.g., return an empty string or the original value)
        return ''  # Or return str(text) if you want to preserve non-string values as strings

# Apply the preprocessing to training and test data
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for faster processing

# Fit and transform on training data, transform on test data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Define a function to preprocess text
def preprocess_text(text):
    # Check if the input is a string
    if isinstance(text, str):
        # Remove special characters and digits
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)
        # Convert text to lowercase
        text = text.lower()
        # Remove stop words but keep at least one word
        words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
        if not words:  # If all words are stop words, keep the original text
            words = text.split()
        text = ' '.join(words)
        return text
    else:
        # Handle non-string inputs (e.g., return an empty string or the original value)
        return ''  # Or return str(text) if you want to preserve non-string values as strings

# Apply the preprocessing to training and test data
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

from sklearn.linear_model import LogisticRegression

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate the model on the test set
from sklearn.metrics import classification_report

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))


ipython-input-86-ab139e641044
ipython-input-85-ab139e641044
ipython-input-87-ab139e641044

# Instead of the content of cell ipython-input-88-517472b22bf4,
# simply run cells ipython-input-86, ipython-input-85, and ipython-input-87 in order.
# You can do this by clicking on each cell and pressing Shift+Enter.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Basic evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC curve (for binary classification)
y_probs = model.predict_proba(X_test_vec)[:, 1]  # Probability estimates for positive class
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# AUC score
auc_score = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {auc_score}")


import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Define a function to preprocess text
def preprocess_text(text):
    # Check if the input is a string
    if isinstance(text, str):
        # Remove special characters and digits
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)
        # Convert text to lowercase
        text = text.lower()
        # Remove stop words but keep at least one word
        words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
        if not words:  # If all words are stop words, keep the original text
            words = text.split()  # or return a placeholder like 'empty_text'
        text = ' '.join(words)
        return text
    else:
        # Handle non-string inputs (e.g., return an empty string or the original value)
        return ''  # Or return str(text) if you want to preserve non-string values as strings

# Apply the preprocessing to training and test data
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Check the shapes of X_train, X_test, y_train, y_test after preprocessing
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# If the shapes are inconsistent, investigate further and adjust your preprocessing accordingly

from sklearn.model_selection import GridSearchCV

# Example grid search with Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_vec, y_train)
best_model = grid_search.best_estimator_
print(f"Best Model: {best_model}")


import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


# ... (Your preprocess_text function remains the same) ...

# Apply the preprocessing to training and test data
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Create a TfidfVectorizer to convert text to numerical features
vectorizer = TfidfVectorizer()

# Fit the vectorizer to your training data and transform both train and test data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ... (Rest of your code, including GridSearchCV, remains the same) ...

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


# Define a function to preprocess text
def preprocess_text(text):
    # Check if the input is a string
    if isinstance(text, str):
        # Remove special characters and digits, but preserve spaces
        text = re.sub(r'[^\w\s]', '', text)  # Changed regex to preserve spaces
        # Convert text to lowercase
        text = text.lower()
        # Remove stop words but keep at least one word
        words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
        if not words:  # If all words are stop words, keep some common words
            words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS or word in ['the', 'a', 'an']]  # Add back common words if needed
        text = ' '.join(words)
        return text
    else:
        # Handle non-string inputs (e.g., return an empty string or the original value)
        return ''  # Or return str(text) if you want to preserve non-string values as strings

# Apply the preprocessing to training and test data
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Create a TfidfVectorizer to convert text to numerical features
vectorizer = TfidfVectorizer(min_df=2) # ignore terms that appear in less than 2 documents

# Fit the vectorizer to your training data and transform both train and test data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ... (Rest of your code, including GridSearchCV, remains the same) ...

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


# Define a function to preprocess text
def preprocess_text(text):
    # Check if the input is a string
    if isinstance(text, str):
        # Remove special characters and digits, preserving spaces and important characters
        text = re.sub(r"[^a-zA-Z0-9 ]", '', text)  # Changed regex to preserve spaces and alphanumeric characters
        # Convert text to lowercase
        text = text.lower()
        # Remove stop words but keep at least one word
        words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
        if not words:  # If all words are stop words, keep the original text
            words = text.split()  # or return a placeholder like 'empty_text'
        text = ' '.join(words)
        return text
    else:
        # Handle non-string inputs (e.g., return an empty string or the original value)
        return ''  # Or return str(text) if you want to preserve non-string values as strings

# Apply the preprocessing to training and test data
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Create a TfidfVectorizer to convert text to numerical features
# Lower min_df to 1 to include terms that appear in only one document
vectorizer = TfidfVectorizer(min_df=1)

# Fit the vectorizer to your training data and transform both train and test data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ... (Rest of your code, including GridSearchCV, remains the same) ...

import joblib

# Save model and vectorizer
joblib.dump(best_model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')


from sklearn.linear_model import LogisticRegression
best_model = LogisticRegression()
best_model.fit(X_train_vec, y_train)  # Ensure you replace X_train_vec and y_train with your actual variables


import joblib

# Save model and vectorizer
joblib.dump(best_model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)  # Transform the test data without refitting


from sklearn.linear_model import LogisticRegression

best_model = LogisticRegression()
best_model.fit(X_train_vec, y_train)


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)  # Transform the test data without refitting


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize vectorizer with adjusted parameters
vectorizer = TfidfVectorizer(
    stop_words=None,  # Or use a custom stop word list if needed
    min_df=1,          # Consider words appearing in at least 1 document
    max_df=1.0,       # Consider words appearing in at most 100% of documents
    token_pattern=r"(?u)\b\w+\b" # Consider words with at least one alphanumeric character
)

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)  # Transform the test data without refitting

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize vectorizer with adjusted parameters
vectorizer = TfidfVectorizer(
    stop_words='english',  # Use 'english' for built-in stop words or provide a list
    min_df=1,          # Consider words appearing in at least 1 document
    max_df=1.0,       # Consider words appearing in at most 100% of documents
    #token_pattern=r"(?u)\b\w+\b",  # Consider words with at least one alphanumeric character # remove the explicit pattern
    #analyzer='word' # Use the word level analyzer
)

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)  # Transform the test data without refitting

# Remove any rows in X_train where the text is empty or just whitespace
X_train = [text for text in X_train if text.strip() != ""]


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


print("Sample data in X_train:")
print(X_train[:10])  # Adjust the slice as needed to see more samples


X_train = [text for text in X_train if text.strip()]


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


print("Sample data in X_train:", X_train[:50])  # Check first 10 entries


print("Balanced Data Sample:", balanced_data.head())
print("X_train:", X_train[:5])  # Check the first 5 entries


balanced_data['text'] = balanced_data['text'].str.strip()  # Remove whitespace
balanced_data = balanced_data[balanced_data['text'] != '']  # Filter out empty rows


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    balanced_data['text'], balanced_data['label'], test_size=0.2, random_state=42
)


print("X_train sample after cleaning:", X_train[:5])


# Remove NaN values
balanced_data = balanced_data.dropna(subset=['text'])


X_train, X_test, y_train, y_test = train_test_split(
    balanced_data['text'], balanced_data['label'], test_size=0.2, random_state=42
)
print("Sample data in X_train after cleaning:", X_train[:5])


from sklearn.model_selection import train_test_split

# Drop rows where 'text' is NaN
balanced_data = balanced_data.dropna(subset=['text'])

# Re-run the train-test split
X_train, X_test, y_train, y_test = train_test_split(
    balanced_data['text'], balanced_data['label'], test_size=0.2, random_state=42
)

# Check the first few entries to ensure they contain data
print("Sample data in X_train after cleaning:", X_train[:5])


# Check the number of entries in balanced_data after dropping NaN values
print("Number of entries in balanced_data after dropping NaNs:", len(balanced_data))

# If there are any entries, continue with the split
if len(balanced_data) > 0:
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_data['text'], balanced_data['label'], test_size=0.2, random_state=42
    )
    print("Sample data in X_train after cleaning:", X_train[:5])
else:
    print("No data available for training and testing after removing NaN values.")


# Check for NaN values in each column
print("NaN counts in each column of balanced_data:")
print(balanced_data.isna().sum())

# Display a few rows of balanced_data to inspect
print("Sample data from balanced_data:")
print(balanced_data.head(10))


# Import necessary library
import pandas as pd

# Load the initial dataset with explicit column names
# Assuming 'twitter_validation.csv' contains text and label columns
initial_data = pd.read_csv('/content/twitter_validation.csv', names=['text', 'label'])

# Load the positive reviews dataset with a single column 'text' and add a 'label' column
# Assuming positive_examples.txt contains only the text of positive reviews
positive_reviews = pd.read_csv('/content/positive_examples.txt', header=None, names=['text'])
positive_reviews['label'] = 1  # Set label to 1 for positive sentiment

# Combine the initial data (negative and positive reviews) into one balanced dataset
balanced_data = pd.concat([initial_data, positive_reviews], ignore_index=True)

# Display a sample of the balanced data to verify structure
print("Sample data from balanced_data:")
print(balanced_data.head())

# Check for NaN values in each column
print("NaN counts in each column of balanced_data:")
print(balanced_data.isna().sum())


# Re-load the initial dataset with explicit column names
initial_data = pd.read_csv('/content/twitter_validation.csv', names=['text', 'label'])

# Load the positive reviews dataset with the correct column name
positive_reviews = pd.read_csv('/content/positive_examples.txt', header=None, names=['text'])
positive_reviews['label'] = 1  # Assuming 1 indicates Positive sentiment

# Combine initial data with positive reviews
balanced_data = pd.concat([initial_data, positive_reviews], ignore_index=True)

# Display a sample of the balanced data to verify structure
print("Sample data from balanced_data:")
print(balanced_data.head())


# Check for NaN values in each column
print("NaN counts in each column of balanced_data:")
print(balanced_data.isna().sum())


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    balanced_data['text'], balanced_data['label'], test_size=0.2, random_state=42
)

# Check the first few entries in X_train and y_train to ensure they contain data
print("Sample data in X_train:", X_train[:5])
print("Sample labels in y_train:", y_train[:5])


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data, transform the test data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train_vec, y_train)


from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test set
y_pred = model.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))


# Verify unique labels and their counts
print("Unique labels in y_train:", y_train.value_counts())
print("Unique labels in y_test:", y_test.value_counts())


# Display a sample of the vectorized data
print("Sample of X_train vectorized data:")
print(X_train_vec[:5].toarray())  # .toarray() converts sparse matrix to dense matrix for inspection

print("Sample of X_test vectorized data:")
print(X_test_vec[:5].toarray())


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Option 1: Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Option 2: Multinomial Naive Bayes
# model = MultinomialNB()
# model.fit(X_train_vec, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


from sklearn.feature_extraction.text import TfidfVectorizer

# Adjust vectorizer parameters
vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


from sklearn.model_selection import cross_val_score

# Use cross-validation on the model
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())


from sklearn.model_selection import cross_val_score

# Adjust 'cv' to a smaller value if needed based on the size of your data
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=3, scoring='accuracy')  # Use 3 folds instead of 5
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())


y_train = y_train.astype('category')

cv_scores = cross_val_score(model, X_train_vec, y_train, cv=3, scoring='accuracy')  # Use 3 folds instead of 5

import pandas as pd # Assuming you're already importing pandas for your data handling
from sklearn.model_selection import StratifiedKFold

# 1. Reduce number of folds if you have target values of float or int type:
# Find the minimum number of samples in any class
min_samples_per_class = pd.Series(y_train).value_counts().min()
# Set cv to the minimum or 2, whichever is larger to avoid issues with extremely small classes.
cv = min(min_samples_per_class, 3)
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='accuracy')

# 2. Consider changing target type to category if you haven't:
# y_train = y_train.astype('category')
# cv_scores = cross_val_score(model, X_train_vec, y_train, cv=3, scoring='accuracy')

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Assuming 'model' is your classifier, 'X_train_vec' is your feature matrix, and 'y_train' is your target variable
# Create a StratifiedKFold object with a smaller number of splits (e.g., 2)
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # Reduced splits to 2, added shuffling and random state

# Use the StratifiedKFold object in cross_val_score
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Assuming 'model' is your classifier, 'X_train_vec' is your feature matrix, and 'y_train' is your target variable

# 1. Calculate the minimum number of samples in any class
min_samples_per_class = pd.Series(y_train).value_counts().min()

# 2. Set n_splits to the minimum or 2, whichever is smaller
n_splits = min(min_samples_per_class, 2)

# 3. Create a StratifiedKFold object with the calculated n_splits
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 4. Use the StratifiedKFold object in cross_val_score
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

min_samples_per_class = pd.Series(y_train).value_counts().min()
n_splits = min(min_samples_per_class, 2)

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Assuming 'model' is your classifier, 'X_train_vec' is your feature matrix, and 'y_train' is your target variable

# 1. Calculate the minimum number of samples in any class
min_samples_per_class = pd.Series(y_train).value_counts().min()

# 2. Set n_splits to the minimum or 2, whichever is larger (corrected from smaller)
n_splits = max(min_samples_per_class, 2)  # Changed min to max

# 3. Create a StratifiedKFold object with the calculated n_splits
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 4. Use the StratifiedKFold object in cross_val_score
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Assuming 'model' is your classifier, 'X_train_vec' is your feature matrix, and 'y_train' is your target variable

# 1. Calculate the minimum number of samples in any class
min_samples_per_class = pd.Series(y_train).value_counts().min()

# 2. Set n_splits to the minimum between the minimum samples per class and 2 to ensure at least 1 sample in each fold
n_splits = min(min_samples_per_class, 2)  # Changed max to min to make sure n_splits is less than or equal to the smallest class size

# 3. Create a StratifiedKFold object with the calculated n_splits
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 4. Use the StratifiedKFold object in cross_val_score
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# Assuming "model" is your classifier, "X_train_vec" is your feature matrix, and "y_train" are your labels

# 1. Calculate the minimum number of samples in any class
min_samples_per_class = pd.Series(y_train).value_counts().min()

# 2. Set n_splits to the minimum between the minimum samples per class and 5, with a lower limit of 2
n_splits = max(2, min(min_samples_per_class, 5))

# 3. Create a StratifiedKFold object with the calculated n_splits
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 4. Use the StratifiedKFold object in cross_val_score
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", np.mean(cv_scores))


from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# Assuming "model" is your classifier, "X_train_vec" is your feature matrix, and "y_train" are your labels

# 1. Calculate the minimum number of samples in any class
min_samples_per_class = pd.Series(y_train).value_counts().min()

# 2. Ensure n_splits is at least 2 but no greater than the minimum samples per class
if min_samples_per_class >= 2:
    n_splits = min(5, min_samples_per_class)  # Set a max of 5 folds if possible

    # 3. Create a StratifiedKFold object with the calculated n_splits
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 4. Use the StratifiedKFold object in cross_val_score
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='accuracy')
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Accuracy:", np.mean(cv_scores))
else:
    print("Not enough samples in one or more classes to perform 2 or more splits in cross-validation.")
    print("Consider collecting more data or using a simpler validation strategy.")


# Train the model on the training set
model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vec)

# Calculate accuracy and classification report
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_report_text = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_text)


import joblib

# Save the model
joblib.dump(model, 'sentiment_model.pkl')
# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')


# Save this code in a file (e.g., app.py), and run `streamlit run app.py` in the terminal.

import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Sentiment Analysis Tool")
user_input = st.text_input("Enter a sentence to analyze its sentiment:")

if user_input:
    user_input_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vec)
    st.write("Predicted Sentiment:", "Positive" if prediction[0] == 1 else "Negative")



from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Sample Dataset
messages = [
    "Win a free prize now",
    "Claim your cash reward",
    "Congratulations you won a lottery",
    "Click here to claim your gift",
    "Get a free mobile recharge",
    "Urgent offer buy now",
    "You have won a free ticket",
    "Claim your bonus today",
    "Free entry in a contest",
    "Earn money quickly click here",
    "Hi how are you",
    "Are we meeting tomorrow",
    "Please call me later",
    "I will reach home soon",
    "Your class starts at ten",
    "Can you send the notes",
    "Thank you for your help",
    "Let us meet this evening",
    "I am going to college",
    "Please bring the book"
]

labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Normal", "Normal", "Normal", "Normal",
    "Normal", "Normal", "Normal", "Normal",
    "Normal", "Normal"
]

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

# 3. Build and Train Model
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# 4. Model Evaluation
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, pos_label="Spam", zero_division=0
)
recall = recall_score(
    y_test, predictions, pos_label="Spam", zero_division=0
)
f1 = f1_score(
    y_test, predictions, pos_label="Spam", zero_division=0
)

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test, predictions,
    labels=["Normal", "Spam"]
))

print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# 5. Input Validation
def validate_message(message):

    if not message.strip():
        return "Please enter a message."

    if len(message.strip()) < 2:
        return "Please enter a longer message."

    if not any(char.isalpha() for char in message):
        return "Please enter valid text."

    if len(message) > 2000:
        return "Message is too long. Maximum 2000 characters."

    return None

# 6. Intelligent Prediction
def predict_message(message):

    error = validate_message(message)

    if error:
        print("Error:", error)
        return

    prediction = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]
    confidence = max(probabilities) * 100

    print("\n===== PREDICTION RESULT =====")
    print("Message:", message)
    print("Prediction:", prediction)
    print("Confidence:", round(confidence, 2), "%")

    if prediction == "Spam":
        print("Warning: Suspicious message detected!")
    else:
        print("This message is classified as Normal.")

# 7. Evaluation Examples
print("\n===== TEST EXAMPLES =====")

test_messages = [
    "Congratulations! Claim your free prize now",
    "Are you coming to college today?",
    "Claim your cash reward immediately",
    "Please call me when you reach home"
]

for msg in test_messages:
    print("\nInput:", msg)
    predict_message(msg)

# 8. Failure Case Testing
print("\n===== ERROR HANDLING TEST =====")

failure_cases = [
    "",
    "7",
    "12345 !!!"
]

for msg in failure_cases:
    print("\nInput:", repr(msg))
    error = validate_message(msg)

    if error:
        print("Handled:", error)

# 9. User Interface
print("\n===== SMS SPAM CLASSIFIER =====")

while True:

    new_message = input("\nEnter a message (or type exit): ")

    if new_message.strip().lower() == "exit":
        print("Thank you for using the classifier!")
        break

    predict_message(new_message)

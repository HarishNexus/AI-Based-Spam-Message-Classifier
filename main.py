from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Small sample dataset
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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    messages, labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

# Build and train the model
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# Predict a new message
new_message = ["Congratulations! Claim your free prize now"]
result = model.predict(new_message)[0]

print("Message:", new_message[0])
print("Prediction:", result)

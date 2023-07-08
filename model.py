from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

upi = 'UPI-KULDEEP S O CHATTAR -KULDEEP9650@YBL -PUNB0679900-232343088841-NA (Ref# 232343088841)'
trimmed_upi = upi[4:upi.index('@')]

print(trimmed_upi)

data = {
    'narration': [
            "Dr.URMILA SHARMA",
            "Hospital",
            "Pharmacy",
            "Clinic",
            "Dental",
    "AIRTEL-PAYAIR7673",
            "JIO-PAYAIR7673",
    "BAJAJ PRIME FUELS HP",
            "Indian Oil",
            "Shell",
            "Bharath Petroleaum",
            "Bharath Petroleaum",
            "Hindustan Petroleaum",
            "ongc petroleaum",
    "UPKAR RAGHAV-UPKARRAGHAV0",
          "Super Market",
            "MORE Market",
            "BigBazaar Market",
            "D Mart Market",
    "AIRTEL-PAYAIR767",
    "BHAGAVAN DAS",
    "KULDEEP-9650946599",
    "Dr.URMILA SHARMA-PAYTM",
    "ECOMEXPRESS-PAYPHI.ECOMEXPRESS",
    "NAVAL KUMAR",
    "RG SONS AND OTHERS L",
    "VIJAY KUMAR-PAYTM-6244823",
    "LAXMI DAS-PAYTM-70579281",
    "LAXMI DAS-PAYTM-70579281",
    "LAXMI DAS-PAYTM-70579281",
    "BHAGAVAN DAS-PAYTM-67127062",
    "AIRTEL-PAYAIR7673",
    "AIRTEL-PAYAIR7673",
            "SHADOWFAX TECHNOLOGI",
            "Techno Mobiles",
            "Sri Surya Electronics",
            "Modern softwares",
            "Reddy hardwares",
            "vineeth hardwares",
            "Rao technology",
      #               "SALARY SHANKAR FENESTRATIONS & GLASSES I NDIA P LTD",
      #   "Jaikanth Glasses and furnitures",
      #   "Saravanna Stores and funitures",
      #   "Chairs and furnitures",
      #   "Vishal Beds and Sofas",
      #  "Sleepwell furnitures and matress",
    ],
    'category': [
        'Healthcare',
        'Healthcare',
        'Healthcare',
        'Healthcare',
        'Healthcare',
'Mobile Recharge',
        'Mobile Recharge',
'Transportation',
        'Transportation',
        'Groceries',
        'Transportation',
        'Transportation',
        'Transportation',
        'Transportation',
'Groceries',
        'Groceries',
        'Groceries',
        'Groceries',
        'Groceries',
'Mobile Recharge',
'Groceries',
'Healthcare',
'Healthcare',
'Transportation',
'Transportation',
'Transportation',
'Transportation',
'Groceries',
'Groceries',
'Groceries',
'Transportation',
'Mobile Recharge',
'Mobile Recharge',
        'Electronics',
         'Electronics',
         'Electronics',
         'Electronics', 
        'Electronics',
         'Electronics',
          'Electronics',
        #         'Furnitures',
        # 'Furnitures',
        # 'Furnitures',
        # 'Furnitures',
        # 'Furnitures',
        # 'Furnitures',
    ]
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('narration_dataset.csv', index=False)


# Load the dataset into a pandas DataFrame
df = pd.read_csv('narration_dataset.csv')

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df['narration'], df['category'], test_size=0.2, random_state=42)
# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the training data
train_features = vectorizer.fit_transform(train_data)

# Transform the testing data using the same vectorizer
test_features = vectorizer.transform(test_data)



# Initialize the Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model on the training data
model.fit(train_features, train_labels)

# Predict the categories of the testing data
predictions = model.predict(test_features)

# Calculate the accuracy of the model
accuracy = accuracy_score(test_labels, predictions)
print(f"Model accuracy: {accuracy}")


# Preprocess the narration prompt
prompt = "alaya hospital-BHARATPE30059351755@YESBANKLTD-YESB0YESUPI-233325041546 -PAY BY WHATSAPP (Ref# 233325041546)"
preprocessed_prompt = vectorizer.transform([prompt])

# Make a prediction using the trained model
predicted_category = model.predict(preprocessed_prompt)[0]

# Print the predicted category
print(f"The predicted category for the narration prompt '{prompt}' is: {predicted_category}")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/quiz')
def quiz():

    return render_template('quiz.html')


@app.route('/saving-quest')
def savingquest():

    return render_template('saving-quest.html')



@app.route('/challenges')
def challenges():

    return render_template('challenges.html')




@app.route('/predict')
def predict():

    return render_template('predict.html', predicted_category=predicted_category)

if __name__ == '__main__':
    app.run(debug=True)


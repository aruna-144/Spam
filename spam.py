# importing necessary modules
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score

data = pd.read_csv('mail_data.csv') #loading the dataset 
print(data)

data.info()
data.shape #shape of the dataset
data.head() #prints 1st 5 rows
data.isnull().sum() #checking for null values
data.drop_duplicates(inplace=True) #removing duplicate values
data.head(10)
data.shape #shape of the dataset after removing the duplicates

# Assuming 'data' is your DataFrame with 'Category' column
category_counts = data['Category'].value_counts() # Count occurrences of each category

# Create the pie chart using category counts
plt.pie(category_counts, labels=category_counts.index,radius = 1.0,shadow = True,explode = [0.05,.2],autopct='%1.1f%%') 
plt.show()

#label encoding
data.loc[data['Category']=='spam','Category'] = 1
data.loc[data['Category']=='ham','Category'] = 0 
print(data)

data.head()

#separating the text as data and labels
x = data['Message']
y = data['Category']
x.head()
y.head()

#splitting the dataset into training & testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 3)

#displays how much data is used for testing
x_test.head()
y_test.head()

# Converting text data into numerical values to use as input.
feature_extraction = TfidfVectorizer(min_df = 1,stop_words = 'english',lowercase = True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

print(x_train_features)

model = LogisticRegression()
model.fit(x_train_features,y_train) 



prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train,prediction_on_training_data)
print("Accuracy on training data : ", accuracy_on_training_data)

prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test,prediction_on_test_data)
print("Accuracy on test data : ", accuracy_on_test_data)

# Calculate F1 Score
f1 = f1_score(y_test, prediction_on_test_data)

# Calculate Precision
precision = precision_score(y_test, prediction_on_test_data)

# Calculate Recall
recall = recall_score(y_test, prediction_on_test_data)

# Output the results
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

cm = confusion_matrix(y_test, prediction_on_test_data)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

def func(data):
    message = data.get('message', '')
    input_data_features = feature_extraction.transform([message])
    prediction = model.predict(input_data_features)
    return prediction[0]


from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/process', methods=['POST'])
def process_data():
    data = request.json
    prediction = func(data)
    result = "Spam Mail" if prediction == 1 else "Ham Mail"
    return jsonify({"message": result}), 200

if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd

messages = pd.read_csv('spam.csv', skiprows = 1,  names = ['label', 'message']) 
## The initial column names were 'v1' and 'v2'. so I skipped that column and then gave new column names 'label' and 'message'.


## Data Clearning and Preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()   ## Stemmer object (not going to use this)
lemmatizer = WordNetLemmatizer() ## Lemmatizer object  (this is going to be used)
corpus = []

for i in range(len(messages)):
    result = re.sub('[^a-zA-Z]', ' ', messages['message'][i])  ## remove everything except from a-z and A-Z. also selected 'message' column of 'message dataframe'
    result = result.lower()
    result = result.split()  ## It will give you the list of result
    
    result = [lemmatizer.lemmatize(word) for word in result if word not in set(stopwords.words('english'))]
    result = ' '.join(result)
    corpus.append(result)
    
    
# Createing the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()

'''## We can use TF-IDF too. here is the code for it
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_v = TfidfVectorizer(max_features = 5000)
X = tf_idf_v.fit_transform(corpus).toarray()'''



y = pd.get_dummies(messages['label']) ## Here we get 2 columns. 'ham' and 'spam'.
y = y.iloc[:, 1].values ## but instead of 2 columns, we can present output in one column. where 'ham = 0' and 'spam = 1'



## Test Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training model using Naive bayes calssifier
# It's a classification technique and it workes on probability
from sklearn.naive_bayes import MultinomialNB ## MultinomialNB library works for single and multiple classes.
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test) ## now compare the 'y_test' and 'y_pred'. They look almost same. but still we need to compare all the rows.


## Confusion Matrix
## It gives you 2x2 matrix and shows you that how many no. of elements are correctly predicted.
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test, y_pred)

## TO check the accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy Score: ', accuracy) ## we got 98% accuracy. Hurrah!



'''-----------------------------------------------------------------------------------------
how to check the correct prediction from confusion matrix?
this is the matrix I got:
                                
                             columns
      
                          |  0    |   1
                        --|-------|--------   
                rows    0 |  939  |  10
                        --|-------|--------    
                        1 |  8    |  158
                        --|-------|--------
                        

note: rows 0 and 1 are actual output
and  columns 0 and 1 are predicted output.

1. 0 and 0 cell/box giving us 939 value
2. 1 and 1 cell/box giving us 185 value

so total 939 + 185 = 1124 are correctly predicted 
and 8 + 10 = 18 are not correctly predicted.
-----------------------------------------------------------------------------------------'''
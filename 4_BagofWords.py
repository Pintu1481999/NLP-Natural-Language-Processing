# Bag of Words

import nltk

paragraph = """  My biological mother found out later that my mother had never graduated from college and that my 
                 father had never graduated from high school. She refused to sign the final adoption papers. 
                 She only relented a few months later when my parents promised that I would go to college.
                This was the start in my life. And 17 years later I did go to college. But I naively chose a 
                college that was almost as expensive as Stanford, and all of my working-class parents’ savings 
                were being spent on my college tuition. After six months, I couldn’t see the value in it. 
                I had no idea what I wanted to do with my life and no idea how college was going to help 
                me figure it out. And here I was spending all of the money my parents had saved their 
                entire life. So I decided to drop out and trust that it would all work out OK.
                It was pretty scary at the time, but looking back it was one of the best decisions I ever made.
                The minute I dropped out I could stop taking the required classes that didn’t interest me, 
                and begin dropping in on the ones that looked far more interesting.It wasn’t all romantic. 
                I didn’t have a dorm room, so I slept on the floor in friends’ rooms, I returned coke 
                bottles for the $0.05 deposits to buy food with, and I would walk the 7 miles across 
                town every Sunday night to get one good meal a week at the Hare Krishna temple. I loved it.               
            """
            
# cleaning the part
import re
from nltk.corpus import stopwords              
from nltk.stem.porter import PorterStemmer     
from nltk.stem import WordNetLemmatizer

#stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()  
sentences = nltk.sent_tokenize(paragraph)
corpus = []  ## after clearning the paragraph, will store it in this corpus list. then we will be see the diff. between sentences and corpus

for i in range(len(sentences)):
    result = re.sub('[^a-zA-Z]', ' ', sentences[i]) ## remove everything except the words between a-z and A-Z.
    result = result.lower()
    result = result.split() ## it will give us list.
    result = [lemmatizer.lemmatize(word) for word in result if word not in set(stopwords.words('english'))]
    result = ' '.join(result)
    corpus.append(result)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer  
cv = CountVectorizer() 
bow = cv.fit_transform(corpus).toarray()

## It will convert the whole corpus into vector format. and will assign the frequencies of words present in corpus (i.e. 0, 1, 2, etc...)
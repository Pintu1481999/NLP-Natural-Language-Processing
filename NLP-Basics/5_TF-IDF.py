import nltk

paragraph = """My biological mother found out later that my mother had never graduated from college and that 
            my father had never graduated from high school. She refused to sign the final adoption papers. 
            She only relented a few months later when my parents promised that I would go to college.
            This was the start in my life.And 17 years later I did go to college. But I naively chose a 
            college that was almost as expensive as Stanford, and all of my working-class parents’ savings 
            were being spent on my college tuition. After six months, I couldn’t see the value in it. 
            I had no idea what I wanted to do with my life and no idea how college was going to help me figure it out.
            And here I was spending all of the money my parents had saved their entire life. So I decided to drop out 
            and trust that it would all work out OK.
            It was pretty scary at the time, but looking back it was one of the best decisions I ever made.
            The minute I dropped out I could stop taking the required classes that didn’t interest me, and begin 
            dropping in on the ones that looked far more interesting. It wasn’t all romantic. I didn’t have a dorm room, 
            so I slept on the floor in friends’ rooms, I returned coke bottles for the $0.05 deposits to buy food with, 
            and I would walk the 7 miles across town every Sunday night to get one good meal a week at the 
            Hare Krishna temple. I loved it. And much of what I stumbled into by following my curiosity and 
            intuition turned out to be priceless later on.
            Let me give you one example: Reed College at that time offered perhaps the best calligraphy instruction 
            in the country. Throughout the campus every poster, every label on every drawer, was beautifully hand 
            calligraphed. Because I had dropped out and didn’t have to take the normal classes, I decided to take a 
            calligraphy class to learn how to do this. I learned about serif and san serif typefaces, about varying 
            the amount of space between different letter combinations, about what makes great typography great.
            It was beautiful, historical, artistically subtle in a way that science can’t capture, and I found it 
            fascinating. None of this had even a hope of any practical application in my life. But 10 years later, 
            when we were designing the first Macintosh computer, it all came back to me. And we designed it all 
            into the Mac. It was the first computer with beautiful typography."""
            
# Cleaning the paragraph
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stemmer = PorterStemmer() 
lemmatizer = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []

for i in range(len(sentences)):
    result = re.sub('[^a-zA-Z]', ' ', sentences[i])
    result = result.lower()
    result = result.split()
    result = [lemmatizer.lemmatize(word) for word in result if word not in set(stopwords.words('english'))] ## we used set() here. so that it will select only unique words.
    result = ' '.join(result)
    corpus.append(result)
    
    
# Creating TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_cv = TfidfVectorizer()
final_result = tf_idf_cv.fit_transform(corpus).toarray()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
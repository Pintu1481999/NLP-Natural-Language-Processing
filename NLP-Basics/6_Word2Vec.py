# Word2Vec

import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords


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
            
## Data Clearning and Preprocessing
import re

text = re.sub(r'\[[0-9]*\]', ' ', paragraph)  ## removing numbers, punctuations
text = re.sub(r'\s+', ' ', text)       ## removing unnecessary spaces
text = text.lower()
text = re.sub(r'\d', ' ', text) ## Matches any Unicode digit (which includes [0-9], and also many other digit characters)
text = re.sub(r'\s+', ' ', text) ## Matches Unicode whitespace characters (which includes [ \t\n\r\f\v], and also many other characters


## Preparing dataset
sentences = nltk.sent_tokenize(text)  # paragraph into sentences

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]  # sentences into words

for i in range(len(sentences)):
    sentences[i] = [ word for word in sentences[i] if word not in set(stopwords.words('english'))]
    
## Training Word2Vec model
model = Word2Vec(sentences, min_count = 1) ##word should be presented more than 1 time.

words = model.wv.vocab   ## vocabularies in Word2Vec model

## Finding word vectors
vector = model.wv['college']  # vector of 100 dimentions for the word 'college'. 

## Most similar words
similar = model.wv.most_similar('college')  # similar word to the 'college'
# Tokenization of paragraph or corpus.

import nltk
nltk.download()

paragraph = """Natural Language Processing (NLP) is a subfield of computer science, 
               artificial intelligence, information engineering, and human-computer interaction.
               This field focuses on how to program computers to process and analyze large amounts of natural language 
               data. It is difficult to perform as the process of reading and understanding languages is far more 
               complex than it seems at first glance.
               Tokenization is the process of tokenizing or splitting a string, text into a list of tokens. 
               One can think of token as parts like a word is a token in a sentence, and a sentence is a 
               token in a paragraph.
               How sent_tokenize works ?
               The sent_tokenize function uses an instance of PunktSentenceTokenizer from the nltk.tokenize.punkt 
               module, which is already been trained and thus very well knows to mark the end and beginning of 
               sentence at what characters and punctuation.
            """
            
# Tokenizing sentences
# sent_tokenize uses and intance of PunktSentenceTokenizer which is already trained and wll known
# to mark the end and beginning of sentence at what characters and punctuations. using REGEX.
sentences = nltk.sent_tokenize(paragraph)


# Tokeninzing sentences into words 
# it separates the words using punctuations and spaces.
words = nltk.word_tokenize(paragraph)   ## even commas and special charcters are considered as a word.


## ----------------------------------EXTRA----------------------------------
'''
# If we have huge chunks of data then it's efficient to use  "PunkSentenceTokenizer"
import nltk.data
tokenizer = nltk.data.load('tokenizers/punk/PY3/english.pickle')
tokenizer.tokenize(paragraph)

# tokenization in different language
import nltk.data
spanish_tokenizer = nltk.data.load('tokenizers/punk/PY3/spanish.pickle')
text = 'Hola amigo. Estoy bien'
spanish_tokenizer.tokenize(text)'''
##----------------------------------------------------------------------------


from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import SnowballStemmer
import string

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text

def preprocessing_function(text: str) -> str:

    # Begin your code (Part 0)
    text = text.lower() # lower cases
    text = text.replace("<br />", " ") # Remove <br />
    # text = remove_stopwords(text) # Remove stopwords
    text = "".join([char for char in text if (char not in string.punctuation)]) # Remove punctuations
    english_stemmer = SnowballStemmer(language='english') # SnowballStemmer
    text = " ".join([english_stemmer.stem(i) for i in text.split()]) 
    preprocessed_text = text
    # End your code
    return preprocessed_text

if __name__ == "__main__":
    s = "It is a truth universally acknowledged that<br /> a single man in possession of a good fortune must be in want of a wife."
    print(preprocessing_function(s))
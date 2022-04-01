from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import SnowballStemmer

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
    preprocessed_text = text.replace("<br />", "")
    english_stemmer = SnowballStemmer(language='english')
    preprocessed_text = english_stemmer.stem(preprocessed_text)
    preprocessed_text = remove_stopwords(preprocessed_text)
    # End your code
    print(preprocessed_text)
    return preprocessed_text
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
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

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def preprocessing_function(text: str) -> str:

    # Begin your code (Part 0)
    text = text.lower()
    text = text.replace("<br />", " ")
    # text = remove_stopwords(text)
    # stop_word_list = stopwords.words('english')
    # text = "".join([char for char in text if (char not in ['.', ',', "'", '"'])])
    text = "".join([char for char in text if (char not in string.punctuation)])
    preprocessed_text = text.split()
    english_stemmer = SnowballStemmer(language='english')
    preprocessed_text = [english_stemmer.stem(i) for i in preprocessed_text]
    # preprocessed_text = [i for i in preprocessed_text if i not in stop_word_list]
    # text = [english_stemmer.stem(i) for i in text]
    # lemmatizer = WordNetLemmatizer()
    # preprocessed_text = []
    # tags = pos_tag(text)
    # for tag in tags:
    #     wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
    #     preprocessed_text.append(lemmatizer.lemmatize(tag[0], wordnet_pos))
    preprocessed_text = " ".join(preprocessed_text)
    # End your code
    return preprocessed_text

if __name__ == "__main__":
    s = "It is a truth universally acknowledged that<br /> a single man in possession of a good fortune must be in want of a wife."
    print(preprocessing_function(s))
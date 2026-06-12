import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def download_nltk_resources():
    """Mengunduh pustaka NLTK secara otomatis jika belum ada"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)

def clean_text(text):
    """Pipeline pembersihan teks sesuai dengan eksperimen"""
    # 1. Menghapus URL/Link
    text = re.sub(r'http\S+', '', text)

    # 2. Menghapus Mention (@username)
    text = re.sub(r'@\w+', '', text)

    # 3. Menghapus Hashtag (#)
    text = re.sub(r'#', '', text)

    # 4. Menghapus angka dan tanda baca (Hanya menyisakan huruf a-z dan spasi)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 5. Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def buang_stopword(tokens):
    """Menghapus kata-kata umum (stopword) tapi tetap mempertahankan kata 'not'"""
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')
    return [kata for kata in tokens if kata not in stop_words]
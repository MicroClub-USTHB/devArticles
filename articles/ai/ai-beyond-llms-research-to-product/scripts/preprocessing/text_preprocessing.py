"""
╔══════════════════════════════════════════════════════════════════════╗
║             text_preprocessing.py — NLP Preprocessing Pipeline      ║
║                                                                      ║
║  Everything you need to clean, normalize, tokenize, and vectorize   ║
║  text data before feeding it to a model.                             ║
║                                                                      ║
║  Pipeline order (almost always correct):                             ║
║    1. Clean (HTML, noise, special chars)                             ║
║    2. Normalize (lowercase, unicode, contractions)                   ║
║    3. Tokenize (split into words or subwords)                        ║
║    4. Filter (stopwords, punctuation, length)                        ║
║    5. Normalize tokens (stem or lemmatize)                           ║
║    6. Vectorize (BoW, TF-IDF, word embeddings)                       ║
║                                                                      ║
║  Dependencies:                                                       ║
║    pip install nltk spacy scikit-learn                               ║
║    python -m spacy download en_core_web_sm                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import re
import logging
import string
import unicodedata
import html as html_lib
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, Union


logger = logging.getLogger("text_preprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

#   Standalone, self-contained text cleaners
def remove_html_tags(text: str) -> str:
    """
    Strip HTML/XML tags from text.

    Also decodes HTML entities like &amp; → & and &lt; → <.

    Example:
        >>> remove_html_tags("<p>Hello &amp; <b>World</b></p>")
        'Hello & World'
    """
    text = html_lib.unescape(text)
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str, replace_with: str = "") -> str:
    """
    Remove URLs (http, https, www, ftp) from text.

    Args:
        text:         Input string.
        replace_with: Token to put in place of removed URLs.
                      Use " URL " to keep a placeholder for ML models
                      (signals "there was a URL here").

    Example:
        >>> remove_urls("Check out https://example.com for more info")
        'Check out  for more info'
        >>> remove_urls("Visit https://ai.com", replace_with=" URL ")
        'Visit  URL  for more info'
    """
    pattern = r"(https?://[^\s]+|www\.[^\s]+|ftp://[^\s]+)"
    return re.sub(pattern, replace_with, text)


def remove_emails(text: str, replace_with: str = "") -> str:
    """Remove email addresses from text."""
    return re.sub(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
                  replace_with, text)


def remove_phone_numbers(text: str, replace_with: str = "") -> str:
    """Remove common phone number patterns (US/international)."""
    patterns = [
        r"\+?\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}",
        r"\(\d{3}\)[\s\-]?\d{3}[\s\-]?\d{4}",
        r"\d{3}[\s\-]\d{3}[\s\-]\d{4}",
    ]
    for pat in patterns:
        text = re.sub(pat, replace_with, text)
    return text


def remove_special_characters(
    text: str,
    keep_punctuation: bool = False,
    keep_numbers: bool = True,
) -> str:
    """
    Remove non-alphanumeric characters.

    Args:
        text:             Input string.
        keep_punctuation: Keep . , ! ? ; : - ' "
        keep_numbers:     Keep digits.

    Example:
        >>> remove_special_characters("Hello! #world @2024")
        'Hello  world 2024'
        >>> remove_special_characters("Hello! #world", keep_punctuation=True)
        'Hello! world'
    """
    allowed = r"a-zA-Z\s"
    if keep_numbers:
        allowed += r"0-9"
    if keep_punctuation:
        allowed += r".,:;!?'\"\-"
    return re.sub(f"[^{allowed}]", " ", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces/newlines/tabs into a single space.

    Almost always needed as a final cleaning step.

    Example:
        >>> normalize_whitespace("Hello   world\\n\\nfoo")
        'Hello world foo'
    """
    return re.sub(r"\s+", " ", text).strip()


def to_lowercase(text: str) -> str:
    """
    Lowercase all characters.

    Warning: destroys named-entity casing (NER). If you plan to do
    entity recognition, lowercase AFTER NER or not at all.
    """
    return text.lower()


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Normalize Unicode to a canonical form.

    Handles accented characters, ligatures, and look-alike characters.

    Forms:
        NFC:  Canonical composition (most common — keeps accents).
        NFD:  Canonical decomposition (splits letters and diacritics).
        NFKC: Compatibility composition (normalizes "ﬁ" → "fi", etc.).
        NFKD: Compatibility decomposition.

    For most NLP: use NFKC (collapses look-alikes, good for search).

    Example:
        >>> normalize_unicode("café", "NFC")
        'café'
        >>> normalize_unicode("ﬁne", "NFKC")
        'fine'
    """
    return unicodedata.normalize(form, text)


def expand_contractions(text: str) -> str:
    """
    Expand English contractions to their full forms.

    Prevents "don't" and "do not" from being treated as different tokens
    when they mean the same thing.

    Example:
        >>> expand_contractions("I can't believe it's not butter")
        "I cannot believe it is not butter"
    """
    contractions = {
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bshan't\b": "shall not",
        r"\bn't\b": " not",
        r"\b're\b": " are",
        r"\b've\b": " have",
        r"\b'll\b": " will",
        r"\b'd\b": " would",
        r"\b'm\b": " am",
        r"\bit's\b": "it is",
        r"\bthat's\b": "that is",
        r"\bwhat's\b": "what is",
        r"\bwhere's\b": "where is",
        r"\bhe's\b": "he is",
        r"\bshe's\b": "she is",
        r"\blet's\b": "let us",
        r"\bI'm\b": "I am",
        r"\bI've\b": "I have",
        r"\bI'd\b": "I would",
        r"\bI'll\b": "I will",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

# Tokenization and filtering
def simple_tokenize(text: str) -> list[str]:
    """
    Basic whitespace + punctuation tokenizer.

    No dependencies required. Good enough for many use cases.
    For advanced tokenization, see nltk_tokenize() or spacy_tokenize().

    Example:
        >>> simple_tokenize("Hello, world! How's it going?")
        ['Hello', 'world', 'How', 's', 'it', 'going']
    """
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def nltk_tokenize(text: str, tokenizer: str = "word") -> list[str]:
    """
    Tokenize using NLTK.

    Args:
        text:      Input string.
        tokenizer: "word"     → word_tokenize (handles contractions better)
                   "sentence" → sent_tokenize (split into sentences)
                   "tweet"    → TweetTokenizer (preserves #hashtags @mentions)

    Example:
        >>> nltk_tokenize("I can't stop. It's amazing!", tokenizer="word")
        ['I', 'ca', "n't", 'stop', '.', 'It', "'s", 'amazing', '!']
    """
    try:
        import nltk
        if tokenizer == "word":
            return nltk.word_tokenize(text)
        elif tokenizer == "sentence":
            return nltk.sent_tokenize(text)
        elif tokenizer == "tweet":
            return nltk.tokenize.TweetTokenizer().tokenize(text)
    except ImportError:
        logger.warning("nltk not installed. Falling back to simple_tokenize.")
        return simple_tokenize(text)


def get_stopwords(language: str = "english", custom: Optional[list[str]] = None) -> set[str]:
    """
    Get stopword set for a given language.

    Stopwords are high-frequency words that usually carry little meaning
    (the, a, is, are, …). Removing them reduces feature space.

    Args:
        language: Language name for NLTK stopwords corpus.
                  Supported: english, french, german, spanish, italian, etc.
        custom:   Extra words to add to the stopword set.

    Example:
        stops = get_stopwords("english", custom=["however", "therefore"])
        tokens = [t for t in tokens if t not in stops]
    """
    try:
        from nltk.corpus import stopwords as nltk_stops
        stops = set(nltk_stops.words(language))
    except Exception:
        logger.warning("NLTK stopwords not available. Using minimal built-in set.")
        stops = {"the", "a", "an", "is", "it", "in", "on", "at", "to",
                 "of", "and", "or", "but", "for", "with", "this", "that",
                 "was", "are", "be", "been", "have", "has", "had", "not"}
    if custom:
        stops.update(w.lower() for w in custom)
    return stops


def filter_tokens(
    tokens: list[str],
    remove_stopwords: bool = True,
    remove_punctuation: bool = True,
    min_length: int = 2,
    max_length: int = 50,
    language: str = "english",
    custom_stopwords: Optional[list[str]] = None,
) -> list[str]:
    """
    Filter a token list by multiple criteria.

    Order of filtering:
      1. Remove pure punctuation tokens
      2. Remove stopwords
      3. Remove by length

    Args:
        tokens:            List of token strings.
        remove_stopwords:  Remove common stopwords.
        remove_punctuation: Remove tokens that are all punctuation.
        min_length:        Drop tokens shorter than this.
        max_length:        Drop tokens longer than this.
        language:          Language for stopwords.
        custom_stopwords:  Extra words to treat as stopwords.

    Example:
        tokens = nltk_tokenize("The quick brown fox jumps over the lazy dog")
        filtered = filter_tokens(tokens, remove_stopwords=True, min_length=3)
        # → ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
    """
    stops = get_stopwords(language, custom_stopwords) if remove_stopwords else set()
    result = []
    for tok in tokens:
        if remove_punctuation and all(c in string.punctuation for c in tok):
            continue
        if tok.lower() in stops:
            continue
        if not (min_length <= len(tok) <= max_length):
            continue
        result.append(tok)
    return result

#   Morphological normalization
def stem_tokens(tokens: list[str], stemmer: str = "porter") -> list[str]:
    """
    Reduce tokens to their root/stem form.

    Stemming: fast, rule-based, may produce non-words.
      "running" → "run", "studies" → "studi"

    Available stemmers:
        "porter":    Most common English stemmer (Porter, 1980).
        "snowball":  Improved Porter, also multilingual.
        "lancaster": Very aggressive stemmer.

    When to use stems vs lemmas:
        - Stems:  When speed matters and exact words aren't needed.
                  Good for search, IR, basic classification.
        - Lemmas: When grammatical correctness matters (sentiment, NLU).
                  See lemmatize_tokens().

    Example:
        >>> stem_tokens(["running", "studies", "happily"])
        ['run', 'studi', 'happili']
    """
    try:
        from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
        if stemmer == "porter":
            s = PorterStemmer()
        elif stemmer == "snowball":
            s = SnowballStemmer("english")
        elif stemmer == "lancaster":
            s = LancasterStemmer()
        else:
            raise ValueError(f"Unknown stemmer: {stemmer}")
        return [s.stem(t) for t in tokens]
    except ImportError:
        logger.warning("nltk not installed. Returning tokens unstemmed.")
        return tokens


def lemmatize_tokens(tokens: list[str], use_spacy: bool = True) -> list[str]:
    """
    Reduce tokens to their dictionary base form (lemma).

    Lemmatization uses morphological analysis and is more accurate
    than stemming but slower.

    "running" → "run", "studies" → "study", "better" → "good"

    Args:
        tokens:    List of strings.
        use_spacy: Use spaCy (recommended) or fall back to NLTK WordNetLemmatizer.

    Note (spaCy):
        Requires: pip install spacy && python -m spacy download en_core_web_sm

    Example:
        >>> lemmatize_tokens(["running", "studies", "better", "geese"])
        ['run', 'study', 'good', 'goose']
    """
    if use_spacy:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            doc = nlp(" ".join(tokens))
            return [token.lemma_ for token in doc]
        except Exception as e:
            logger.warning(f"spaCy unavailable ({e}). Falling back to NLTK.")

    try:
        from nltk.stem import WordNetLemmatizer
        import nltk
        nltk.download("wordnet", quiet=True)
        lem = WordNetLemmatizer()
        return [lem.lemmatize(t, pos="v") for t in tokens]
    except ImportError:
        logger.warning("Neither spaCy nor NLTK available. Returning tokens as-is.")
        return tokens

#  Vectorization
def build_vocabulary(
    corpus: list[list[str]],
    max_vocab: int = 10000,
    min_freq: int = 2,
) -> dict[str, int]:
    """
    Build a word → index vocabulary from a tokenized corpus.

    Special tokens (always included):
        <PAD>: 0  — padding for fixed-length batches
        <UNK>: 1  — unknown words not in vocabulary
        <BOS>: 2  — beginning of sequence
        <EOS>: 3  — end of sequence

    Args:
        corpus:    List of token lists (one per document).
        max_vocab: Maximum vocabulary size (excluding special tokens).
        min_freq:  Minimum frequency to include a word.

    Returns:
        Dict mapping word → integer index.

    Example:
        corpus = [["hello", "world"], ["hello", "there"]]
        vocab = build_vocabulary(corpus, min_freq=1)
        # {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
        #  "hello": 4, "world": 5, "there": 6}
    """
    counter = Counter(tok for doc in corpus for tok in doc)
    # Filter by frequency
    valid = [(w, c) for w, c in counter.items() if c >= min_freq]
    # Sort by frequency (most frequent first)
    valid.sort(key=lambda x: x[1], reverse=True)

    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    for word, _ in valid[:max_vocab]:
        if word not in vocab:
            vocab[word] = len(vocab)

    logger.info(f"Vocabulary size: {len(vocab)} tokens (min_freq={min_freq})")
    return vocab


def encode_tokens(
    tokens: list[str],
    vocab: dict[str, int],
    add_bos: bool = False,
    add_eos: bool = False,
    max_length: Optional[int] = None,
    pad: bool = True,
) -> list[int]:
    """
    Convert a list of tokens to integer IDs using a vocabulary.

    Handles:
      - Unknown words → <UNK> (index 1)
      - Optional BOS/EOS wrapping
      - Optional truncation and padding

    Args:
        tokens:     Token list.
        vocab:      Word → index mapping (from build_vocabulary).
        add_bos:    Prepend BOS token.
        add_eos:    Append EOS token.
        max_length: Truncate (and optionally pad) to this length.
        pad:        Pad with <PAD> (index 0) to max_length if shorter.

    Returns:
        List of integer IDs.

    Example:
        ids = encode_tokens(["hello", "world", "xyz"],
                            vocab, max_length=5, pad=True)
        # → [4, 5, 1, 0, 0]  (xyz→<UNK>, then 2 pads)
    """
    unk = vocab.get("<UNK>", 1)
    ids = [vocab.get(t, unk) for t in tokens]
    if add_bos:
        ids = [vocab["<BOS>"]] + ids
    if add_eos:
        ids = ids + [vocab["<EOS>"]]
    if max_length:
        ids = ids[:max_length]
        if pad:
            ids += [vocab.get("<PAD>", 0)] * (max_length - len(ids))
    return ids


def tfidf_vectorize(
    corpus: list[str],
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> tuple["np.ndarray", "TfidfVectorizer"]:
    """
    Transform a list of raw text documents into a TF-IDF matrix.

    TF-IDF = Term Frequency × Inverse Document Frequency.
    Rare terms get higher weight; common terms get lower weight.

    Args:
        corpus:       List of raw text strings (NOT pre-tokenized).
        max_features: Keep only top-N features by term frequency.
        ngram_range:  (min_n, max_n) — (1,1)=unigrams, (1,2)=uni+bigrams.
        min_df:       Ignore terms in fewer than min_df documents.
        max_df:       Ignore terms in more than max_df fraction of documents.

    Returns:
        (X_matrix, fitted_vectorizer)
        X_matrix:   shape (n_documents, n_features)
        vectorizer: Fitted sklearn object. Use vectorizer.transform() on new data.

    Example:
        X, vec = tfidf_vectorize(train_texts)
        X_test = vec.transform(test_texts)
        # Then use X in sklearn models
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
        analyzer="word",
    )
    X = vec.fit_transform(corpus)
    logger.info(f"TF-IDF matrix: {X.shape[0]} docs × {X.shape[1]} features")
    return X, vec

#  PIPELINE — Compose all steps into one callable
class TextPreprocessingPipeline:
    """
    Composable text preprocessing pipeline.

    Wraps all individual steps into a single object you can configure
    once and apply to any number of documents.

    Design pattern: each step is optional and controlled by __init__ flags.
    This makes the pipeline reusable and self-documenting.

    Example ( just clean):
        pipeline = TextPreprocessingPipeline(
            lowercase=True,
            remove_html=True,
            remove_urls=True,
        )
        clean = pipeline.transform("Check <b>this</b> https://example.com out!")

    Example ( clean + tokenize + filter):
        pipeline = TextPreprocessingPipeline(
            lowercase=True,
            expand_contractions=True,
            tokenize=True,
            remove_stopwords=True,
            min_token_length=3,
        )
        tokens = pipeline.transform("I can't believe how great this is!")

    Example (advanced — full pipeline with lemmatization):
        pipeline = TextPreprocessingPipeline(
            lowercase=True,
            expand_contractions=True,
            tokenize=True,
            remove_stopwords=True,
            lemmatize=True,
            join_tokens=True,  # returns string instead of list
        )
        text = pipeline.transform("The children are running very quickly")
        # → "child run quick"
    """

    def __init__(
        self,
        # Cleaning flags
        remove_html:          bool = True,
        remove_urls:          bool = True,
        remove_emails:        bool = False,
        remove_special_chars: bool = False,
        keep_numbers:         bool = True,
        # Normalization
        lowercase:            bool = True,
        unicode_form:         str  = "NFC",
        expand_contractions:  bool = False,
        # Tokenization
        tokenize:             bool = False,
        tokenizer_type:       str  = "simple",   # "simple" | "nltk" | "nltk_tweet"
        # Filtering
        remove_stopwords:     bool = True,
        remove_punctuation:   bool = True,
        min_token_length:     int  = 2,
        max_token_length:     int  = 50,
        custom_stopwords:     Optional[list[str]] = None,
        language:             str  = "english",
        # Morphological normalization
        stem:                 bool = False,
        lemmatize:            bool = False,
        stemmer_type:         str  = "porter",
        # Output format
        join_tokens:          bool = False,   # True → return str, False → list
        join_sep:             str  = " ",
    ):
        # Store all config
        self.config = locals()
        self.config.pop("self")

    def transform(self, text: str) -> Union[str, list[str]]:
        """
        Apply the full pipeline to a single string.

        Returns str if join_tokens=True, else list[str].
        """
        cfg = self.config

        # --- Step 1: Clean ---
        if cfg["remove_html"]:
            text = remove_html_tags(text)
        if cfg["remove_urls"]:
            text = remove_urls(text)
        if cfg["remove_emails"]:
            text = remove_emails(text)
        if cfg["expand_contractions"]:
            text = expand_contractions(text)
        if cfg["remove_special_chars"]:
            text = remove_special_characters(text, keep_numbers=cfg["keep_numbers"])

        # --- Step 2: Normalize ---
        text = normalize_unicode(text, cfg["unicode_form"])
        if cfg["lowercase"]:
            text = to_lowercase(text)
        text = normalize_whitespace(text)

        # --- Step 3: Tokenize ---
        if not cfg["tokenize"] and not cfg["stem"] and not cfg["lemmatize"]:
            return text

        tt = cfg["tokenizer_type"]
        if tt == "simple":
            tokens = simple_tokenize(text)
        elif tt == "nltk":
            tokens = nltk_tokenize(text, tokenizer="word")
        elif tt == "nltk_tweet":
            tokens = nltk_tokenize(text, tokenizer="tweet")
        else:
            tokens = simple_tokenize(text)

        # --- Step 4: Filter ---
        tokens = filter_tokens(
            tokens,
            remove_stopwords=cfg["remove_stopwords"],
            remove_punctuation=cfg["remove_punctuation"],
            min_length=cfg["min_token_length"],
            max_length=cfg["max_token_length"],
            language=cfg["language"],
            custom_stopwords=cfg["custom_stopwords"],
        )

        # --- Step 5: Morphological normalization ---
        if cfg["stem"]:
            tokens = stem_tokens(tokens, stemmer=cfg["stemmer_type"])
        elif cfg["lemmatize"]:
            tokens = lemmatize_tokens(tokens)

        # --- Step 6: Output format ---
        if cfg["join_tokens"]:
            return cfg["join_sep"].join(tokens)
        return tokens

    def transform_batch(self, texts: list[str], show_progress: bool = True) -> list:
        """
        Apply pipeline to a list of documents.

        Args:
            texts:         List of raw strings.
            show_progress: Log progress every 1000 documents.

        Returns:
            List of cleaned strings or token lists.
        """
        results = []
        for i, text in enumerate(texts):
            results.append(self.transform(text))
            if show_progress and i > 0 and i % 1000 == 0:
                logger.info(f"Processed {i}/{len(texts)} documents…")
        logger.info(f"Pipeline complete: {len(texts)} documents processed")
        return results

# 📊 DIAGNOSTICS — Inspect your corpus before modeling
def corpus_stats(texts: list[str]) -> dict:
    """
    Compute basic statistics about a text corpus.

    Useful before modeling to understand:
      - Are texts very long or very short?
      - Are there duplicate entries?
      - What's the vocabulary richness?

    Args:
        texts: List of raw or cleaned strings.

    Returns:
        Dict with keys: n_docs, avg_words, median_words, max_words,
        min_words, unique_vocab, duplication_rate.

    Example:
        stats = corpus_stats(train_texts)
        print(stats)
    """
    import statistics
    lengths = [len(t.split()) for t in texts]
    all_words = [w for t in texts for w in t.lower().split()]
    duplicates = len(texts) - len(set(texts))

    return {
        "n_docs":            len(texts),
        "avg_words":         round(statistics.mean(lengths), 1),
        "median_words":      statistics.median(lengths),
        "max_words":         max(lengths),
        "min_words":         min(lengths),
        "total_tokens":      len(all_words),
        "unique_vocab":      len(set(all_words)),
        "type_token_ratio":  round(len(set(all_words)) / max(len(all_words), 1), 4),
        "duplicates":        duplicates,
        "duplication_rate":  round(duplicates / len(texts), 4),
    }


def top_n_words(
    texts: list[str],
    n: int = 20,
    stopwords: bool = True,
) -> list[tuple[str, int]]:
    """
    Return the n most frequent words in the corpus.

    Args:
        texts:     List of documents.
        n:         Top N words.
        stopwords: Exclude English stopwords.

    Returns:
        List of (word, count) tuples.

    Example:
        words = top_n_words(corpus, n=10)
        for word, count in words:
            print(f"  {word}: {count}")
    """
    stops = get_stopwords() if stopwords else set()
    counter: Counter = Counter()
    for text in texts:
        words = [w.strip(string.punctuation).lower() for w in text.split()]
        counter.update(w for w in words if w and w not in stops)
    return counter.most_common(n)

# 🚀 QUICK-START DEMO
if __name__ == "__main__":
    print("=" * 60)
    print("  text_preprocessing.py — Demo")
    print("=" * 60)

    raw = """
    <p>Check out our <b>NEW</b> product at https://shop.example.com!!
    You can't miss it. Contact us at support@company.com or call +1-800-555-1234.
    It's truly AMAZING &amp; life-changing   for EVERYONE!!! 😍🔥</p>
    """

    print(f"\n Raw input:\n{raw.strip()}\n")

    # single-step functions
    print(" Step-by-step cleaning:")
    t = remove_html_tags(raw)
    print(f"  After HTML removal:    {t[:80].strip()}…")
    t = remove_urls(t)
    print(f"  After URL removal:     {t[:80].strip()}…")
    t = remove_emails(t)
    t = remove_phone_numbers(t)
    t = expand_contractions(t)
    t = to_lowercase(t)
    t = remove_special_characters(t)
    t = normalize_whitespace(t)
    print(f"  Final clean text:      {t}")

    # full pipeline
    print("\n Full pipeline:")
    pipeline = TextPreprocessingPipeline(
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        lowercase=True,
        expand_contractions=True,
        remove_special_chars=True,
        tokenize=True,
        remove_stopwords=True,
        min_token_length=3,
        join_tokens=True,
    )
    result = pipeline.transform(raw)
    print(f"  Result: {result}")

    # Corpus stats
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown fox is a fast animal.",
        "Dogs are lazy but lovable pets.",
        "The quick brown fox jumps over the lazy dog.",  # duplicate
    ]
    print("\n📊 Corpus stats:")
    stats = corpus_stats(corpus)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n🔑 Top words:")
    for word, count in top_n_words(corpus, n=5, stopwords=True):
        print(f"  {word}: {count}")

    print("\n✅ Text preprocessing demo complete.")
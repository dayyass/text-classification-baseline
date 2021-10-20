from abc import ABC, abstractmethod
from typing import Dict

import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer


class AbstractLemmatizer(ABC):
    """Abstract base class for lemmatizers."""

    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def __call__(self, token: str) -> str:
        """Token lemmatization.

        Args:
            token (str): Token for lemmatization.
        """
        ...


class LemmatizerPymorphy2(AbstractLemmatizer):
    """Pymorphy2 lemmatizer."""

    def __init__(self) -> None:
        """
        Init pymorphy2 lemmatizer.
        Use cache for better perfomance.
        """

        self.cache: Dict[str, str] = {}
        self.lemmatizer = pymorphy2.MorphAnalyzer()

    def __call__(self, token: str) -> str:

        if token in self.cache:
            lemma = self.cache[token]
        else:
            lemma = self.lemmatizer.parse(token)[0].normal_form
            self.cache[token] = lemma

        return lemma


class Preprocessor:
    """Sentence preprocessor for TfidfVectorizer."""

    def __init__(self, lemmatizer: AbstractLemmatizer) -> None:
        """Init sentence preprocessor with lemmatizer.

        Args:
            lemmatizer (AbstractLemmatizer): Token lemmatizer.
        """

        self.tokenizer = TfidfVectorizer().build_tokenizer()  # hardcoded
        self.lemmatizer = lemmatizer

    def __call__(self, sentence: str) -> str:
        """Preprocess sentence.

        Args:
            sentence (str): Sentence for preprocessing.

        Returns:
            str: Preprocessed sentence.
        """

        return " ".join([self.lemmatizer(token) for token in self.tokenizer(sentence)])

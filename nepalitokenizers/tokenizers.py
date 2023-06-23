import pathlib
import tokenizers

MODELS_DIR = pathlib.Path(__file__).parent.absolute() / "models"


class Delegate:
    """
    A generic wrapper class that delegates attributes and method calls
    to the specified self.delegate instance.
    """

    @property
    def _items(self):
        return dir(self.delegate)

    def __getattr__(self, name):
        if name in self._items:
            return getattr(self.delegate, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "delegate" or name not in self._items:
            super().__setattr__(name, value)
        else:
            setattr(self.delegate, name, value)

    def __dir__(self):
        return dir(type(self)) + list(self.__dict__.keys()) + self._items


class Tokenizer(Delegate):
    """
    An interface to HuggingFace's Tokenizer pipeline.

    This class wraps the Tokenizer instance from the HuggingFace tokenizers library
    and sets the default normalization, pre and post-processing, and decoder steps.
    """

    def __init__(self):
        tokenizer = tokenizers.Tokenizer.from_file(str(self.model_path))

        # need to initialize the delegate before setting the default values
        self.delegate = tokenizer

        self.default_normalizer = tokenizer.normalizer
        self.default_pre_tokenizer = tokenizer.pre_tokenizer
        self.default_post_processor = tokenizer.post_processor
        self.default_decoder = tokenizer.decoder


class WordPiece(Tokenizer):
    """
    An interface to HuggingFace's Tokenizer pipeline
    using a trained WordPiece model for the Nepali language.

    This class configures a Tokenizer instance with a WordPiece model
    specifically designed for the Nepali language.
    """

    model_path = MODELS_DIR / "WordPiece.json"


class SentencePiece(Tokenizer):
    """
    An interface to HuggingFace's Tokenizer pipeline
    using a trained SentencePiece model for the Nepali language.

    This class configures a Tokenizer instance with a SentencePiece model
    specifically designed for the Nepali language.
    """

    model_path = MODELS_DIR / "SentencePiece.json"
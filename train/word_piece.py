from pathlib import Path
from tokenizers import Tokenizer, decoders
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

data_files = [str(x) for x in Path("<path_to_dir>").glob("*.txt")]
output_path = "WordPiece.json"

VOCAB_SIZE = 30_522  # Standard BERT vocab size

UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
PAD = "[PAD]"
SPECIAL_TOKENS = [UNK, CLS, SEP, PAD, "[MASK]"]


def train_nepali_word_piece_tokenizer():
    tokenizer = Tokenizer(WordPiece(unk_token=UNK))
    tokenizer.normalizer = BertNormalizer(strip_accents=False)
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS
    )

    tokenizer.train(
        files=data_files,
        trainer=trainer
    )

    tokenizer.post_processor = TemplateProcessing(
        single=f'{CLS} $A {SEP}',
        pair=f'{CLS} $A {SEP} $B:1 {SEP}:1',
        special_tokens=[
            (CLS, tokenizer.token_to_id(CLS)),
            (SEP, tokenizer.token_to_id(SEP)),
        ]
    )

    tokenizer.decoder = decoders.WordPiece(prefix="##")

    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(PAD),
        pad_token=PAD
    )

    tokenizer.save(output_path)


if __name__ == "__main__":
    train_nepali_word_piece_tokenizer()

from pathlib import Path
from tokenizers import Tokenizer, Regex, decoders
from tokenizers.models import Unigram
from tokenizers.normalizers import Sequence, Replace
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import UnigramTrainer
from tokenizers.processors import TemplateProcessing

data_files = [str(x) for x in Path("<path_to_dir>").glob("*.txt")]
output_path = "SentencePiece.json"

VOCAB_SIZE = 32_000

CLS = "<cls>"
SEP = "<sep>"
UNK = "<unk>"
PAD = "<pad>"
SPECIAL_TOKENS = [CLS, SEP, UNK, PAD, "<mask>", "<s>", "</s>"]


def train_nepali_sentence_piece_tokenizer():
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = Sequence(
        [
            Replace("“", "\""),
            Replace("”", "\""),
            Replace("‘", "'"),
            Replace("’", "'"),
            Replace(Regex(" {2,}"), " ")
        ]
    )
    tokenizer.pre_tokenizer = Metaspace()

    trainer = UnigramTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        unk_token=UNK
    )

    tokenizer.train(
        files=data_files,
        trainer=trainer
    )

    tokenizer.post_processor = TemplateProcessing(
        single=f'$A:0 {SEP}:0 {CLS}:2',
        pair=f'$A:0 {SEP}:0 $B:1 {SEP}:1 {CLS}:2',
        special_tokens=[
            (SEP, tokenizer.token_to_id(SEP)),
            (CLS, tokenizer.token_to_id(CLS))
        ],
    )

    tokenizer.decoder = decoders.Metaspace()

    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(PAD),
        pad_token=PAD
    )

    tokenizer.save(output_path)

if __name__ == "__main__":
    train_nepali_sentence_piece_tokenizer()
import unittest
from nepalitokenizers import WordPiece, SentencePiece
from tokenizers.processors import TemplateProcessing


texts = [
    "हाम्रा सबै क्रियाकलापहरु भोलिवादी छन् । मेरो पानीजहाज वाम माछाले भरिपूर्ण छ । इन्जिनियरहरुले गएको हप्ता राजधानीमा त्यस्तै बहस गरे ।",
    "कोभिड महामारीको पिडाबाट मुक्त नहुँदै मानव समाजलाई यतिबेला युद्धको विध्वंसकारी क्षतिको चिन्ताले चिन्तित बनाएको छ ।",
]


class TestTokenizers(unittest.TestCase):
    def setUp(self):
        self.tokenizer_wp = WordPiece()
        self.tokenizer_sp = SentencePiece()

    def _test_tokenizer(self, tokenizer):
        # encoding and decoding
        for text in texts:
            tokens = tokenizer.encode(text)            
            decoded_text = tokenizer.decode(tokens.ids)
            self.assertEqual(text, decoded_text)
        
        # batch encoding and decoding
        tokens_list = tokenizer.encode_batch(texts)
        decoded_texts = tokenizer.decode_batch([tokens.ids for tokens in tokens_list])
        self.assertEqual(texts, decoded_texts)

        # customization
        tokenizer.post_processor = TemplateProcessing()

    def test_wordpiece_tokenization(self):
        self._test_tokenizer(self.tokenizer_wp)
        
    def test_sentencepiece_tokenization(self):
        self._test_tokenizer(self.tokenizer_sp)
    

if __name__ == "__main__":
    unittest.main()
# Nepali Tokenizers

[![LICENSE](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE) [![Build and Release](https://github.com/basnetsoyuj/nepali-tokenizers/actions/workflows/build.yml/badge.svg)](https://github.com/basnetsoyuj/nepali-tokenizers/actions/workflows/build.yml)

This package provides access to pre-trained __WordPiece__ and __SentencePiece__ (Unigram) tokenizers for Nepali language, trained using HuggingFace's `tokenizers` library. It is a simple and short Python package tailored specifically for Nepali language with a default set of configurations for the normalizer, pre-tokenizer, post-processor, and decoder. 

It delegates further customization by providing an interface to HuggingFace's `Tokenizer` pipeline, allowing users to adapt the tokenizers according to their requirements.


## Installation

You can install `nepalitokenizers` using pip:

```sh
pip install nepalitokenizers
```


## Usage

After installing the package, you can use the tokenizers in your Python code:

### WordPiece Tokenizer

```python
from nepalitokenizers import WordPiece

text = "हाम्रा सबै क्रियाकलापहरु भोलिवादी छन् । मेरो पानीजहाज वाम माछाले भरिपूर्ण छ । इन्जिनियरहरुले गएको हप्ता राजधानीमा त्यस्तै बहस गरे ।"

tokenizer_wp = WordPiece()

tokens = tokenizer_wp.encode(text)
print(tokens.ids)
print(tokens.tokens)

print(tokenizer_wp.decode(tokens.ids))
```

**Output**

```
[1, 11366, 8625, 14157, 8423, 13344, 9143, 8425, 1496, 9505, 22406, 11693, 12679, 8340, 27445, 1430, 1496, 13890, 9008, 9605, 13591, 14547, 9957, 12507, 8700, 1496, 2]
['[CLS]', 'हाम्रा', 'सबै', 'क्रियाकलाप', '##हरु', 'भोलि', '##वादी', 'छन्', '।', 'मेरो', 'पानीजहाज', 'वाम', 'माछा', '##ले', 'भरिपूर्ण', 'छ', '।', 'इन्जिनियर', '##हरुले', 'गएको', 'हप्ता', 'राजधानीमा', 'त्यस्तै', 'बहस', 'गरे', '।', '[SEP]']
हाम्रा सबै क्रियाकलापहरु भोलिवादी छन् । मेरो पानीजहाज वाम माछाले भरिपूर्ण छ । इन्जिनियरहरुले गएको हप्ता राजधानीमा त्यस्तै बहस गरे ।
```

### SentencePiece (Unigram) Tokenizer

```python
from nepalitokenizers import SentencePiece

text = "कोभिड महामारीको पिडाबाट मुक्त नहुँदै मानव समाजलाई यतिबेला युद्धको विध्वंसकारी क्षतिको चिन्ताले चिन्तित बनाएको छ ।"

tokenizer_sp = SentencePiece()

tokens = tokenizer_sp.encode(text)
print(tokens.ids)
print(tokens.tokens)

print(tokenizer_wp.decode(tokens.ids))
```

**Output**

```
[7, 9, 3241, 483, 12081, 9, 11079, 23, 2567, 11254, 1002, 789, 20, 3334, 2161, 9, 23517, 2711, 1115, 9, 1718, 12, 5941, 781, 19, 8, 1, 0]
['▁', 'को', 'भि', 'ड', '▁महामारी', 'को', '▁पिडा', 'बाट', '▁मुक्त', '▁नहुँदै', '▁मानव', '▁समाज', 'लाई', '▁यतिबेला', '▁युद्ध', 'को', '▁विध्वंस', 'कारी', '▁क्षति', 'को', '▁चिन्ता', 'ले', '▁चिन्तित', '▁बनाएको', '▁छ', '▁।', '<sep>', '<cls>']
कोभिड महामारीको पिडाबाट मुक्त नहुँदै मानव समाजलाई यतिबेला युद्धको विध्वंसकारी क्षतिको चिन्ताले चिन्तित बनाएको छ ।
```


## Configuration & Customization

Each tokenizer class has a default and standard set of configurations for the normalizer, pre-tokenizer, post-processor, and decoder. For more information, look at the training files available in the [`train/`](train/) directory.

The package delegates further customization by providing an interface to directly access to HuggingFace's tokenizer pipeline. Therefore, you can treat `nepalitokenizers`'s tokenizer instances as HuggingFace's `Tokenizer` objects. For example:

```python
from nepalitokenizers import WordPiece

# importing from the HuggingFace tokenizers package
from tokenizers.processors import TemplateProcessing

text = "हाम्रो मातृभूमि नेपाल हो"

tokenizer_sp = WordPiece()

# using default post processor
tokens = tokenizer_sp.encode(text)
print(tokens.tokens)

# change the post processor to not add any special tokens
# treat tokenizer_sp as HuggingFace's Tokenizer object
tokenizer_sp.post_processor = TemplateProcessing()

tokens = tokenizer_sp.encode(text)
print(tokens.tokens)
```

**Output**
```
['[CLS]', 'हाम्रो', 'मातृ', '##भूमि', 'नेपाल', 'हो', '[SEP]']
['हाम्रो', 'मातृ', '##भूमि', 'नेपाल', 'हो']
```

To learn more about further customizations that can be performed, visit [HuggingFace's Tokenizer Documentation](https://huggingface.co/docs/tokenizers/api/tokenizer).


> **Note**: The delegation to HuggingFace's Tokenizer pipeline was done with the following generic wrapper class because `tokenizers.Tokenizer` is not an acceptable base type for inheritance.
> It is a useful trick I use for solving similar issues:
> ```python
> class Delegate:
>    """
>    A generic wrapper class that delegates attributes and method calls
>    to the specified self.delegate instance.
>    """
>
>    @property
>    def _items(self):
>        return dir(self.delegate)
>
>    def __getattr__(self, name):
>        if name in self._items:
>            return getattr(self.delegate, name)
>        raise AttributeError(
>            f"'{self.__class__.__name__}' object has no attribute '{name}'")
>
>    def __setattr__(self, name, value):
>        if name == "delegate" or name not in self._items:
>            super().__setattr__(name, value)
>        else:
>            setattr(self.delegate, name, value)
>
>    def __dir__(self):
>        return dir(type(self)) + list(self.__dict__.keys()) + self._items
> ```


## Training

The python files used to train the tokenizers are available in the [`train/`](train/) directory. You can also use these files to train your own tokenizers on a custom text corpus. 

These tokenizers were trained on two datasets:

#### 1. The Nepali Subset of the [OSCAR](https://oscar-project.github.io/documentation/versions/oscar-2301/) dataset

You can download it using the following code:

```python
import datasets
from tqdm.auto import tqdm
import os

dataset = datasets.load_dataset(
    'oscar', 'unshuffled_deduplicated_ne',
    split='train'
)

os.mkdir('data')

batch = []
counter = 0

for sample in tqdm(dataset):
    sample = sample['text'].replace('\n', ' ')
    batch.append(sample)

    if len(batch) == 10_000:
        with open(f'data/ne_{counter}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(batch))
            batch = []
            counter += 1
```

#### 2. A Large Scale Nepali Text Corpus by Rabindra Lamsal (2020)
To download the dataset, follow the instructions provided in this link: [A Large Scale Nepali Text Corpus](https://dx.doi.org/10.21227/jxrd-d245).


## License

This package is licensed under the Apache 2.0 License, which is consistent with the license used by HuggingFace's `tokenizers` library. Please see the [`LICENSE`](LICENSE) file for more details.
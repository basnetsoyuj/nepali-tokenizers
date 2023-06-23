## Training

The python files used to train the tokenizers are available in the [`train/`](train/) directory. You can use these files to train your own tokenizers on a custom text corpus. 

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



# Student B: Deep Learning Models

## Instructions to Reproduce Results
1. Download the notebook
2. Upload all models in model artifacts (in case running without retraining)
3. Upload all needed datasets from Student A's dataset folders

---

## AI Usage Log

Where and how AI tools were used by Student B for deep learning model development.

### Tools Used
- Gemini
- Claude
- ChatGPT

---

### Tasks Where AI Was Used
*(In-depth details provided in the next section)*
- Brainstorming modeling options and their feasibility given project time and CPU constraints, and clarifying concepts related to model working mechanisms
- Debugging Python code and enhancing code quality by evaluating proposed approaches and suggesting improvements
- Reviewing model architecture implementation and identifying possible issues in defining layers
- Suggesting validation checks
- Grammatical, styling and phrasing improvements for the report only

### Tasks Where AI Was NOT Used
- Fabricating results
- Inventing experiments that were not run
- Generating fake figures or tables
- Replacing manual code execution
- Replacing our understanding of the project
- Writing report sections

---

## Details of AI Usage

### Brainstorming
After searching for relevant previous works highlighted in the related work section of the report, AI was used to clarify architectural concepts and validate the feasibility of the student's ideas — including reimplementing BiLSTM+CNN, using FastText with CNN as a word-level combination, and Hybrid Feature Fusion (validated by the Random Forest baseline model features to compensate for features not detected by the CNN, such as URL path length due to truncation).

---

### Debugging and Code Quality Enhancement

#### Bug (Major): Class Imbalance Across Batches
Using `shuffle=True` in the training DataLoader led to uneven distribution of the minority class across batches.

**Student's approach:**
```python
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=True, num_workers=2)
```

**AI's fix — WeightedRandomSampler to ensure minority class representation:**
```python
class_weights = {0: 1.0/class_counts[0], 1: 1.0/class_counts[1]}
sample_weights = [class_weights[label] for label in train_df['Label'].values]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
```

---

#### Bug (Major): Frozen FastText Embedding Layer
Setting `freeze=True` in the embedding layer prevented the model from learning correct semantic representations, resulting in very low accuracy.

**Fix:** Set embedding `freeze` to `False`.

---

#### Bug (Major): Zero-Initialized OOV Embeddings
Zero-initializing out-of-vocabulary tokens made them indistinguishable from padding tokens, so the model could not learn from them.

**Student's approach:**
```python
embedding_matrix = torch.zeros(vocab_size, embed_dim)
```

**AI's fix — random initialization for OOV tokens:**
```python
std_dev = 0.1
embedding_matrix = torch.randn(vocab_size, embed_dim) * std_dev

for token, idx in url_vocab.get_stoi().items():
    if token in ft.stoi:
        embedding_matrix[idx] = ft.vectors[ft.stoi[token]]
```

---

#### Bug (Moderate): Flipped Early Stopping Logic
The condition logic was inverted, causing the patience counter to increment on improvement and reset on non-improvement — the exact opposite of intended behavior. As a result, early stopping never triggered and all epochs ran to completion, significantly increasing training time. This did not affect model quality since checkpoints saved the best model throughout training; however, it was fixed in version 3 to reduce training time in future use.

**Student's implementation:**
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stopping = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:  # WRONG: triggers on improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopping = True
        else:
            self.best_score = val_loss
            self.counter = 0
        return self.early_stopping
```

**AI's fix:**
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stopping = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss <= self.best_score - self.min_delta:  # loss improved enough
            self.best_score = val_loss
            self.counter = 0
        else:  # no meaningful improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopping = True
        return self.early_stopping
```

---

#### Bug (Minor): Incomplete `<UNK>` Token String in CharVocab
Missing closing parenthesis — did not affect training or inference.

```python
# Buggy
self.idx2char = {0: '<PAD>', 1: '<UNK'}

# Fixed
self.idx2char = {0: '<PAD>', 1: '<UNK>'}
```

---

#### Bug (Minor): Printing Object Reference Instead of Parameter Count
```python
# Buggy
print(f"CNN-BiLSTM parameters: {cnn_lstm.parameters()}")

# Fixed
print(f"CharCNN parameters: {sum(p.numel() for p in char_cnn.parameters()):,}")
```

---

#### Bug (Minor): torch/torchtext Version Mismatch
```bash
# Buggy
!pip install torchtext

# Fixed
!pip install -U torch==2.3.0+cpu torchtext==0.18.0 --index-url https://download.pytorch.org/whl/cpu
```

---

#### Bug (Minor): Passing Full DataFrame Instead of URL Column; `return` Instead of `yield`

**Student's implementation:**
```python
def get_tokens(urls):
    result = []
    for url in urls:
        result.append(url_tokenizer(url))
    return result
```

**AI's improvement:**
```python
def yield_tokens(urls):
    for url in urls:
        yield url_tokenizer(url)

url_vocab = build_vocab_from_iterator(
    yield_tokens(train_df['URL']),
    specials=['<pad>', '<unk>'],
    min_freq=3
)
url_vocab.set_default_index(url_vocab['<unk>'])
```

---

#### Quality Issue: Saving Model Artifacts
The initial approach required manual reloading overhead and limited reproducibility.

**Student's approach:**
```python
torch.save(char_cnn.state_dict(), "char_cnn_model1.pth")
```

**AI's improvement — save full reproducibility bundle:**
```python
torch.save({
    'model_state_dict': char_cnn.state_dict(),
    'char2idx': vocab.char2idx,
    'idx2char': vocab.idx2char,
    'vocab_size': len(vocab.char2idx),
    'max_len': MAX_LEN,
}, 'char_cnn_(1).pth')
```

---

#### Quality Issue: Adding Checkpoints
Starting from the BiLSTM+CNN model, training time increased due to the larger parameter count. Saving only the final model state is suboptimal; the best parameters by validation F1 should be saved instead.

**Student's approach:** Save final model state only (same as above).

**AI's improvement — checkpoint saving by best F1:**
```python
if val_metrics['F1'] > best_f1:
    best_f1 = val_metrics['F1']
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_f1': val_metrics['F1']
    }, f'{model_name}.pth')
    print(f" Checkpoint saved (F1: {best_f1:.4f})")

# Restoring from checkpoint
checkpoint = torch.load('cnn_bilstm_best.pth', map_location='cpu', weights_only=False)
cnn_lstm = CNN_LSTM(vocab_size=len(vocab.char2idx)).to(device)
cnn_lstm.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.Adam(cnn_lstm.parameters(), lr=0.001, weight_decay=1e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

#### Quality Issue: Long Training Time on CPU
**Student's approach:** Reduce number of epochs for BiLSTM+CNN.

**AI's suggestions:**
1. **Accepted** — Set `num_workers=0` instead of `2`: running two workers on CPU caused context switching and resource contention with no throughput gain, since only a single worker can use the CPU at a time. This reduced training time.
2. **Refused** — Stop using `tqdm`: rejected in order to keep track of the training process and detect issues.

---

#### Quality Issue: Avoiding Overfitting
The dataset was relatively smaller than those used in most prior literature, which was a reasonable trade-off for CPU-only feasibility. Best practices were applied: Batch Normalization, Dropout, L2 regularization, and data augmentation.

**AI's role — correct random masking implementation:**
```python
if self.augment and np.random.random() < 0.1:  # augmentation with 10% probability
    mask = np.random.random(len(encoded)) < 0.05  # mask 5% of tokens
    for i in range(len(encoded)):
        if mask[i] and encoded[i] > 1:  # avoid masking <PAD>=0 or <UNK>=1
            encoded[i] = self.vocab.unknown_idx  # replace with <UNK>
```

---

#### Quality Issue: `feature_cols` as Global Instead of Parameter
**Student's approach:**
```python
def __init__(self, df, feat_df, vocab, max_len=200, augment=False):
```

**AI's fix — pass as parameter:**
```python
def __init__(self, df, feat_df, vocab, max_len=200, augment=False, feature_cols=None):
    ...
    cols = feature_cols if feature_cols is not None else feat_df.columns.tolist()
    self.features = feat_df[cols].values.astype(np.float32)
```

---

#### Quality Issue: Formatting Print Output
AI suggested prettifying printed results instead of directly printing raw DataFrames.

---

*Note: All significant bugs and quality issues are listed above. Some code cell reorganization was also applied for readability with no functional changes.*

---

### Validation Checks
```python
# Verify URL order in train_df matches train_feat row order
assert len(train_df) == len(train_features), "Row count mismatch!"
assert (train_features['Label'].values == train_df['Label'].values).all(), "Label mismatch!"
print("Alignment confirmed")
```
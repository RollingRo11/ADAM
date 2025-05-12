import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


class URLDataset(Dataset):
    def __init__(self, urls, labels, vectorizer=None, max_length=100):
        self.urls = urls
        self.labels = labels
        self.max_length = max_length

        self.class_to_idx = {"benign": 0, "phishing": 1, "defacement": 2}

        self.vectorizer = (
            vectorizer
            if vectorizer
            else CountVectorizer(analyzer="char", ngram_range=(1, 2))
        )
        if vectorizer is None:
            self.vectorizer.fit(self.urls)

        self.vocab = self.vectorizer.vocabulary_
        self.vocab_size = len(self.vocab) + 1

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        label = self.labels[idx]

        if isinstance(label, str):
            label = self.class_to_idx.get(label.lower(), 0)

        tokens = self._tokenize_url(url)

        return {
            "url": url,
            "tokens": tokens,
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _tokenize_url(self, url):
        features = self.vectorizer.transform([url]).indices

        if len(features) > self.max_length:
            features = features[: self.max_length]
        else:
            features = np.pad(
                features, (0, self.max_length - len(features)), "constant"
            )

        return torch.tensor(features, dtype=torch.long)


def collate_fn(batch):
    urls = [item["url"] for item in batch]
    tokens = torch.stack([item["tokens"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    return {"urls": urls, "tokens": tokens, "labels": labels}


def load_and_split_data(csv_path, test_size=0.2, random_state=42, batch_size=32):
    df = pd.read_csv(csv_path)

    if "url" in df.columns and "type" in df.columns:
        urls = df["url"].values
        labels = df["type"].values
    else:
        # Try to infer columns
        if len(df.columns) >= 2:
            urls = df.iloc[:, 0].values
            labels = df.iloc[:, 1].values
        else:
            raise ValueError("CSV must have at least two columns: url and type")

    # Split data
    train_urls, test_urls, train_labels, test_labels = train_test_split(
        urls, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Create datasets
    train_dataset = URLDataset(train_urls, train_labels)
    test_dataset = URLDataset(
        test_urls, test_labels, vectorizer=train_dataset.vectorizer
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, test_loader, train_dataset.vocab_size

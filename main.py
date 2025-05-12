import torch
import torch.nn as nn
from model import URLClassifier
from data import load_and_split_data
from optimizer import Adam


def main():
    print("initializing model...")
    # Load and split data
    train_loader, test_loader, vocab_size = load_and_split_data(
        "malicious_phish.csv", batch_size=64
    )

    # Initialize model
    model = URLClassifier(
        input_dim=vocab_size,
        embedding_dim=128,
        hidden_dim=64,
        output_dim=3,  # benign/phishing/defacement
        dropout=0.5,
    )

    # loss & optim
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = Adam(model.parameters(), lr=0.001)

    # train loop
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            tokens = batch["tokens"]
            labels = batch["labels"]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(tokens)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total
        print(
            f"Epoch {epoch}: loss: {avg_train_loss:.4f} | accuracy: {train_accuracy:.2f}%"
        )

    # testing
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch["tokens"]
            labels = batch["labels"]

            outputs = model(tokens)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # stats
    correct = sum(pred == label for pred, label in zip(all_predictions, all_labels))
    total = len(all_labels)
    test_accuracy = 100.0 * correct / total

    print(f"Test accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()

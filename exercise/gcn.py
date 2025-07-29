import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess the Cora dataset
print("Loading Cora dataset...")
dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0].to(device)

print(f"Dataset: {dataset}")
print(f"Number of nodes: {data.x.shape[0]}")
print(f"Number of edges: {data.edge_index.shape[1]}")
print(f"Number of features: {data.x.shape[1]}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Training nodes: {data.train_mask.sum()}")
print(f"Validation nodes: {data.val_mask.sum()}")
print(f"Test nodes: {data.test_mask.sum()}")

class GCN(nn.Module):
    """
    Graph Convolutional Network with two GCN layers
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        # First GCN layer for feature extraction
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # Second GCN layer for classification
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # First layer with ReLU activation and dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer for final classification
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Model hyperparameters
input_dim = dataset.num_node_features
hidden_dim = 16
output_dim = dataset.num_classes
learning_rate = 0.01
dropout = 0.5
epochs = 200

# Initialize model, loss function, and optimizer
model = GCN(input_dim, hidden_dim, output_dim, dropout).to(device)
criterion = nn.NLLLoss()  # Negative log likelihood loss (works with log_softmax)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

print(f"\nModel architecture:")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Training loop with logging
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).float().sum()
        accuracy = correct / mask.sum()
        loss = criterion(out[mask], data.y[mask])
    return loss.item(), accuracy.item()

# Training history
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print("\nStarting training...")
start_time = time.time()

for epoch in range(epochs):
    # Training
    train_loss = train()
    
    # Evaluation every 20 epochs
    if epoch % 20 == 0:
        train_loss_eval, train_acc = evaluate(data.train_mask)
        val_loss, val_acc = evaluate(data.val_mask)
        
        train_losses.append(train_loss_eval)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Final evaluation on test set
test_loss, test_acc = evaluate(data.test_mask)
print(f"\nTest Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Detailed evaluation metrics
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # Get predictions and true labels for test set
    test_pred = pred[data.test_mask].cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()
    
    # Calculate precision, recall, and F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        test_true, test_pred, average='weighted'
    )
    
    print(f"\nDetailed Test Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Per-class metrics
    print(f"\nPer-class Classification Report:")
    class_names = ['Case Based', 'Genetic Algorithms', 'Neural Networks', 
                   'Probabilistic Methods', 'Reinforcement Learning', 
                   'Rule Learning', 'Theory']
    print(classification_report(test_true, test_pred, target_names=class_names))

# Plot training curves
plt.figure(figsize=(15, 5))

# Loss curve
plt.subplot(1, 3, 1)
epochs_logged = list(range(0, epochs, 20))
plt.plot(epochs_logged, train_losses, 'b-', label='Training Loss')
plt.plot(epochs_logged, val_losses, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Accuracy curve
plt.subplot(1, 3, 2)
plt.plot(epochs_logged, train_accuracies, 'b-', label='Training Accuracy')
plt.plot(epochs_logged, val_accuracies, 'r-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

# Final performance comparison
plt.subplot(1, 3, 3)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [test_acc, precision, recall, f1]
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('Score')
plt.title('Test Set Performance Metrics')
plt.ylim(0, 1)
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.show()
plt.savefig('performance_metrics.png', dpi=300)
plt.close()

# Save training logs
training_log = {
    'epochs': epochs_logged,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'test_accuracy': test_acc,
    'test_precision': precision,
    'test_recall': recall,
    'test_f1': f1
}

print(f"\nTraining completed successfully!")
print(f"Model achieved {test_acc:.4f} test accuracy with F1-score of {f1:.4f}")

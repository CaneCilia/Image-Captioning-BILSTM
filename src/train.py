import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from bilstm_model import BiLSTMCaptioner
import os

def train():
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 5
    feature_dim = 512 # Depends on YOLO extraction
    embed_size = 256
    hidden_size = 512
    vocab_size = 1000 # Mock vocab size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = BiLSTMCaptioner(feature_dim, embed_size, hidden_size, vocab_size).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Assuming 0 is <PAD> token
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize Dataloader
    dataloader = get_dataloader('data/images', 'data/annotations.json', batch_size)
    print("DataLoader initialized with Mock Data. Ready to train!")

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for idx, (features, captions) in enumerate(dataloader):
            features, captions = features.to(device), captions.to(device)
            
            # Forward pass
            outputs = model(features, captions)
            
            # outputs: (batch_size, max_length + 1, vocab_size)
            # targets: captions (batch_size, max_length)
            # We align outputs to match targets. Simple alignment for mock:
            outputs = outputs[:, :-1, :] # discard last prediction to match caption length
            
            # Flatten outputs and targets for CrossEntropyLoss
            outputs = outputs.reshape(-1, vocab_size)
            targets = captions.reshape(-1)
            
            # Calculate Loss
            loss = criterion(outputs, targets)
            
            # Backward pass & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
    # Save the model
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/bilstm_captioner.pth')
    print("Training complete! Model saved to ../models/bilstm_captioner.pth")

if __name__ == '__main__':
    train()

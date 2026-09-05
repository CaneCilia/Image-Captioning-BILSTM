import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from bilstm_model import BiLSTMCaptioner

def train():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    feature_dim = 512 # Depends on YOLO extraction
    embed_size = 256
    hidden_size = 512
    vocab_size = 10000 # To be updated dynamically
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f\"Using device: {device}\")

    # Initialize model
    model = BiLSTMCaptioner(feature_dim, embed_size, hidden_size, vocab_size).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Assuming 0 is <PAD> token
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # TODO: Initialize Dataloader
    # dataloader = get_dataloader('data/images', 'data/annotations.json', batch_size)
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        # for idx, (images, captions) in enumerate(dataloader):
            # 1. Extract YOLO features
            # 2. Forward pass
            # 3. Calculate Loss
            # 4. Backward pass & optimize
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: 0.0000') # Placeholder
        
    # Save the model
    # torch.save(model.state_dict(), '../models/bilstm_captioner.pth')

if __name__ == '__main__':
    train()

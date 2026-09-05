import torch
import torch.nn as nn

class BiLSTMCaptioner(nn.Module):
    def __init__(self, feature_dim, embed_size, hidden_size, vocab_size, num_layers=1):
        super(BiLSTMCaptioner, self).__init__()
        
        # Linear layer to project YOLO features to embedding dimension
        self.feature_projection = nn.Linear(feature_dim, embed_size)
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        
        # Output linear layer to predict the next word
        # Note: Since it's bidirectional, the hidden state size is hidden_size * 2
        self.linear = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, features, captions):
        \"\"\"
        features: YOLO extracted features (batch_size, feature_dim)
        captions: Ground truth captions (batch_size, max_length)
        \"\"\"
        # Project features
        features = self.feature_projection(features)
        
        # Embed captions
        embeddings = self.embedding(captions)
        
        # TODO: Combine features and caption embeddings
        # usually features are prepended as the first step to the LSTM
        
        # Pass through BiLSTM
        lstm_out, _ = self.lstm(embeddings)
        
        # Predict words
        outputs = self.linear(lstm_out)
        
        return outputs

    def generate_caption(self, features, max_length=20):
        \"\"\"
        Generate caption during inference.
        \"\"\"
        # TODO: Implement greedy search or beam search for inference
        pass

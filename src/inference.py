import torch
from bilstm_model import BiLSTMCaptioner
from yolo_feature_ext import YOLOFeatureExtractor
from PIL import Image
import matplotlib.pyplot as plt

def generate_caption(image_path, model_path, vocab):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load YOLO extractor
    yolo_extractor = YOLOFeatureExtractor()
    features = yolo_extractor.extract_features(image_path)
    
    # Convert features to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    # Load BiLSTM model
    # Note: Ensure parameters match those used during training
    model = BiLSTMCaptioner(feature_dim=512, embed_size=256, hidden_size=512, vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        # TODO: Implement sequence generation loop
        caption_indices = model.generate_caption(features_tensor)
        
    # Convert indices back to words
    # caption = [vocab.idx2word[idx] for idx in caption_indices]
    caption = \"Placeholder generated caption\"
    
    # Display image and caption
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')
    plt.show()
    
    return caption

if __name__ == '__main__':
    # test inference
    print(\"Inference script ready.\")

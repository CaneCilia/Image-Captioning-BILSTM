import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CNN-based Feature Extraction & NLP for Object Relation\n",
    "This notebook covers the development of a Convolutional Neural Network (CNN) to extract features from images, and a subsequent Natural Language Processing (NLP) pipeline to process these features to find object relations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torchvision.models as models\n",
    "import torchvision.transforms as transforms\n",
    "from PIL import Image\n",
    "import nltk\n",
    "import numpy as np\n",
    "\n",
    "# Ensure NLTK data is downloaded\n",
    "nltk.download('punkt')\n",
    "nltk.download('averaged_perceptron_tagger')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Feature Extraction using CNN (ResNet50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load a pre-trained ResNet50 model, excluding the final classification layer\n",
    "class CNNFeatureExtractor(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(CNNFeatureExtractor, self).__init__()\n",
    "        resnet = models.resnet50(pretrained=True)\n",
    "        # Remove the final fully connected layer to output feature maps\n",
    "        self.features = nn.Sequential(*list(resnet.children())[:-1])\n",
    "        \n",
    "    def forward(self, x):\n",
    "        # x shape should be (batch_size, 3, 224, 224)\n",
    "        out = self.features(x)\n",
    "        return out.view(out.size(0), -1) # Flatten\n",
    "\n",
    "cnn_extractor = CNNFeatureExtractor()\n",
    "cnn_extractor.eval()\n",
    "print("CNN Feature Extractor loaded.")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Image Preprocessing\n",
    "transform = transforms.Compose([\n",
    "    transforms.Resize((224, 224)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n",
    "])\n",
    "\n",
    "def extract_features(image_path):\n",
    "    img = Image.open(image_path).convert('RGB')\n",
    "    img_t = transform(img)\n",
    "    batch_t = torch.unsqueeze(img_t, 0)\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        features = cnn_extractor(batch_t)\n",
    "    return features\n",
    "\n",
    "# Example usage (replace with actual image path)\n",
    "# features = extract_features('path_to_image.jpg')\n",
    "# print(features.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. NLP Pipeline for Object Relation\n",
    "In a full image captioning or visual question answering system, the CNN features are fed into an RNN/LSTM or Transformer along with text data to learn relations. Here we simulate the NLP portion that processes potential detected objects to find relations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.tokenize import word_tokenize\n",
    "from nltk import pos_tag\n",
    "\n",
    "def extract_object_relations(sentence):\n",
    "    tokens = word_tokenize(sentence)\n",
    "    tagged = pos_tag(tokens)\n",
    "    \n",
    "    objects = []\n",
    "    relations = []\n",
    "    \n",
    "    # Simple rule-based extraction\n",
    "    # Nouns as objects, Verbs/Prepositions as relations\n",
    "    for word, tag in tagged:\n",
    "        if tag.startswith('NN'): # Noun\n",
    "            objects.append(word)\n",
    "        elif tag.startswith('VB') or tag == 'IN': # Verb or Preposition\n",
    "            relations.append(word)\n",
    "            \n",
    "    return objects, relations\n",
    "\n",
    "sample_caption = "A dog is playing with a red ball in the park."\n",
    "objs, rels = extract_object_relations(sample_caption)\n",
    "print(f"Objects: {objs}")\n",
    "print(f"Relations: {rels}")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Connecting CNN Features to NLP\n",
    "Typically, the image features (extracted above) would be projected into a shared embedding space with text, or used as the initial state of an LSTM that generates the sequence of object relations and words."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class VisualSemanticAlignment(nn.Module):\n",
    "    def __init__(self, cnn_feat_dim, embed_size):\n",
    "        super(VisualSemanticAlignment, self).__init__()\n",
    "        self.linear = nn.Linear(cnn_feat_dim, embed_size)\n",
    "        \n",
    "    def forward(self, features):\n",
    "        return self.linear(features)\n",
    "\n",
    "# Map 2048-dim ResNet output to 512-dim embedding space\n",
    "alignment_layer = VisualSemanticAlignment(2048, 512)\n",
    "\n",
    "# Example of passing the extracted features\n",
    "# aligned_features = alignment_layer(features)\n",
    "# print(aligned_features.shape)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(r'D:\GitDesk\Image-Captioning-BILSTM\cnn_nlp_model_dev.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

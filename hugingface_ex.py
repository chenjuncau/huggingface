# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:56:56 2024

@author: chenj
This one is working well. 
"""


import os
os.chdir("C:\\Users\\chenj\\Desktop\\Jun\\GPT")


from transformers import pipeline

# Load a pre-trained model for text generation
generator = pipeline('text-generation', model='gpt2')

# Generate text based on a prompt
prompt = "Once upon a time,"
results = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
for result in results:
    print(result['generated_text'])
    
    

# Generate text based on a prompt
prompt = "a boy named Max, he has a dog, and created a short story"
results = generator(prompt, max_length=500, num_return_sequences=1)

# Print the generated text
for result in results:
    print(result['generated_text'])




# 2 example

# Load a pre-trained BERT model for sentiment analysis
classifier = pipeline('sentiment-analysis')

# Example sentences to classify
sentences = [
    "I love the new design of your website!",
    "I'm not satisfied with the service I received.",
    "The movie was fantastic!"
]

# Perform sentiment analysis
results = classifier(sentences)

# Print the results
for sentence, result in zip(sentences, results):
    print(f"Sentence: {sentence}\nSentiment: {result['label']}, Score: {result['score']}\n")

    


# 3 example

#pip install transformers
##pip install torch
#pip install torchvision
#pip install pillow



from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch

# Load a pre-trained ViT model and feature extractor
model_name = "google/vit-base-patch16-224"
#feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Load an example image from the internet
url = "https://huggingface.co/front/thumbnails/transformers.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB") 

# 2 example 
image=Image.open('chicken.jpg').convert("RGB") 
image=Image.open('cattle.jpg').convert("RGB") 
#image = Image.open("path_to_image").convert("RGB") 
# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform image classification
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])


# Load the labels for ImageNet classes
#labels = feature_extractor.from_pretrained(model_name)

# Print the predicted label
#print(f"Predicted class: {labels[predicted_class_idx]}")

# https://www.philschmid.de/image-classification-huggingface-transformers-keras



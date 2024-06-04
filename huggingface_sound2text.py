# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:56:11 2024

@author: chenj
"""
import os
os.chdir("C:\\Users\\chenj\\Desktop\\Jun\\GPT")


import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Parameters
sample_rate = 16000  # Sample rate
duration = 5  # Duration in seconds

def record_audio():
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete")
    return recording

# Example usage
audio_data = record_audio()
write("output.wav", sample_rate, audio_data)  # Save as WAV file (optional)





from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# Load pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def speech_to_text(audio_data, sample_rate):
    # Preprocess the audio data
    input_values = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_values
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

# Example usage
audio_data = np.squeeze(audio_data)  # Remove single-dimensional entries
transcription = speech_to_text(audio_data, sample_rate)
print("Transcription:", transcription)



# very nice.

#https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/DETR_panoptic_segmentation_minimal_example_(with_DetrFeatureExtractor).ipynb







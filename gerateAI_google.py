# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:46:49 2024
This is generated AI from google. 
@author: chenj
"""

# pip install -U -q google-generativeai 
# pip install -U -q google.colab
# export GOOGLE_API_KEY="AIzaSyDJNmHyogzqSMlta6V4arYUDKrWqR0cnhM"
# 

import os

os.environ['API_KEY'] = 'AIzaSyDJNmHyogzqSMlta6V4arYUDKrWqR0cnhM'

import google.generativeai as genai
#from google.colab import userdata

#GOOGLE_API_KEY=userdata.get('AIzaSyDJNmHyogzqSMlta6V4arYUDKrWqR0cnhM')
#genai.configure(api_key=GOOGLE_API_KEY)
# AIzaSyDJNmHyogzqSMlta6V4arYUDKrWqR0cnhM

import os

GOOGLE_API_KEY = os.getenv('API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)



# 1st example.
model = genai.GenerativeModel('gemini-1.5-flash')   # gemini-1.5-flash
response = model.generate_content("Give me python code to sort a list")
print(response.text)
     


# 2nd example. 

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Write a cute story about cats.", stream=True)
for chunk in response:
  print(chunk.text)
  print("_"*80)







# mistral AI

import os
from mistralai.client import MistralClient

api_key = os.environ["MISTRAL_API_KEY"]

client = MistralClient(api_key=api_key)

model = "codestral-latest"
prompt = "def fibonacci(n: int):"
suffix = "n = int(input('Enter a number: '))\nprint(fibonacci(n))"

response = client.completion(
    model=model,
    prompt=prompt,
    suffix=suffix,
)

print(
    f"""
{prompt}
{response.choices[0].message.content}
{suffix}
"""
)






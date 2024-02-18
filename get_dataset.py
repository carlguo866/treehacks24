#%% 
import datasets

from datasets import load_dataset

dataset = load_dataset("FredZhang7/all-scam-spam")
# %%
dataset['train'][0]

#%% 
import json 


with open("data.jsonl", "w") as f:
    for dictionary in dataset['train']:
        line = dictionary['text']
        label = dictionary['is_spam']
        if label == 1:
            f.write(json.dumps({"text": line + " Spam. "}) + '\n')
        else: 
            f.write(json.dumps({"text": line + " Normal."}) + '\n')
        
# %%

import torch
import pickle
import pandas as pd

path = "Input your path"

def load_pt(path):
    
    try:
        return torch.load(path, map_location="cpu")
    except:
        pass
    
    
    try:
        with open(path, "rb") as f:
            return torch.load(f, map_location="cpu", pickle_module=pickle)
    except:
        pass
    
    raise RuntimeError("Failed to load .pt file using all methods.")

# load data
data = load_pt(path)

print("Loaded type:", type(data))
print(data)

has_text = hasattr(data, "raw_texts")
has_label = hasattr(data, "y")

if not (has_text and has_label):
    raise ValueError("Dataset missing raw_texts or y fields.")


texts = data.raw_texts[:500]
labels = data.y[:500]

df = pd.DataFrame({
    "text": [str(t) for t in texts],
    "label": labels.tolist()
})

df.to_csv("Input your path", index=False)

print("Saved reddit_samples500.csv successfully!")


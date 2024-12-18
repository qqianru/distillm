import datasets
import os
import re

print(os.listdir('/content/distillm'))

dataset = datasets.load_dataset('json', '/content/distillm/test.json', split='train')

os.makedirs("data/openwebtext", exist_ok=True)

num = 0
with open("data/openwebtext/data.txt", "w") as f:
    for data in dataset:
        f.write(re.sub(r"\n+", "<@x(x!>", data['text']) + "\n")
        num += 1

print("Number of lines:", num)

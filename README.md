# Two-Sentences-Are-Similar

# It's three different training loop for train your data on pretrained model.

if you want test one data on end of jupyter I show how can use it
## first you should tokenize your sentences
```
sentences = tokneizer({
    "sentence1":"...",
    "sentence2":"..."
})
```
## second you should change it to tensor for you can use it
```
sentences['input_ids'] = torch.tensor(sentences['input_ids'])
sentences['token_type_ids'] = torch.tensor(sentences['token_type_ids'])
sentences['attention_mask'] = torch.tensor(sentences['attention_mask'])
```
## third you should move it to your device
```
sentences = {key: value.to('cuda:0') for key, value in sentences.items()}
```
## final you can predict your inputs
```
with torch.no_grad():
    out = model(sentences['input_ids'].unsqueeze(0), sentences['attention_mask'].unsqueeze(0),
                  sentences['token_type_ids'].unsqueeze(0))
    print(f"the label of predict is : {torch.argmax(out.logits)}")
```
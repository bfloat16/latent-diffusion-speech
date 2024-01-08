import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bert_feature(text, word2ph, model=AutoModelForMaskedLM.from_pretrained("chinese-roberta-wwm-ext-large", cache_dir="./pretrain").to(device), tokenizer=AutoTokenizer.from_pretrained("chinese-roberta-wwm-ext-large", cache_dir="./pretrain")):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res['hidden_states'][-3:-2], -1)[0].cpu()

    assert len(word2ph) == len(text)+2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

def get_bert_token(text, tokenizer = AutoTokenizer.from_pretrained("chinese-roberta-wwm-ext-large", cache_dir="./pretrain")):
    inputs = tokenizer(text)
    return inputs["input_ids"], tokenizer.convert_ids_to_tokens(inputs["input_ids"])
!pip install pytorch-pretrained-bert
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = FullTokenizer('/BERT/vocab.txt')

def tokenize(sent_0, sent_1):
    
    tokens_0 = sent_0
    tokens_1 = sent_1

    tokens_0 = '[CLS]' + tokens_0 + '[SEP]'
    tokens_1 = '[CLS]' + tokens_1 + '[SEP]'

    tokens_0 = tokenizer.tokenize(tokens_0)
    tokens_1 = tokenizer.tokenize(tokens_1)

    indexed_tokens_0 = tokenizer.convert_tokens_to_ids(tokens_0)
    indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokens_1)

    segments_ids_0 = [0] * len(tokens_0)
    segments_ids_1 = [1] * len(tokens_1)
    
    return (tokens_0, tokens_1), (indexed_tokens_0, indexed_tokens_1), (segments_ids_0, segments_ids_1)

def embed(indexed_tokens, segments_ids):
    
    tokens_0 = torch.tensor([indexed_tokens[0]])
    tokens_1 = torch.tensor([indexed_tokens[1]])
    segments_ids_0 = torch.tensor([segments_ids[0]])
    segments_ids_1 = torch.tensor([segments_ids[1]])
    
    with torch.no_grad():
        encoded_layers_0, _ = model(tokens_0, segments_ids_0)
        encoded_layers_1, _ = model(tokens_1, segments_ids_1)
    
    token_embeddings_0 = torch.stack(encoded_layers_0, dim=0)
    token_embeddings_0 = torch.squeeze(token_embeddings_0, dim=1)
    token_embeddings_0 = token_embeddings_0.permute(1,0,2)
    
    token_embeddings_1 = torch.stack(encoded_layers_1, dim=0)
    token_embeddings_1 = torch.squeeze(token_embeddings_1, dim=1)
    token_embeddings_1 = token_embeddings_1.permute(1,0,2)
    
    token_vecs_sum_0 = []
    token_vecs_sum_1 = []

    for token in token_embeddings_0:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum_0.append(sum_vec)
    for token in token_embeddings_1:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum_1.append(sum_vec)
    
    return token_vecs_sum_0, token_vecs_sum_1
    
    
    
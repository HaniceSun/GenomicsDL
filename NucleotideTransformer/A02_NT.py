from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

# Create a dummy dna sequence and tokenize it
sequences = ['ATTCTG' * 9]
tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt")["input_ids"]

# Compute the embeddings
attention_mask = tokens_ids != tokenizer.pad_token_id
torch_outs = model(
    tokens_ids,
    attention_mask=attention_mask,
    encoder_attention_mask=attention_mask,
    output_hidden_states=True
)

# Compute sequences embeddings
embeddings = torch_outs['hidden_states'][-1].detach().numpy()
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings per token: {embeddings}")

# Compute mean embeddings per sequence
mean_sequence_embeddings = torch.sum(attention_mask.unsqueeze(-1)*embeddings, axis=-2)/torch.sum(attention_mask, axis=-1)
print(f"Mean sequence embeddings: {mean_sequence_embeddings}")

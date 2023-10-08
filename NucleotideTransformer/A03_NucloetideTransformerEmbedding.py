import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model
import numpy as np
import os
import h5py

## to avoid quota error: ln -s /oak/stanford/groups/agloyn/hansun/MLdiabetes/06-NucleotideTransformer/.cache/nucleotide_transformer ~/.cache/nucleotide_transformer
# ['500M_human_ref', '500M_1000G', '2B5_1000G', '2B5_multi_species']
def getModel(model_name='500M_human_ref', embeddings_layers_to_save=(20,), max_positions=32):
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
            model_name=model_name,
            mixed_precision=False,
            embeddings_layers_to_save=embeddings_layers_to_save,
            max_positions=max_positions,
            )
    forward_fn = hk.transform(forward_fn)
    return([parameters, forward_fn, tokenizer, config, embeddings_layers_to_save[0]])

def getSeq(n, NT=['A', 'T', 'C', 'G']):
    if n > 1:
        s = getSeq(n - 1)
        return([x + y for y in NT for x in s])
    elif n == 1:
        return(NT)
    else:
        return(['N'])

def TransformerEmbedding(model_name='500M_human_ref', n_mers=6, max_tokens=1000, extra=['<unk>', '<pad>', '<mask>', '<cls>', '<eos>', '<bos>']):
    parameters, forward_fn, tokenizer, config, embeddings_layer = getModel(model_name=model_name)
    random_key = jax.random.PRNGKey(0)

    tokenL = []
    for n in range(n_mers+1):
        tokenL += getSeq(n)
    tokenL += extra

    tokens_ids = []
    for tk in tokenL:
        try:
            tk_id = tokenizer.token_to_id(tk)
        except:
            tk_id = tokenizer.token_to_id('N')
        tokens_ids.append(tk_id)

    L = []
    for n in range(0, len(tokens_ids), max_tokens):
        tokens_ids_part = tokens_ids[n:n+max_tokens]
        tokens = jnp.asarray([tokens_ids_part], dtype=jnp.int32)
        outs = forward_fn.apply(parameters, random_key, tokens)
        res = outs['embeddings_%s'%embeddings_layer][0]
        L.append(res)
    res = np.concatenate(L, axis=0)

    '''
    with h5py.File('NucleotideTransformer-%s.h5'%model_name, 'w') as h5:
        for n in range(len(tokenL)):
            h5.create_dataset(tokenL[n], data=res[n:n+1])
    '''

    with h5py.File('NucleotideTransformer.h5', 'a') as h5:
        for n in range(len(tokenL)):
            h5.create_dataset(model_name + '/' + tokenL[n], data=res[n:n+1])


TransformerEmbedding('500M_human_ref')
TransformerEmbedding('500M_1000G')
TransformerEmbedding('2B5_1000G')
TransformerEmbedding('2B5_multi_species')

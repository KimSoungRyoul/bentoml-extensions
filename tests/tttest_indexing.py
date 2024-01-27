# import torch
# from torch.nn.utils.rnn import pad_sequence
#
# # Example vocabulary
# vocab = ["apples", "bananas", "orange", "love", "and", "grapefruit","I", "mango"]
#
# # Create a dictionary mapping words to unique integers
# word_to_idx = {word: i for i, word in enumerate(vocab)}
#
# # Example sentence
# sentence = "I love apples and bananas"
#
# # Tokenize the sentence
# tokens = sentence.split()
#
# # Convert tokens to integers using the dictionary
# int_tokens = [word_to_idx[token] for token in tokens]
#
# # Create a one-hot embedding for each token
# embeddings = torch.eye(len(vocab))[int_tokens]
#
# # Pad the sequence to a fixed length (optional)
# padded_embeddings = pad_sequence([embeddings], batch_first=True)
#
# # Alternatively, use nn.Embedding layer for larger vocabularies
# embedding_layer = torch.nn.Embedding(len(vocab), embedding_dim=10)
# int_tokens = torch.tensor(int_tokens)
# embedded_sentence = embedding_layer(int_tokens)
#
# print(f"One-hot embeddings: {embeddings}")
# print(f"Padded embeddings: {padded_embeddings}")
# print(f"Embeddings from nn.Embedding: {embedded_sentence}")
import asyncio
from collections import defaultdict

import torch

vocab = defaultdict(int, {"apple": 1, "banana": 2, "orange": 3, "love": 4})
sentence = "I love apple and banana"

tokens = sentence.split()
int_tokens = [vocab[token] for token in tokens]
int_tensor = torch.IntTensor(int_tokens)

print(f"Sentence: {sentence}")
print(f"Tokens: {tokens}")
print(f"intTensor: {int_tensor}")

import numpy as np

arr = np.array([["청바지", "유튜브"], ["검색어", "하하하"]])

# tarr  = torch.from_numpy(arr)

print(arr.dtype, arr)

# print(tarr.dtype, tarr)

import random


async def asdf_co(aa):
    await asyncio.sleep(random.randint(0, 6))
    r = "234" + aa
    print(r)
    return r


def task_aiter_asdf(aa):
    yield asdf_co(aa)


async def main():
    task1 = asyncio.create_task(asdf_co("하하"))
    task2 = asyncio.create_task(asdf_co("하하222"))
    task_list = [asyncio.create_task(asdf_co(f"하하{i}")) for i in range(10)]
    args_list = [f"하하{i}" for i in range(10)]
    #
    # async for t in task_aiter_asdf("aa"):
    #     print(t)

    print("*" * 10)
    result = await asyncio.gather(*task_list)
    print("result:", result)
    print("-" * 10)
    result = await asyncio.gather(*task_list)
    print("result:", result)
    print("-" * 10)
    result = await asyncio.gather(*task_list)
    print("result:", result)
    print("-" * 10)


asyncio.run(main())

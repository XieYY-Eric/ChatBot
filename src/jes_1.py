from __future__ import unicode_literals, print_function, division
from pickle import NONE

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from matplotlib import pyplot as plt
import os
from logger import Logger
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import typing
import random
import nltk

class Tokens:
    SOS = "<START>"
    EOS = "<EOL>"
    UNKNOWN_TOKEN = "<UNKNOWN>"
    PAD = "<PAD>"
    
    @staticmethod
    def get_all_utility_tokens() -> typing.List[str]:
        """Returns a list of all custom utility tokens."""
        # NOTE(Sean) add all tokens above to list
        return [Tokens.SOS, Tokens.EOS, Tokens.UNKNOWN_TOKEN, Tokens.PAD]

def tokenized_sentence_to_indices(tokenized_sentence: typing.List[str], query_size: int, word_dict, add_eos_token: bool = False) -> typing.List[int]:
    """Takes a tokenized sentence and returns a list of indices for each of the tokens."""
    sentence = [word_dict[token] if token in word_dict else word_dict[Tokens.UNKNOWN_TOKEN] for token in tokenized_sentence]
    # add <EOS> token
    sentence += [word_dict[Tokens.EOS]] if add_eos_token else []
    # add padding
    sentence += [0]*(query_size - len(sentence))
    return sentence

def tokenize_sentence(sentence: str, 
                      strip_tokens: typing.List[str] = ['\'', ".", "?", ",", "--"], 
                      tokenizer = nltk.word_tokenize,
                      token_list: typing.List[str] = None,
                      verbose_logging: bool = False) -> typing.List[str]:
    """takes a sentence and tokenizes, then strips any characters in the strip_tokens list.
    if token_list is not None, any tokens not found in the token list will be replaced with the <unknown> token"""
    tokens = [word.lower() for word in tokenizer(sentence) if word not in strip_tokens]
    
    if verbose_logging:
        logger.log_debug(f"sentence after tokenizer: {' '.join(tokens)}")
    
    # replace unknown tokens with <UNKNOWN>
    if token_list is not None:
        tokens = [token if token in token_list else Tokens.UNKNOWN_TOKEN for token in tokens]
        
        if verbose_logging:
            logger.log_debug(f"sentence after stripping: {' '.join(tokens)}")
    
    return tokens

def tokenize(
    dataset_path: str, 
    remove_punctuation: bool = True,
    force_compute: bool = False) -> typing.List[typing.List[str]]:
    """
    Tokenize transcript sentences using nltk.
    Return a list of tokenized sentences
    """
    # read dataset and store lines
    words: typing.List[str] = []
    with open(dataset_path, "r") as f:
        words = [word.strip().lower() for word in f.readlines()]

    logger.log_info(f"starting to tokenizing sentences from '{dataset_path}'")
    
    # check if cached version of tokenized sentences exists
    cache_path = os.path.join("./data", TOKENS_CACHED_FILENAME)
    if os.path.isfile(cache_path) and not force_compute:
        logger.log_info("found cached tokenized sentences!")
        sentences_as_tokens = None
        with open(cache_path, "rb") as f_bytes:
            sentences_as_tokens = pickle.load(f_bytes)
        
        logger.log_info("finished loading cached tokenized sentences!")
        return sentences_as_tokens
    
    if not os.path.isfile(cache_path):
        logger.log_warn("could not find cache of tokenized sentences.")
    
    start_time = time.time()
    
    # use nltk to tokenize each line of the transcript
    # NOTE(Sean) should we add start-of-line and end-of-line tokens?
    sentences_as_tokens: typing.List[typing.List[str]] = [nltk.word_tokenize(word) for word in words]
    
    # exclude punctuation
    # NOTE(Sean) instead of completely removing, maybe add <PUNC> token instead?
    if remove_punctuation:
        logger.log_info("removing punctuation from tokenized sentences...")
        sentences_without_punc: typing.List[typing.List[str]] = []
        for sentence in sentences_as_tokens:
            strip_tokens = ['\'', ".", "?", ",", "--"]
            stripped_sentence = [token for token in sentence if token not in strip_tokens]
            if stripped_sentence:
                sentences_without_punc.append(stripped_sentence)
        sentences_as_tokens = sentences_without_punc
        logger.log_info("finished removing punctuation")
    
    
    logger.log_info(f"finished tokenizing sentences from '{dataset_path}', took {time.time() - start_time:.2f} seconds")
    
    logger.log_info(f"caching tokenized sentences in '{cache_path}'...")
    
    with open(cache_path, "wb") as f_bytes:
        pickle.dump(sentences_as_tokens, f_bytes)
    
    logger.log_info("finished caching tokenized sentences.")
    
    return sentences_as_tokens

def simple_token(sentence):
    tokens= nltk.word_tokenize(sentence)
    tokens = [token.lower() for token in tokens]
    return tokens

def output_to_sentence_str(output: typing.Union[list[int], torch.Tensor], token_list: list[str], include_padding: bool = False):
    if type(output) is torch.Tensor:
        if len(output.shape) == 3:
            output = output[0]  
    
    if include_padding:
        return " ".join([token_list[tok] for tok in output])
    else:
        return " ".join([token_list[tok] for tok in output if tok != token_list.index(Tokens.PAD)])

def get_cornell_movie_lines(filename:str,number_of_data:int=-1,delimiter="\t"):
    """
    read data lines from a file, delimiter is what separate a sentence, by default, it is a tap
    """
    if not os.path.exists(filename):
        logger.log_error("File doesn't exist")
        return []
    result = []
    with open(filename,"r") as f:
        for i,data in enumerate(f):
            if number_of_data!=-1 and i == number_of_data:
                break
            if len(data.strip().split(delimiter)) != 2:
                continue
            result.append(data.strip().split(delimiter))
    return result

def evaluate(model,test_pairs,token_list,max_word):
    logger.log_info("starting to evaluate model...")
    X_train,Y_train = [pair[0] for pair in test_pairs], [pair[1] for pair in test_pairs]
    #sentence as int for now
    predicted = [model.predict([x],max_word, topk_choices=1) for x in X_train]
    predicted = [output_to_sentence_str(p,token_list=token_list) for p in predicted]
    Y_train = [output_to_sentence_str(y,token_list=token_list) for y in Y_train]
    logger.log_info("finished evaluating model")
    logger.log_info(f"  score: {bleu(Y_train,predicted)}")

try:
    from typing_extensions import Final
except:
    # If you don't have `typing_extensions` installed, you can use a
    # polyfill from `torch.jit`.
    from torch.jit import Final
from math import sqrt

logger = Logger()

# ================
# hyper parameters
# ================
MAX_LENGTH: int = 24
EMBEDDING_SIZE: int = 64 # gpt3 has 12288 embedding dims (for pre-trained model, set to 64)
ATTENTION_SIZE: int = 8 # gpt uses attention size of 64 (for pre-trained model, set to 8)
MAX_EPOCHS: int = 500
BATCH_SIZE: int = 1024
DECODER_LAYER_COUNT: int = 4 # gpt3 has 96 layers (for pre-trained model, set to 4)
MODEL_NAME: str = "jes-1"
LEARNING_RATE: float = 0.002
SKIP_LOADING_MODEL: bool = False # set to true to train from scratch, False will try to load a pre-trained model
EARLY_STOPPING_THRESHOLD: typing.Union[float, None] = None #0.0001
VERBOSE_LOG: bool = True
EVALUATE_MODEL: bool = False # set to true to compute BLEU evaluation scores
DEVICE: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# ================

class data:
    def __init__(self, sentences_as_tokens, max_length) -> None:
        self.all_tokens = [Tokens.PAD, Tokens.SOS, Tokens.EOS, Tokens.UNKNOWN_TOKEN]
        self.word_dict = {
            Tokens.PAD: self.all_tokens.index(Tokens.PAD),
            Tokens.SOS: self.all_tokens.index(Tokens.SOS),
            Tokens.EOS: self.all_tokens.index(Tokens.EOS), 
            Tokens.UNKNOWN_TOKEN: self.all_tokens.index(Tokens.UNKNOWN_TOKEN)
        }
        # check for programmer error
        assert len(self.all_tokens) == len(self.word_dict), "uh oh, looks like you forgot to add a token!"
        
        self.num_of_word = len(self.all_tokens)
        self.all_datas = []
        self.number_of_sentence = len(sentences_as_tokens)
        #read all the tokens from the sentences and generate a dict and list
        all_words = []
        for sentence in sentences_as_tokens:
            for token in sentence[:max_length]:
                all_words.append(token)
        self.freqency = nltk.FreqDist(all_words)
        unique_word = list(self.freqency.keys())
        for token in unique_word:
            self.word_dict[token] = self.num_of_word
            self.num_of_word += 1
        self.all_tokens += unique_word
        for i,sentence in enumerate(sentences_as_tokens):
            self.all_datas.append([])
            for token in sentence[:max_length]:
                self.all_datas[i].append(self.word_dict[token])
    
    def load(self, checkpoint):
        """Load dataset from a saved checkpoint"""
        if checkpoint:
            if checkpoint.get("dataset", None) is not None:
                self.__dict__ = checkpoint["dataset"]
                logger.log_info("  dataset loaded from checkpoint")
            else:
                logger.log_warn("  dataset failed to load from checkpoint!")
            
        return self

    def plotDistribution(self,topv=10):
        most_common = self.freqency.most_common(topv)
        total_word = sum([v for k,v in self.freqency.items()])
        fig, ax = plt.subplots()
        names,values = zip(*most_common)
        names = list(names)
        values = list(values)
        ax.bar(names,values)
        plt.show()
    
    def get_word_dict(self):
        return self.word_dict

    def get_token_list(self):
        return self.all_tokens

    def get_num_word(self):
        return self.num_of_word

    def get_sentence(self,num=1):
        return self.all_datas[:num]

    def get_num_sentence(self):
        return self.number_of_sentence

# testing for the JES-1 model, based on the gpt-3 architecture

# attention layer based on (simplified) gpt3 architecture
# reference https://dugas.ch/artificial_curiosity/GPT_architecture.html
class AttentionJES(nn.Module):
    
    def __init__(self, input_size, attention_size, device: torch.device, use_bias: bool = False, dropout_percent: float = 0.05):
        super(AttentionJES, self).__init__()
        # multiply (n x input_size) x (input_size x output_size), resulting in n x output_size matrix
        self.query_attn: nn.Linear = nn.Linear(input_size, attention_size, bias=use_bias)
        self._query_dropout: nn.Dropout = nn.Dropout(dropout_percent)
        # TODO document...
        self.key_attn = nn.Linear(input_size, attention_size, bias=use_bias)
        self._key_dropout: nn.Dropout = nn.Dropout(dropout_percent)
        # TODO document...
        self.values_attn = nn.Linear(input_size, attention_size, bias=use_bias)
        self._values_dropout: nn.Dropout = nn.Dropout(dropout_percent)
        # softmax layer that "importance" is passed through
        self.softmax = F.softmax
        # device the attention layer is operating on
        self.__device: torch.device = device
        
    def forward(self, embedding_input, use_lookahead_mask: bool) -> typing.Tuple[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Returns attention probabilities, and individual Q, K, V tensors as a secondary tuple"""
        # TODO(Sean) tbh, i don't know if this is fully correct? it seems to work and the math is based on the paper.
        # the input might be wrong
        
        # Q tensor represents attention to "queries"
        Q: torch.Tensor = self._query_dropout(self.query_attn(embedding_input))
        # K tensor represents attention to "keys"
        K: torch.Tensor = self._key_dropout(self.key_attn(embedding_input))
        # V tensor represents attention to "values"
        V: torch.Tensor = self._values_dropout(self.values_attn(embedding_input))
        # importance is scaled by the dimension of the K tensor, to not have exploading gradients
        dk: int = embedding_input.shape[-1]
        scale: float = sqrt(dk)
        # compute an "importance" tensor, which represents how "important" each token is to eachother
        Q_K_t: torch.Tensor = Q.bmm(K.mT) / scale
        # compute look ahead mask to mask future tokens that should not have attention
        look_ahead_mask: torch.Tensor = torch.zeros(Q_K_t.shape, dtype=torch.float, device=self.__device)
        if use_lookahead_mask:
            # TODO(Sean) there is probably a faster way to do this...
            for j in range(Q_K_t.shape[1]):
                for k in range(Q_K_t.shape[2]):
                    look_ahead_mask[:,j,k] = 0.0 if j >= k else -torch.inf
        # get importance probs by passing through softmax
        importance: torch.Tensor = self.softmax(Q_K_t + look_ahead_mask, dim=2)
        # then multiply the value tensor in, where each token has a mix of all other token values weighted by the importance of their token
        output: torch.Tensor = importance.bmm(V)
        return output, (Q, K, V)
    
# TODO(Sean) sparse attention based on paper from https://openai.com/blog/block-sparse-gpu-kernels/

    
class FeedForward(nn.Module):
    
    def __init__(self, input_size, embedding_size, hidden_size_multiplier: int = 4, use_bias: bool = True):
        super(FeedForward, self).__init__()
        # hidden layer uses multiplier * embedding size for number of hidden nodes
        self.__hidden_size_multiplier = hidden_size_multiplier
        # gpt3 uses 4 * embedding_size for their hidden layer, with bias
        self._hidden: nn.Linear = nn.Linear(input_size, self.__hidden_size_multiplier * embedding_size, bias=use_bias)
        # linear output projection layer
        # NOTE(Sean) should output have bias?
        self._output: nn.Linear = nn.Linear(self.__hidden_size_multiplier * embedding_size, embedding_size, bias=use_bias)
        # relu layer
        self._relu = nn.ReLU()
        # noramalization layer
        self._norm_layer: nn.LayerNorm = nn.LayerNorm(embedding_size)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # pass through hidden layer
        hidden: torch.Tensor = self._hidden(input)
        # pass through output and relu layers
        output: torch.Tensor = self._relu(self._output(hidden))
        # output of ff network is added to input and normalized
        return self._norm_layer(input + output)


# ==============================================================
# from models.py, modified to have batch size as first dimension
# ==============================================================

# TODO(Sean) comment the code below

def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return torch.BoolTensor(m).to(torch.device("cpu"))

def maskNLLLoss(inp, target, mask):
    # TODO(Sean) make target and mask correct shape in batch creation code so transpositions are not necessary
    nTotal = mask.sum()
    new_target_t = target.view(target.shape[0], target.shape[1], 1)
    gathered = torch.gather(inp, 2, new_target_t).squeeze(2)
    cross_entropy = -torch.log(gathered)
    loss = cross_entropy.masked_select(mask).mean()
    return loss, nTotal.item() 

def get_X_train(X, query_size: int, add_eos_token: bool = True):
    length = torch.tensor([len(sentence) for sentence in X], device=torch.device("cpu"))
    for i,s in enumerate(X):
        if add_eos_token:
            if len(s) < query_size:
                s.append(2)
            else:
                s[query_size - 1] = 2
        padding_count = query_size - length[i] - (1 if add_eos_token else 0)
        s.extend([0]*(padding_count))
    return torch.tensor(X, device=torch.device("cpu")), length

def get_Y_train(Y, query_size: int):
    max_Y_length = max([len(sentence) for sentence in Y])
    for i,s in enumerate(Y):
        if len(s) < query_size:
            s.append(2)
        else:
            s[max_Y_length - 1] = 2
        s.extend([0]*(query_size - len(s)))
    Y = torch.tensor(Y, device=torch.device("cpu"))
    return Y,binaryMatrix(Y),max_Y_length 

def data_to_sequence(pairs, query_size: int):
    sorted_pairs = sorted(pairs, key=lambda pair:len(pair[0]),reverse=True)
    length = torch.tensor([len(pair[0]) for pair in sorted_pairs], device=torch.device("cpu"))
    X , Y = [] , []
    for pair in pairs:
        X.append(pair[0][:])
        Y.append(pair[1][:])
    X,length = get_X_train(X, query_size=query_size)
    Y,mask,max_Y_length = get_Y_train(Y, query_size=query_size)
    return X,length,Y,mask,max_Y_length

# ==============================================================

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size: int, attention_size: int, device: torch.device):
        super(TransformerEncoder, self).__init__()
        # dimensions of the embedding
        self.__embedding_size: Final[int] = embedding_size
        # dimensionality of each attention head
        self.__attention_size: Final[int] = attention_size
        # create num_attention attention layers (GPT3 uses 96)
        assert self.__embedding_size % self.__attention_size == 0, "embedding size must be divisible by attention size"
        # number of attention heads
        self.__num_attention: Final[int] = int(self.__embedding_size / self.__attention_size)
        # device that the transformer layer will compute on
        self.__device: Final[torch.device] = device
        # linear layer that the output of the multi-head attention is passed through
        self._attn_linear_proj: nn.Linear = nn.Linear(self.__embedding_size, self.__embedding_size, bias=False)
        # feed forward encoding layer
        self._ff_encoder: FeedForward = FeedForward(self.__embedding_size, self.__embedding_size, hidden_size_multiplier=4, use_bias=True).to(self.__device)
        # set attention layers
        self._attention_layers: nn.ModuleList = nn.ModuleList((AttentionJES(self.__embedding_size, self.__attention_size, device=self.__device) for _ in range(self.__num_attention)))
        # softmax layer
        self._softmax = F.softmax
        # attention normalization layer
        self._attn_norm_layer: nn.LayerNorm = nn.LayerNorm(self.__embedding_size)
        # feed forward normalization layer
        self._feed_forward_norm_layer: nn.LayerNorm = nn.LayerNorm(self.__embedding_size)
    
    def forward(self, embedding: torch.Tensor, positional_encoding: torch.Tensor) -> typing.Tuple[torch.Tensor, typing.List[typing.Tuple[torch.Tensor, torch.Tensor]]]:
        # a single attention output results in a tensor (MAX_SIZE x attention_size)
        # add the two tensors together since they are the same shape
        embedding_and_attn: torch.Tensor = embedding + positional_encoding
        #attentions = tuple(getattr(self, f"_attn{i}")(embedding_and_attn) for i in range(self.__num_attention))
        attentions: typing.List[torch.Tensor] = []
        attn_kv_tensors: typing.List[typing.Tuple[torch.Tensor, torch.Tensor]] = []
        for attention_layer in self._attention_layers:
            attn, qvk_tup = attention_layer(embedding_and_attn, True)
            attentions.append(attn)
            attn_kv_tensors.append(tuple([qvk_tup[1], qvk_tup[2]]))
        # concatenate all attentions, resulting in a (MAX_SIZE x embedding_size) tensor
        attention: torch.Tensor = torch.cat(attentions, dim=2)
        # pass through a linear layer
        attention: torch.Tensor = self._attn_linear_proj(attention)
        # add & normalize multi-head attention
        multi_head_attn: torch.Tensor = self._attn_norm_layer(embedding_and_attn + attention)
        # pass through feed forward layer (encoder)
        encoded: torch.Tensor = self._ff_encoder(multi_head_attn)
        # add & normalize feed forward output
        encoded: torch.Tensor = self._feed_forward_norm_layer(multi_head_attn + encoded)
        
        return encoded, attn_kv_tensors
    
class TransformerDecoder(nn.Module):
    def __init__(self, embedding_size: int, attention_size: int, device: torch.device):
        super(TransformerDecoder, self).__init__()
        # dimensions of the embedding
        self.__embedding_size: Final[int] = embedding_size
        # dimensionality of each attention head
        self.__attention_size: Final[int] = attention_size
        # create num_attention attention layers (GPT3 uses 96)
        assert self.__embedding_size % self.__attention_size == 0, "embedding size must be divisible by attention size"
        # number of attention heads
        self.__num_attention: Final[int] = int(self.__embedding_size / self.__attention_size)
        # device that the transformer layer will compute on
        self.__device: Final[torch.device] = device
        # linear layer that the output of the multi-head attention is passed through
        self._input_attn_linear_proj: nn.Linear = nn.Linear(self.__embedding_size, self.__embedding_size, bias=False)
        # feed forward encoding layer
        self._ff_decoder: FeedForward = FeedForward(self.__embedding_size, self.__embedding_size, hidden_size_multiplier=4, use_bias=True).to(self.__device)
        # set attention layers for decoder input
        self._input_attention_layers: nn.ModuleList = nn.ModuleList((AttentionJES(self.__embedding_size, self.__attention_size, device=self.__device) for _ in range(self.__num_attention)))
        # softmax layer
        self._softmax = F.softmax
        # input layer normalization
        self._input_layer_norm: nn.LayerNorm = nn.LayerNorm(self.__embedding_size)
        # layer normalization for attention with K, V tensors from encoder, and Q tensor from decoder
        self._encoder_decoder_layer_norm: nn.LayerNorm = nn.LayerNorm(self.__embedding_size)
        # output layer normalization
        self._output_layer_norm: nn.LayerNorm = nn.LayerNorm(self.__embedding_size)
        
    def qvk_project(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, use_lookahead_mask: bool = False):
        """Compute self-attention with pre-computed Q, K, and V tensors"""
        # importance is scaled by the dimension of the K tensor, to not have exploading gradients
        scale: float = sqrt(K.shape[1])
        # compute an "importance" tensor, which represents how "important" each token is to eachother
        Q_K_t: torch.Tensor = Q.bmm(K.mT) / scale
        # get importance probs by passing through softmax
        look_ahead_mask: torch.Tensor = torch.zeros(Q_K_t.shape, dtype=torch.float, device=self.__device)
        if use_lookahead_mask:
            # TODO(Sean) there is probably a faster way to do this...
            for j in range(Q_K_t.shape[1]):
                for k in range(Q_K_t.shape[2]):
                    look_ahead_mask[:,j,k] = 0.0 if j >= k else -torch.inf
        importance: torch.Tensor = self._softmax(Q_K_t + look_ahead_mask, dim=2)
        # then multiply the value tensor in, where each token has a mix of all other token values weighted by the importance of their token
        output: torch.Tensor = importance.bmm(V)
        return output
    
    def forward(self, embedding: torch.Tensor, positional_encoding: torch.Tensor, kv_tensors: typing.List[torch.Tensor]) -> torch.Tensor:
        """"""
        # =============
        # decoder input
        # =============
        # a single attention output results in a tensor (MAX_SIZE x attention_size)
        # add the two tensors together since they are the same shape
        embedding_and_attn: torch.Tensor = embedding + positional_encoding
        #attentions = tuple(getattr(self, f"_attn{i}")(embedding_and_attn) for i in range(self.__num_attention))
        attentions: typing.List[torch.Tensor] = []
        attn_q_tensors: typing.List[torch.Tensor] = []
        for attention_layer in self._input_attention_layers:
            attn, qvk_tup = attention_layer(embedding_and_attn, True)
            attentions.append(attn)
            attn_q_tensors.append(qvk_tup[0])
        # concatenate all attentions, resulting in a (MAX_SIZE x embedding_size) tensor
        attention: torch.Tensor = torch.cat(attentions, dim=2)
        # pass through a linear layer
        attention: torch.Tensor = self._input_attn_linear_proj(attention)
        # add & normalize multi-head attention
        input_multi_head_attn: torch.Tensor = self._input_layer_norm(embedding_and_attn + attention)
        # =============
        
        # =====================
        # input query + decoder
        # =====================
        #attentions = tuple(getattr(self, f"_attn{i}")(embedding_and_attn) for i in range(self.__num_attention))
        output_attentions: typing.Tuple[torch.Tensor] = tuple(self.qvk_project(Q=attn_q_tensors[j], K=kv_tensors[i][j][0], V=kv_tensors[i][j][1], use_lookahead_mask=False) for j in range(len(self._input_attention_layers)) for i in range(len(kv_tensors)))
        # concatenate all attentions, resulting in a (MAX_SIZE x embedding_size) tensor
        output_attention: torch.Tensor = torch.cat(output_attentions, dim=2)
        # add&norm
        final_attn = self._encoder_decoder_layer_norm(output_attention + input_multi_head_attn)
        # pass through feed forward layer (encoder)
        decoded: torch.Tensor = self._ff_decoder(final_attn)
        # add & normalize multi-head attention for input + output embedding
        output_multi_head_attn: torch.Tensor = self._output_layer_norm(final_attn + decoded)
        # =====================
        return output_multi_head_attn

class Transformer(nn.Module):
    
    def __init__(self, embedding_size: int, attention_size: int, device: torch.device):
        super(Transformer, self).__init__()
        # dimensions of the embedding
        self.__embedding_size: Final[int] = embedding_size
        # dimensionality of each attention head
        self.__attention_size: Final[int] = attention_size
        # create num_attention attention layers (GPT3 uses 96)
        assert self.__embedding_size % self.__attention_size == 0, "embedding size must be divisible by attention size"
        # number of attention heads
        self.__num_attention: Final[int] = int(self.__embedding_size / self.__attention_size)
        # device that the transformer layer will compute on
        self.__device: Final[torch.device] = device
        # linear layer that the output of the multi-head attention is passed through
        self._attn_linear_proj: nn.Linear = nn.Linear(self.__embedding_size, self.__embedding_size, bias=False)
        # feed forward encoding layer
        self._ff_encoder: FeedForward = FeedForward(self.__embedding_size, self.__embedding_size, hidden_size_multiplier=4, use_bias=True).to(self.__device)
        # set attention layers
        self._attention_layers: nn.ModuleList = nn.ModuleList((AttentionJES(self.__embedding_size, self.__attention_size, device=self.__device) for _ in range(self.__num_attention)))
        # softmax layer
        self._softmax = F.softmax
        
    @staticmethod
    def get_attention_head_count(embedding_size: int, attention_size: int) -> int:
        """calculate the number of attention heads a transformer layer will have"""
        assert embedding_size % attention_size == 0, "embedding size must be divisible by attention size"
        return int(embedding_size / attention_size)
    
    def forward(self, embedding: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        # a single attention output results in a tensor (MAX_SIZE x attention_size)
        # add the two tensors together since they are the same shape
        embedding_and_attn: torch.Tensor = embedding + attn_bias
        #attentions = tuple(getattr(self, f"_attn{i}")(embedding_and_attn) for i in range(self.__num_attention))
        attentions: typing.Tuple[torch.Tensor] = tuple(attention_layer(embedding_and_attn) for attention_layer in self._attention_layers)
        # concatenate all attentions, resulting in a (MAX_SIZE x embedding_size) tensor
        attention: torch.Tensor = torch.cat(attentions, dim=2)
        # pass through a linear layer
        attention: torch.Tensor = self._attn_linear_proj(attention)
        # add & normalize multi-head attention
        multi_head_attn: torch.Tensor = torch.nn.functional.normalize(embedding_and_attn + attention)
        # pass through feed forward layer (encoder)
        encoded: torch.Tensor = self._ff_encoder(multi_head_attn)
        # add & normalize feed forward output
        encoded: torch.Tensor = torch.nn.functional.normalize(multi_head_attn + encoded)
        
        return encoded
    
class JESModel():
    def __init__(self, 
                 tokens_size: int, 
                 query_len: int, 
                 embedding_size: int, 
                 attention_size: int, 
                 decoder_layer_count: int,
                 encoder_layer_count: int, 
                 device: typing.Union[torch.device, None], 
                 model_name: str,
                 dataset_name: str,
                 data_directory_path: str,
                 learning_rate: float = 0.002,
                 verbose_log: bool = True):
        # dimensionality of the embedding
        self.__embedding_size: Final[int] = embedding_size
        # number of tokens in our vocabulary
        self.__tokens_size: Final[int] = tokens_size
        # device the model will compute on
        self.__device: torch.device = device if device is not None else torch.device("cpu")
        # learning rate for the optimizer
        self.__learning_rate: Final[float] = learning_rate
        # dimensionality of an attention head
        self.__attention_size: Final[int] = attention_size
        # number of transformer layers
        self.__encoder_layer_count: Final[int] = encoder_layer_count
        self.__decoder_layer_count: Final[int] = decoder_layer_count
        # include more logging?
        self.__verbose_log: Final[bool] = verbose_log
        # maximum length of a query
        self.__query_length: Final[int] = query_len
        # embedding layer
        self._embedding: nn.Embedding = nn.Embedding(self.__tokens_size, embedding_dim=embedding_size).to(self.__device)
        # linear projection layer
        self._decoder_projection: nn.Linear = nn.Linear(self.__embedding_size, self.__tokens_size, bias=False).to(self.__device)
        # softmax function for output
        self._softmax = F.softmax
        # encoder transformer layers
        assert self.__encoder_layer_count > 0, "no layers?"
        self._encoder_layers = nn.ModuleList(TransformerEncoder(self.__embedding_size, self.__attention_size, self.__device).to(self.__device) for _ in range(self.__encoder_layer_count))
        # post encoder normalization layer
        self._post_encoder_layer_norm: nn.LayerNorm = nn.LayerNorm(self.__embedding_size).to(self.__device)
        # decoder transformer layers
        assert self.__decoder_layer_count > 0, "no layers?"
        self._decoder_layers = nn.ModuleList(TransformerDecoder(self.__embedding_size, self.__attention_size, self.__device).to(self.__device) for _ in range(self.__decoder_layer_count))
        # post encoder normalization layer
        self._post_decoder_layer_norm: nn.LayerNorm = nn.LayerNorm(self.__embedding_size).to(self.__device)
        # cache for learnable parameters
        self.__learnable_params_cache: typing.List[torch.nn.Parameter] = []
        # create optimizer
        self._optimizer = torch.optim.AdamW(self.__get_all_learnable_parameters(), self.__learning_rate, weight_decay=0.1)
        # model name (used for checkpoint filepath)
        self.__model_name: str = model_name
        # dataset name (used for checkpoint filepath)
        self.__dataset_name: str = dataset_name
        # data directory (where checkpoints will be saved)
        self.__data_directory_path: str = data_directory_path
        # parameter to start at a specific iteration when the model is loaded from a checkpoint
        self.__loaded_model_start_iteration: typing.Union[int, None] = None
        
        if self.__verbose_log:
            logger.log_info(f"JES model '{self.__model_name}' parameters:")
            logger.log_info(f"  dataset name:       {self.__dataset_name}")
            logger.log_info(f"  embedding size:     {self.__embedding_size}")
            logger.log_info(f"  attention size:     {self.__attention_size}")
            logger.log_info(f"  encoder layers:     {self.__encoder_layer_count}")
            logger.log_info(f"    attention heads:  {Transformer.get_attention_head_count(self.__embedding_size, self.__attention_size)}")
            logger.log_info(f"  decoder layers:     {self.__decoder_layer_count}")
            logger.log_info(f"    attention heads:  {Transformer.get_attention_head_count(self.__embedding_size, self.__attention_size)}")
            logger.log_info(f"  device:             {self.__device}")
            logger.log_info(f"  learning rate:      {self.__learning_rate}")
            logger.log_info(f"  learnable params:   {sum(p.numel() for p in self.__get_all_learnable_parameters() if p.requires_grad)}")
    
    def save_checkpoint(self, iteration: int, dataset) -> None:
        """save a model from a training checkpoint"""
        # directory where checkpoint file(s) will be saved
        # embedding-attention-layers-tokens-query_len
        model_parameters: str = f"{self.__embedding_size}-{self.__attention_size}-{self.__encoder_layer_count}-{self.__decoder_layer_count}-{self.__tokens_size}-{self.__query_length}"
        checkpoint_dir_path: str = os.path.join(self.__data_directory_path, self.__model_name, self.__dataset_name, model_parameters)
        
        logger.log_info(f"trying to save model '{self.__model_name}'")
        
        if not os.path.isdir(checkpoint_dir_path):
            logger.log_warn(f"  dir path '{checkpoint_dir_path}', creating directories...")
            os.makedirs(checkpoint_dir_path)
            logger.log_info("  finished making directories")
        
        checkpoint_state = {
            "iteration": iteration,
            "embedding": self._embedding.state_dict(),
            "decoder_proj": self._decoder_projection.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "encoder_layer_count": self.__encoder_layer_count,
            **{ f"encoder{i}" : encoder.state_dict() for i, encoder in enumerate(self._encoder_layers) },
            "decoder_layer_count": self.__decoder_layer_count,
            **{ f"decoder{i}" : decoder.state_dict() for i, decoder in enumerate(self._decoder_layers) },
            "post_encoder_normalization": self._post_encoder_layer_norm.state_dict(),
            "post_decoder_normalization": self._post_decoder_layer_norm.state_dict(),
            "dataset": dataset.__dict__
        }
        
        checkpoint_filepath = os.path.join(checkpoint_dir_path, f"{iteration}-checkpoint.tar")
        logger.log_info(f"  saving model to '{checkpoint_filepath}'")
        if self.__verbose_log:
            for key in checkpoint_state.keys():
                logger.log_info(f"    saved {key}")
        torch.save(checkpoint_state, checkpoint_filepath)
        
    def load_checkpoint(self, dataset, skip_loading_model: bool, load_iteration: typing.Union[None, int] = None):
        """load a model from a saved checkpoint
        optional 'load_teration' parameter can be used to load a specific checkpoint, rather than the latest one found"""
        if skip_loading_model:
            return self
        # directory where checkpoint file(s) will be saved
        # embedding-attention-layers-tokens-query_len
        model_parameters: str = f"{self.__embedding_size}-{self.__attention_size}-{self.__encoder_layer_count}-{self.__decoder_layer_count}-{self.__tokens_size}-{self.__query_length}"
        checkpoint_dir_path: str = os.path.join(self.__data_directory_path, self.__model_name, self.__dataset_name, model_parameters)
        
        logger.log_info(f"trying to load model '{self.__model_name}'")
        
        if not os.path.isdir(checkpoint_dir_path):
            logger.log_warn(f"  could not find saved checkpoints in '{checkpoint_dir_path}'! Aborting")
            return self
        
        iteration: int = 0
        # find latest iteration
        if load_iteration is None:
            logger.log_info("searching for latest checkpoint...")
            for path in os.listdir(checkpoint_dir_path): # count the ammount of files in directory to get the last checkpoint
                if os.path.isfile(os.path.join(checkpoint_dir_path, path)):
                    iteration = max(int(os.path.basename(path).split('-')[0]), iteration) # assumes filename in format "{iteration}-checkpoint.tar"
            
            logger.log_info(f"  latest checkpoint found '{iteration}'")
        else:
            iteration = load_iteration
            
        checkpoint_filepath = os.path.join(checkpoint_dir_path, f"{iteration}-checkpoint.tar")
        if not os.path.isfile(checkpoint_filepath):
            logger.log_error(f"  could not find checkpoint '{checkpoint_filepath}'! Aborting")
        
        checkpoint = None
        # TODO(Sean) i think this should be based on whether the checkpoint was saved from a model training on gpu...
        # not based on current param
        if self.__device == torch.device("cuda"):
            checkpoint = torch.load(checkpoint_filepath)
        else:
            checkpoint = torch.load(checkpoint_filepath, map_location=torch.device("cpu"))
        
        self.__loaded_model_start_iteration = iteration
        
        # load dataset state from checkpoint
        # i don't exactly know why this is necessary, probably something to do with dictionary re-ordering on subsequent runs
        # check that saved token size is same as value the model has
        if checkpoint.get("dataset", None) is not None:
            
            assert len(checkpoint["dataset"]["all_tokens"]) == self.__tokens_size, f"uh oh, self.__tokens_size={self.__tokens_size} does not match cached value = {checkpoint['dataset']['all_tokens']}"
            
            dataset.__dict__ = checkpoint["dataset"]
            logger.log_info("  loaded dataset.")
        else:
            logger.log_error("  failed to load dataset from checkpoint!")
        
        
        # load embedding
        if checkpoint.get("embedding", None) is not None:
            self._embedding.load_state_dict(checkpoint["embedding"])
            logger.log_info("  loaded embedding.")
        else:
            logger.log_error("  failed to load embedding from checkpoint!")
        
        # load decoder linear projection
        if checkpoint.get("decoder_proj", None) is not None:
            self._decoder_projection.load_state_dict(checkpoint["decoder_proj"])
            logger.log_info("  loaded decoder projection.")
        else:
            logger.log_error("  failed to load decoder projection from checkpoint!")
            
        # load optimizer
        if checkpoint.get("optimizer", None) is not None:
            self._optimizer.load_state_dict(checkpoint["optimizer"])
            logger.log_info("  loaded optimizer.")
        else:
            logger.log_error("  failed to load optimizer from checkpoint!")
        
        # =======
        # encoder
        # =======
        
        # check for programmer error...
        assert checkpoint.get("encoder_layer_count", None) == self.__encoder_layer_count, "layer count does not match checkpoint layer count!"
        
        # load transformer layers
        for i, encoder in enumerate(self._encoder_layers):
            if checkpoint.get(f"encoder{i}", None) is not None:
                encoder.load_state_dict(checkpoint[f"encoder{i}"])
                logger.log_info(f"  loaded encoder layer {i}!")
            else:
                logger.log_error(f" failed to load encoder layer {i} from checkpoint!")
    
        # load encoder normalization layer
        if checkpoint.get("post_encoder_normalization", None) is not None:
            self._post_encoder_layer_norm.load_state_dict(checkpoint["post_encoder_normalization"])
            logger.log_info("  loaded encoder normalization layer")
        else:
            logger.log_error("  failed to load encoder normalization layer")
                
        # =======
        
    
        
        # =======
        # decoder
        # =======
        
        # check for programmer error...
        assert checkpoint.get("decoder_layer_count", None) == self.__decoder_layer_count, "layer count does not match checkpoint layer count!"
        
        # load transformer layers
        for i, decoder in enumerate(self._decoder_layers):
            if checkpoint.get(f"decoder{i}", None) is not None:
                decoder.load_state_dict(checkpoint[f"decoder{i}"])
                logger.log_info(f"  loaded decoder layer {i}!")
            else:
                logger.log_error(f" failed to load decoder layer {i} from checkpoint!")
                
        # load encoder normalization layer
        if checkpoint.get("post_decoder_normalization", None) is not None:
            self._post_decoder_layer_norm.load_state_dict(checkpoint["post_decoder_normalization"])
            logger.log_info("  loaded decoder normalization layer")
        else:
            logger.log_error("  failed to load decoder normalization layer")
                
        # =======
                
        # make sure the cache of learnable params is computed after loading checkpoint states
        self.__get_all_learnable_parameters(force_recompute=True)
        
        logger.log_info(f"finished loading model '{self.__model_name}'")
                    
        return self
    
    def __get_all_learnable_parameters(self, force_recompute: bool = False) -> typing.List[torch.nn.Parameter]:
        if not self.__learnable_params_cache or force_recompute:
            # add parameters from each of the layers
            learnable_params = list(self._embedding.parameters()) + list(self._decoder_projection.parameters()) + list(self._post_encoder_layer_norm.parameters()) + list(self._post_decoder_layer_norm.parameters())
            # add parameters from all the encoder layers
            for i in range(self.__encoder_layer_count):
                learnable_params += self._encoder_layers[i].parameters()
            # add parameters from all the decoder layers
            for i in range(self.__decoder_layer_count):
                learnable_params += self._decoder_layers[i].parameters()
            # cache the learnable params so we don't re-lookup millions of parameters
            self.__learnable_params_cache = learnable_params
        
        return self.__learnable_params_cache
    
    def to(self, device: torch.device):
        """Set the device this model will use. Must be called before any forward through networks"""
        # set device
        self.__device = device
        
        # send layers to device
        self._embedding = self._embedding.to(device=self.__device)
        self._decoder_projection = self._decoder_projection.to(device=self.__device)
        for transformer_layer in self._transformer_layers:
            transformer_layer = transformer_layer.to(device=self.__device)
        
        return self
    
    def __get_positional_sine_waves(self, input: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """
        function to return a tensor the frequencies of the positional encoding sine waves.
        
        """
        '''if self.__positional_weights_t is None:
            weights = torch.zeros(size=(batch_size, 1, self.__embedding_size), dtype=torch.float)
            for i in range(batch_size):
                weights[i, 0] = torch.tensor([random.random() * -1 if random.random() < 0.5 else 1 for _ in range(self.__embedding_size)], dtype=torch.float)
            self.__positional_weights_t = weights.to(self.__device)
            
        return self.__positional_weights_t'''
        
        # from https://github.com/kingoflolz/mesh-transformer-jax/blob/master/mesh_transformer/layers.py#L128-L134
        # don't ask me exactly how this code works
        # or why encoding positions through random frequency sine waves works for the model either
        
        # create random scalars to be variable frequencies for our sinusoids
        inverse_freq = 1. / (10000 ** (np.arange(0, self.__embedding_size) / self.__embedding_size))
        
        # TODO(Sean) cache when batch sizes are normalized (all batch sizes are the same)
        # as_tensor moves data from numpy array instead of copying
        sinusoid_input_t: torch.Tensor = torch.as_tensor(np.einsum("i, j -> i j", np.arange(input.shape[seq_dim]), inverse_freq), dtype=torch.float, device=self.__device)
        # there should be a way to do the encoding without this line...
        sinusoid_input_t: torch.Tensor = sinusoid_input_t.unsqueeze(0).expand(input.shape[0], -1, -1)
        # pass input.matmul(weights) through sin, voila, positions encoded as random sinusoids!
        return torch.sin(sinusoid_input_t)
        """batch_size = input.shape[0]
        weights = torch.empty(size=(batch_size, self.__query_length, self.__embedding_size), dtype=torch.float, device=self.__device)
        if self.__cached_noise is None:
            self.__cached_noise: torch.Tensor = torch.tensor([random.random() * -1 if random.random() < 0.5 else 1 for _ in range(self.__embedding_size)], dtype=torch.float, device=self.__device)
        weights[:, :] = self.__cached_noise
        
        x = torch.empty(size=(batch_size, self.__query_length, self.__embedding_size), dtype=torch.float, device=self.__device)
        if input.shape[2] == 1:
            x = input.transpose(1, 2).expand(-1, self.__query_length, -1)
        else:
            x = input
        x = x.bmm(weights)
        return torch.sin(x)"""
    
    def decode(
        self, 
        input: torch.Tensor, 
        batch_size: int, 
        decode_tokens: bool = False, 
        topk: typing.Union[int, None] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # TODO(Sean) topk decoding not implemented
        
        # forward pass
        response_state_t = self.forward(input,batch_size=batch_size)
        
        # transpose response from forward call (batch_size x query size x tokens size) to (batch_size x query size x 1)
        # empty tensor is unitialized memory
        transposed_response_tokens_t: torch.Tensor = torch.empty((batch_size, self.__query_length, self.__tokens_size), dtype=torch.float)
        # only decode when asked, saves on LOTS of memory (double, to be exact)
        if decode_tokens:
            # TODO(Sean) implement topk
            _, topi = response_state_t.topk(1)
            transposed_response_tokens_t = topi
        
        return response_state_t, transposed_response_tokens_t
    
    def decode_generative(
        self, 
        input: torch.Tensor, 
        batch_size: int, 
        decode_tokens: bool = False, 
        topk: typing.Union[int, None] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # TODO(Sean) topk decoding not implemented
        
        # forward pass
        response_state_t = self.autoregressive_forward(input,batch_size=batch_size)
        
        # transpose response from forward call (batch_size x query size x tokens size) to (batch_size x query size x 1)
        # empty tensor is unitialized memory
        transposed_response_tokens_t: torch.Tensor = torch.empty((batch_size, self.__query_length, self.__tokens_size), dtype=torch.float)
        # only decode when asked, saves on LOTS of memory (double, to be exact)
        if decode_tokens:
            # TODO(Sean) implement topk
            _, topi = response_state_t.topk(1)
            transposed_response_tokens_t = topi
        
        return response_state_t, transposed_response_tokens_t
    
    
    def predict(self, 
                input_query: typing.Union[typing.List[typing.List[int]], torch.Tensor], 
                max_length: int, 
                replace_toks_after_eol_with_padding: bool = True
        ) -> torch.Tensor:
        """compatibility function (same api as RNN model) to predict a sentence using the jes-1 model"""
        with torch.no_grad():
            batch_size: int = 1
            if type(input_query) is torch.Tensor:
                batch_size = input_query.shape[0]
            
            input_t: torch.Tensor = torch.empty(batch_size, self.__query_length, self.__tokens_size) 
            if type(input_query) is not torch.Tensor:
                input_t = get_X_train(input_query, self.__query_length)[0]
            else:
                input_t = input_query
            
            
            _, response = self.decode(
                input=input_t.to(self.__device),
                batch_size=batch_size, 
                decode_tokens=True, 
                topk=1
            )
            
            # NOTE(Sean) i think this is an actual problem with the model that should be solved, rather than skirted around...
            #            probably something to do with the masked loss
            # replace everything after first <EOL> with padding
            if replace_toks_after_eol_with_padding:
                for res in response:
                    eol_index: int = 0
                    for i, tok in enumerate(res):
                        eol_index = i
                        if tok == 2: # probably should be hard coded to 2...
                            break
                    
                    res[eol_index+1:] = 0
        
        return response
        
    def forward(
        self, 
        input: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        
        # =========
        # embedding
        # =========
        
        # pass batched inputs through embedding layer
        embedding: torch.Tensor = self._embedding(input)
        # reshape input to be (batch size x query length x 1)
        input_reshaped: torch.Tensor = input.unsqueeze(2).type(torch.float)
        # pass the input indices through sine waves with varied frequencies to 'encode' the positions
        positional_encodings: torch.Tensor = self.__get_positional_sine_waves(input_reshaped)
        
        # =======
        # encoder
        # =======
        x: torch.Tensor = embedding
        kv_tensors: typing.List[typing.Tuple[torch.Tensor, torch.Tensor]] = []
        for encoder in self._encoder_layers:
            hidden, qk_tup = encoder(embedding, positional_encodings)
            x = x + hidden
            kv_tensors.append(qk_tup)
        x = self._post_encoder_layer_norm(x)
        
        # =======
        # decoder
        # =======
        decoder_input_t: torch.Tensor = torch.zeros((batch_size, self.__query_length, self.__embedding_size), dtype=torch.float, device=self.__device)
        decoder_output_t: torch.Tensor = torch.zeros((batch_size, self.__query_length, self.__embedding_size), dtype=torch.float, device=self.__device)
        decoder_positional_encodings: torch.Tensor = self.__get_positional_sine_waves(decoder_input_t)
        for decoder in self._decoder_layers:
            decoder_output_t = decoder_output_t + decoder(decoder_input_t, decoder_positional_encodings, kv_tensors)
            
            
        # normalize final output
        final_decoder_output_t = self._post_decoder_layer_norm(decoder_output_t)
        
        # =======
        
        # pass through a linear layer (classifier) and softmax to get decoder probabilities
        # TODO(Sean) gpt3 paper says to use inverse embedding, but how can we inverse a non-square matrix?
        decoded: torch.Tensor = self._softmax(self._decoder_projection(final_decoder_output_t), dim=2)
        
        # decode by passing through left inverse matrix of the embedding weights
        # NOTE(Sean) model does not seem to work with the left inverse of embedding weights...
        '''embedding_weights_tranpose_t: torch.Tensor = self._embedding.weight.T
        inverse_embedding_t: torch.Tensor = (embedding_weights_tranpose_t.mm(self._embedding.weight)).inverse().mm(embedding_weights_tranpose_t).unsqueeze(0).expand(batch_size, -1, -1)
        decoded: torch.Tensor = self._softmax(final_decoder_output_t.bmm(inverse_embedding_t), dim=2)'''
    
        return decoded
    
    def autoregressive_forward(
        self, 
        input: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        
        # =========
        # embedding
        # =========
        
        # pass batched inputs through embedding layer
        embedding: torch.Tensor = self._embedding(input)
        # reshape input to be (batch size x query length x 1)
        input_reshaped: torch.Tensor = input.unsqueeze(2).type(torch.float)
        # pass the input indices through sine waves with varied frequencies to 'encode' the positions
        positional_encodings: torch.Tensor = self.__get_positional_sine_waves(input_reshaped)
        
        # =======
        # encoder
        # =======
        x: torch.Tensor = embedding
        kv_tensors: typing.List[typing.Tuple[torch.Tensor, torch.Tensor]] = []
        for encoder in self._encoder_layers:
            hidden, qk_tup = encoder(embedding, positional_encodings)
            x = x + hidden
            kv_tensors.append(qk_tup)
        x = self._post_encoder_layer_norm(x)
        
        # =======
        # decoder
        # =======
        decoder_input_t: torch.Tensor = torch.zeros((batch_size, self.__query_length, self.__embedding_size), dtype=torch.float, device=self.__device)
        decoder_output_t: torch.Tensor = torch.zeros((batch_size, self.__query_length, self.__embedding_size), dtype=torch.float, device=self.__device)
        decoder_positional_encodings: torch.Tensor = self.__get_positional_sine_waves(decoder_input_t)
        for i in range(self.__query_length):
            response_state_t = torch.zeros((batch_size, self.__query_length, self.__embedding_size), dtype=torch.float, device=self.__device)
            for decoder in self._decoder_layers:
                response_state_t = response_state_t + decoder(decoder_input_t, decoder_positional_encodings, kv_tensors)
            response_state_t = self._post_decoder_layer_norm(response_state_t)
            
            decoder_output_t[:,i,:] = response_state_t[:,0,:]
            decoder_input_t[:,i,:] = response_state_t[:,0,:]
            
            decoder_positional_encodings: torch.Tensor = self.__get_positional_sine_waves(decoder_input_t)
            
        # normalize final output
        final_decoder_output_t = self._post_decoder_layer_norm(decoder_output_t)
        
        # =======
        
        # pass through a linear layer (classifier) and softmax to get decoder probabilities
        # TODO(Sean) gpt3 paper says to use inverse embedding, but how can we inverse a non-square matrix?
        decoded: torch.Tensor = self._softmax(self._decoder_projection(final_decoder_output_t), dim=2)
        
        # decode by passing through left inverse matrix of the embedding weights
        # NOTE(Sean) model does not seem to work with the left inverse of embedding weights...
        # NOTE(Sean) using left inverse uses a lot of extra memory...
        '''embedding_weights_tranpose_t: torch.Tensor = self._embedding.weight.T
        inverse_embedding_t: torch.Tensor = (embedding_weights_tranpose_t.mm(self._embedding.weight)).inverse().mm(embedding_weights_tranpose_t).unsqueeze(0).expand(batch_size, -1, -1)
        decoded: torch.Tensor = self._softmax(x.bmm(inverse_embedding_t), dim=2)'''
        
        return decoded
    
    def train_one_iteration(self, X_train, Y_train, mask, batch_size):
        # zero gradients
        self._optimizer.zero_grad()
        #for param in self.__get_all_learnable_parameters():
        #    param.grad = None
        
        # ==============================================================
        # send inputs to correct device (batching is always done on cpu)
        # ==============================================================
        if self.__device == torch.device("cuda"):
            X_train = X_train.to(device=self.__device)
            Y_train = Y_train.to(device=self.__device)
            mask = mask.to(device=self.__device)
        # ==============================================================
        
        # =================================
        # TODO(Sean) profiling code, remove
        '''
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
            with record_function("model_inference"):
                state, _ = self.decode(X_train, batch_size)
        print(prof.key_averages().table(sort_by="cpu_time_total"))'''
            
        
        # decode an output from the network
        state, _ = self.decode(input=X_train, batch_size=batch_size)
    
        # compute loss
        loss, n_total = maskNLLLoss(state, Y_train, mask)
        
        # backwards pass to update gradients
        loss.backward()
        
        # clip weights to avoid vanishing gradient problem
        _ = nn.utils.clip_grad_norm_(self.__get_all_learnable_parameters(), 50)
        
        # optimizer step
        self._optimizer.step()

        return loss.item()
    
    def train(self, pairs, max_epochs: int, batch_size: int, dataset, epsilon: typing.Union[float, None] = 0.01, save_every_iteration: typing.Union[int, None] = None):
        """epsilon is the minimum tolerance for a change in loss. training will stop if delta loss < epsilon. set epsilon to None for no early stopping"""
        logger.log_info("training...")
        print_every = 1
        total_loss = 0
        prev_loss = 0
        delta_loss = 0
        # should we compute batches once, before training?
        batches = [pairs[i:i+batch_size] for i in range(0,len(pairs),batch_size)]
        t = time.time()
        # how often checkpoints are created
        save_every: int = save_every_iteration if save_every_iteration is not None else max_epochs // 10
        logger.log_info(f"saving model every {save_every} epochs.")
        # get starting iteration (either 1 or saved with checkpoint)
        start_iteration: int = 1 if self.__loaded_model_start_iteration is None else self.__loaded_model_start_iteration + 1
        losses = []
        for epoch in range(start_iteration, max_epochs+1):
            #mini batch training
            batch_loss = 0
            for batch in batches:
                input_v,_,target_v,mask,_ = data_to_sequence(batch, self.__query_length)
                input_batch_size = input_v.shape[0]
                
                # profiling code, remove...
                '''with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
                    with record_function("model_inference"):
                        loss = self.train_one_iteration(input_v,target_v,mask,input_batch_size)
                print(prof.key_averages().table(sort_by="cpu_time_total"))'''
                loss = self.train_one_iteration(input_v,target_v,mask,input_batch_size)
                losses.append(loss)
                total_loss += loss
                batch_loss += loss
            
            #logger.log_info(f"  batch_loss={batch_loss}") # TODO(Sean) remove
            delta_loss = abs(batch_loss - prev_loss)
            prev_loss = batch_loss

            #print average every print_every iteration
            if epoch % print_every == 0:
                logger.log_info(f"epoch {epoch}/{max_epochs} average loss: {total_loss/print_every} percentage {100*epoch/max_epochs:.2f}%, took {time.time() - t} seconds")
                logger.log_info(f"  delta_loss={delta_loss}, prev_loss={prev_loss}") # TODO(Sean) remove
                total_loss = 0
                t = time.time()
            
            # save model every save_every epochs
            saved: bool = False
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, dataset)
                saved = True
                
            if epsilon is not None and delta_loss < epsilon:
                logger.log_warn(f"delta_loss={delta_loss} crossed epsilon={epsilon} threshold! Stopping training early")
                if not saved:
                    self.save_checkpoint(epoch, dataset)
                break
        
        logger.log_info("done training")
        
        if len(losses) > 0:
            plt.plot(losses)
            plt.xlabel("Iterations")
            plt.ylabel("Loss value")
            plt.show()

def batched_evaluate(model,test_pairs,token_list,batch_size):
    logger.log_info("starting to evaluate model...")
    X_train,Y_train = [pair[0] for pair in test_pairs], [pair[1] for pair in test_pairs]
    #sentence as int for now
    predicted = []
    batches = [test_pairs[i:i+batch_size] for i in range(0,len(test_pairs),batch_size)]
    for batch in batches:
        input_v,input_l,target_v,mask,target_ml = data_to_sequence(batch, MAX_LENGTH)
        result = model.predict(input_v.to(model._JESModel__device), len(batch), replace_toks_after_eol_with_padding=True)
        
        for i in range(result.shape[0]):
            predicted.append(result[i])
    #predicted = [model.predict([x],max_word) for x in X_train]
    predicted = [output_to_sentence_str(p,token_list=token_list) for p in predicted]
    Y_train = [output_to_sentence_str(y,token_list=token_list) for y in Y_train]
    logger.log_info("finished evaluating model")
    logger.log_info(f"  score: {main.bleu(Y_train,predicted)}")

def main():
    logger.log_info("Testing JES-1...")
    
    data_directory = "../data/"
    raw_data_directory = "raw/"
    dataset_name_with_extension = "combined.txt"
    dataset_name = os.path.splitext(os.path.basename(dataset_name_with_extension))[0] # name of the dataset without extension
    dataset_path = os.path.join(data_directory, dataset_name_with_extension)
    if not os.path.isfile(dataset_path):
        assert False, "file not found"
    t = time.time()
    sentences_as_tokens = []
    cornell_corpus = get_cornell_movie_lines(dataset_path,-1)
    for pair in cornell_corpus:
        sentences_as_tokens.append(simple_token(pair[0]))
        sentences_as_tokens.append(simple_token(pair[1]))
    cornell_corpus.clear()
     
    # TODO(Sean) debugging line, remove
    #torch.autograd.set_detect_anomaly(True)
    
    
    # DEVICE = torch.device("cpu") # force train on cpu
    
    # sentences_as_tokens = tokenize(dataset_path, force_compute=True)
        
    dataset = data(sentences_as_tokens, MAX_LENGTH)

    n = dataset.get_num_sentence()
    logger.log_info(f"took {time.time() - t} seconds to read {n} lines from '{dataset_name}'")
    word_dict = dataset.get_word_dict()
    token_list = dataset.get_token_list()
    sentence_as_int = dataset.get_sentence(n) #get the first 10 sentence from the dataset
    number_of_tokens = dataset.get_num_word()
    logger.log_info(f"'{dataset_name}' has {number_of_tokens} tokens")

    Y = sentence_as_int[1::2]  #get a list of sentence_as_int
    X = sentence_as_int[0::2]  #get a list of sentence_as_int
    X = X[:len(Y)] # uneven amount of data, so this evens it out

    #pair them together, have 10 sentence, so 5 pairs
    pairs = []
    for i in range(n//2):
        pairs.append([X[i],Y[i]])

    numbers_of_pairs = len(pairs)
    #split the pairs into training and testing  60% training, 20% evaluate, 20% testing
    train_pairs, evaluate_pairs, test_pairs = pairs[:int(numbers_of_pairs*0.6)],pairs[int(numbers_of_pairs*0.6):int(numbers_of_pairs*0.8)],pairs[int(numbers_of_pairs*0.8):]
    
    model = JESModel(
        tokens_size=number_of_tokens,
        query_len=MAX_LENGTH,
        embedding_size=EMBEDDING_SIZE, 
        attention_size=ATTENTION_SIZE,
        encoder_layer_count=1,
        decoder_layer_count=DECODER_LAYER_COUNT,
        device=DEVICE,
        model_name=MODEL_NAME,
        dataset_name=dataset_name,
        data_directory_path=data_directory,
        learning_rate=LEARNING_RATE,
        verbose_log=VERBOSE_LOG
    ).load_checkpoint(dataset=dataset, skip_loading_model=SKIP_LOADING_MODEL)
    #model = torch.compile(model)
    
    model.train(pairs=train_pairs, max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, dataset=dataset, epsilon=EARLY_STOPPING_THRESHOLD)
    
    #evaluting using BLEU, return a float value from 0-1, the higher the better
    if EVALUATE_MODEL:
        with torch.no_grad():
            logger.log_info("Training BLEU: ")
            batched_evaluate(model,train_pairs,token_list, batch_size=BATCH_SIZE)
            logger.log_info("Evaluation BLEU: ")
            batched_evaluate(model,evaluate_pairs,token_list, batch_size=BATCH_SIZE)
            logger.log_info("Testing BLEU")
            batched_evaluate(model,test_pairs,token_list, batch_size=BATCH_SIZE)
    
    batches = [pairs[i:i+BATCH_SIZE] for i in range(0,len(pairs),BATCH_SIZE)]
    print_n_batches = 10
    print_first_n = 10
    with torch.no_grad():
        for i in range(min(print_first_n, len(batches))):
            batch = batches[i]
            input_v,input_l,target_v,mask,target_ml = data_to_sequence(batch, MAX_LENGTH)
            target_v = target_v
            out = model.predict(input_v.to(DEVICE), len(batch), replace_toks_after_eol_with_padding=True)
            for i in range(min(out.shape[0], print_first_n)):
                logger.log_info(f"Query: {output_to_sentence_str(output=input_v[i], token_list=token_list, include_padding=False)}")
                logger.log_info(f"  Predict: {output_to_sentence_str(output=out[i], token_list=token_list, include_padding=False)}")
                logger.log_info(f"  True Y: {output_to_sentence_str(output=target_v[i], token_list=token_list, include_padding=False)}")
            print()
        
        #real time prediction
        logger.log_info("Lets do some real predictions, type 'quit' to end")
        while True:
            query = input(">")
            # tokenize input query
            tokenized_query = tokenize_sentence(query.strip(), strip_tokens=[], token_list=token_list, verbose_logging=False)
            
            if(len(tokenized_query) == 0 or tokenized_query[0] == "quit"):
                break
            
            # convert tokenized sentence to indices
            index_query = tokenized_sentence_to_indices(tokenized_sentence=tokenized_query, query_size=MAX_LENGTH, word_dict=word_dict, add_eos_token=True)
            batched_index_query = torch.zeros(size=(1, MAX_LENGTH), dtype=torch.int, device=DEVICE)
            batched_index_query[0] = torch.tensor(index_query, dtype=torch.int, device=DEVICE)
            
            result = model.predict(batched_index_query, 1, replace_toks_after_eol_with_padding=True)
            
            print(f"[JES-1]: {output_to_sentence_str(output=result[0], token_list=token_list, include_padding=False)}")

if __name__ == "__main__":
    main()
    
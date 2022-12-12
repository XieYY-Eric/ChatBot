from logger import Logger

import typing
import os
import nltk
import pickle
import time
import matplotlib.pyplot as plt
import sys

logger = Logger()

TOKENS_CACHED_FILENAME = "tokens.cached.pickle"

class Tokens:
    SOS = "<SOS>"
    EOS = "<EOL>"
    UNKNOWN_TOKEN = "<UNKNOWN>"
    PAD = "<PAD>"
    
    @staticmethod
    def get_all_utility_tokens() -> typing.List[str]:
        """Returns a list of all custom utility tokens."""
        # NOTE(Sean) add all tokens above to list
        return [Tokens.PAD,Tokens.SOS, Tokens.EOS, Tokens.UNKNOWN_TOKEN]

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

def tokenized_sentence_to_indices(tokenized_sentence: typing.List[str], query_size: int, word_dict, add_eos_token: bool = False) -> typing.List[int]:
    """Takes a tokenized sentence and returns a list of indices for each of the tokens."""
    sentence = [word_dict[token] if token in word_dict else word_dict[Tokens.UNKNOWN_TOKEN] for token in tokenized_sentence]
    # add <EOS> token
    sentence += [word_dict[Tokens.EOS]] if add_eos_token else []
    # add padding
    sentence += [0]*(query_size - len(sentence))
    return sentence

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


class Tokenizer:
    def __init__(self) -> None:
        self.tokens_list = [Tokens.PAD, Tokens.SOS, Tokens.EOS, Tokens.UNKNOWN_TOKEN]
        self.unique_tokens = set(self.tokens_list)
        self.word_dict = {
            Tokens.PAD: 0,
            Tokens.SOS: 1,
            Tokens.EOS: 2, 
            Tokens.UNKNOWN_TOKEN: 3
        }
        self.num_of_word = len(self.tokens_list)
        self.frequency = {}

    def normalize(self,sentence:[str]):
        """
        lower sentence, remove punctuation
        """
        strip_tokens = ['\'', ".", "?", ",", "--"]
        return [token.strip().lower() for token in sentence if token not in strip_tokens]

    def tokenize(self, dataset_path: str):
        """
        Tokenize the entire file using nltk.
        Return nothing, set internal word_dict and tokens_list
        """        
        # read dataset and store lines
        logger.log_info(f"starting to tokenizing from file '{dataset_path}'")
        start_time = time.time()
        with open(dataset_path, "r",encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                tokens = nltk.word_tokenize(line)
                tokens = self.normalize(tokens)
                for token in tokens:
                    if token not in self.frequency:
                        self.frequency[token] = 0
                    self.frequency[token] += 1
                tokens =[token for token in tokens if token not in self.unique_tokens]
                self.tokens_list.extend(tokens)
                self.unique_tokens.update(tokens)
                for token in tokens:
                    self.word_dict[token] = self.num_of_word
                    self.num_of_word += 1
        logger.log_info(f"finished tokenizing from '{dataset_path}'\n\tnumber of tokens {self.num_of_word}\n\ttook {time.time() - start_time:.2f} seconds")

    def tokenize_sentence(self,sentence):
        """
        add additional tokens from this sentence
        """
        sentence = self.normalize(sentence)
        tokens = nltk.word_tokenize(sentence)
        freq_dist = nltk.FreqDist(tokens)
        tokens =[token for token in freq_dist.keys() if token not in self.unique_tokens]
        tokens = self.normalize(tokens)
        self.tokens_list.extend(tokens)
        for token in tokens:
            self.word_dict[token] = self.num_of_word
            self.num_of_word += 1

    def plotDistribution(self,topv=10):
        most_common = [(k,v) for k,v in sorted(self.frequency.items(),key = lambda a:a[1],reverse=True)[:topv]]
        total_word = sum([v for k,v in self.frequency.items()])
        fig, ax = plt.subplots()
        names,values = zip(*most_common)
        names = list(names)
        values = [v*100/total_word for v in list(values)]
        ax.bar(names,values)
        plt.title('percentage of token')
        plt.xlabel('tokens')
        plt.ylabel('percentage')
        plt.show()

    def encode(self, sentence):
        """
        convert a list of raw string, normalize them, convert them to list of index, no padding
        sentence : 'list of str or str'
        """
        # if sentence is a str, split it into a list first
        if type(sentence) == str:
            sentence = nltk.word_tokenize(sentence)
        # if it is a list of str, perfect
        sentence = self.normalize(sentence)
        return [self.word_dict.get(word,self.word_dict[Tokens.UNKNOWN_TOKEN]) for word in sentence]

    def decode(self, sentence_as_int:[str]):
        """
        convert a list of index, normalize them, convert them into list of tokens, no padding
        """
        return [self.tokens_list[index] for index in sentence_as_int]

    def save(self,filename):
        logger.log_info(f"Saving current tokens to {filename}")
        datas = {"token_list":self.tokens_list,"unique_tokens":self.unique_tokens, "word_dict":self.word_dict,"number_word":self.num_of_word, "freq":self.frequency}
        with open(filename,"wb") as f:
            pickle.dump(datas,f)

    def load(self,filename):
        logger.log_info(f"Loading datas from {filename}...")
        with open(filename,"rb") as f:
            datas = pickle.load(f)
            self.tokens_list = datas["token_list"]
            self.unique_tokens = datas["unique_tokens"]
            self.word_dict = datas["word_dict"]
            self.num_of_word = datas["number_word"]
            self.frequency = datas["freq"]
        logger.log_info(f"Loading complete")

    def __str__(self):
        return f"-----Tokenizer: num_of_token:{self.num_of_word}"

    def __len__(self):
        return self.num_of_word   

    
if __name__ == "__main__":
    # simple demo for demonstration

    filename = "../data/reddit_trim.txt"
    save_filename = "../data/cache/Pre_computed_breaking_bad.pickle"

    if len(sys.argv) == 3:
        filename = "../data/" +sys.argv[1]
        save_filename = "../data/" + sys.argv[2] + ".pickle"


    oldTokenizer = Tokenizer()
    oldTokenizer.tokenize(filename)
    print(oldTokenizer)
    ######save the internal data of this tokenize
    oldTokenizer.save(save_filename)

    ###### and load it back load the data
    newTokenizer = Tokenizer()
    newTokenizer.load(save_filename)
    print(newTokenizer)
    newTokenizer.plotDistribution(10)

    for i in range(5):
        result = newTokenizer.encode(input("enter something to encode:").split())
        print(f"Encoded message {result}")
        print(f"The decoded message is: {newTokenizer.decode(result)}")

    
from __future__ import unicode_literals, print_function, division
import time
import os
import tokenizer
import data
import torch
# import models
import torch.nn as nn
from scraper import scrape_transcripts_from_website
from aggregator import aggregate_raw_transcripts
# from gtp3Model import gtp3_model_setup
from logger import Logger
import json
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
import typing
from pickle import NONE
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from data import Data, DataSet
from logger import Logger
# from src.scraper import scrape_transcripts_from_website
# from src.aggregator import aggregate_raw_transcripts
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
import typing
import sys

# ==================
MAX_LENGTH = 12
HIDDEN_SIZE = 80
LEARNING_RATE = 0.001
ITERATION = 500
BATCH_SIZE = 64
CONTINUE_CHECKPOINT = False #if we want to continue from last checkpoint (LOAD_MODEL_CHECKPOINT NEED TO BE TRUE ASWELL)
SAVE_MODEL_CHECKPOINTS = False
LOAD_MODEL_CHECKPOINT = True
N_LAYERS = 1
MODEL_NAME = "BreakingBot"
EVALUATE_MODEL = False
use_non_greedy_decoding: bool = False
jpt3 = False
# ==================




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = Logger()
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths).to(device)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
    
    def load(self, checkpoint) -> nn.Module:
        if not checkpoint or not checkpoint.get("encoder", None):
            logger.log_warn("  encoder failed to load parameters!")
            return self
        state_dict = checkpoint["encoder"]
        self.load_state_dict(state_dict)
        logger.log_info(f"  encoder loaded parameters")
        return self

#Attention used by Decoder    
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
    
    def load(self, checkpoint):
        if not checkpoint or not checkpoint.get("attention", None):
            logger.log_warn("  attention failed to load parameters!")
            return self
        state_dict = checkpoint["attention"]
        self.load_state_dict(state_dict)
        logger.log_info(f"  attention loaded parameters")
        return self
    
class DecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn("general", hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
    
    def load(self, checkpoint) -> nn.Module:
        if not checkpoint or not checkpoint.get("decoder", None):
            logger.log_warn("  decoder failed to load parameters!")
            return self
        state_dict = checkpoint["decoder"]
        self.load_state_dict(state_dict)
        self.attn.load(checkpoint)
        logger.log_info(f"  decoder loaded parameters")
        return self


##### Helper function used by model internally
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss.to(device=device)
    return loss, nTotal.item()

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


def get_X_train(X, add_eos_token: bool = False):
    length = torch.tensor([len(sentence) for sentence in X], device=torch.device("cpu"))
    for i,s in enumerate(X):
        if add_eos_token:
            s.append(2)
        s.extend([0]*(length.max()-length[i]))
    return torch.tensor(X, device=torch.device("cpu")).T, length

def get_Y_train(Y):
    max_Y_length = max([len(sentence) for sentence in Y]) + 1
    for i,s in enumerate(Y):
        s.append(2)
        s.extend([0]*(max_Y_length-len(s)))
    Y = torch.tensor(Y, device=torch.device("cpu")).T
    mask = binaryMatrix(Y)
    return Y,mask,max_Y_length

def data_to_sequence(pairs):
    pairs.sort(key=lambda pair:len(pair[0]),reverse=True)
    length = torch.tensor([len(pair[0]) for pair in pairs], device=torch.device("cpu"))
    X , Y = [] , []
    for pair in pairs:
        X.append(pair[0][:])
        Y.append(pair[1][:])
    X,length = get_X_train(X)
    Y,mask,max_Y_length = get_Y_train(Y)
    return X,length,Y,mask,max_Y_length



#####
class Dummy3:
    def __init__(self,hidden_size,learning_rate,encoder,decoder, embedding, tokenizer, save_checkpoints: bool, load_checkpoint: bool, cont_checkpoint: bool) -> None:
        self.learning_rate = learning_rate
        self.embedding = embedding
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_size = hidden_size
        self.Eoptimizer = torch.optim.Adam(self.encoder.parameters(),lr=self.learning_rate) 
        self.Doptimizer = torch.optim.Adam(self.decoder.parameters(),lr=self.learning_rate*5)
        
        # flag to control whether we save checkpoints
        self._save_checkpoints = save_checkpoints
        # flag to control whether we load a checkpoint
        self._load_checkpoint = load_checkpoint
        # flag to contril whether we want to contuiue from last checkpoint
        self._cont_load_checkpoint = cont_checkpoint
        # dictionary of saved parameters
        self.checkpoint = None
        
        # print number of learnable parameters
        learnable_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        logger.log_info(f"learnable parameters: {sum(p.numel() for p in learnable_params if p.requires_grad)}")
        
    
    def send_optimizers_to_device(self):
        """send optimizer states to gpu"""
        if device == torch.device("cuda"):
            # from https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
            for state in self.Eoptimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            for state in self.Doptimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
                        
        return self

    def compute_loss(self,size,X_train,X_length,Y_train,mask,target_ml):
        loss = 0
        n_totals = 0
        X_train = X_train.to(device=device)
        Y_train = Y_train.to(device=device)
        mask = mask.to(device=device)
        X_length = X_length.to(device=torch.device("cpu"))
        encoder_output,encoder_hidden = self.encoder(X_train,X_length)
        #decoding
        decoder_input = torch.tensor([[1 for _ in range(size)]], device=device)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        for i in range(target_ml):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input,decoder_hidden,encoder_output)
            # print("training",decoder_output.shape,decoder_hidden.shape)
            topv,topi = decoder_output.topk(1)
            decoder_input = torch.tensor([[topi[i][0] for i in range(size)]], device=device)
            mask_loss,nTotal = maskNLLLoss(decoder_output,Y_train[i],mask[i])
            loss += mask_loss.item()*nTotal
            n_totals += nTotal

        return loss / n_totals
        
    def train_one_iteration(self,X_train,X_length,Y_train,Y_length,mask,batch_size):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        loss = 0
        print_losses = []
        n_totals = 0

        X_train = X_train.to(device=device)
        Y_train = Y_train.to(device=device)
        mask = mask.to(device=device)
        X_length = X_length.to(device=torch.device("cpu"))
        #encoding process
        encoder_output,encoder_hidden = self.encoder(X_train,X_length)

        #decoding
        decoder_input = torch.tensor([[1 for _ in range(batch_size)]], device=device)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        for i in range(Y_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input,decoder_hidden,encoder_output)
            topv,topi = decoder_output.topk(1)
            decoder_input = torch.tensor([[topi[i][0] for i in range(batch_size)]], device=device)
            mask_loss,nTotal = maskNLLLoss(decoder_output,Y_train[i],mask[i])
            loss += mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals += nTotal

        
        loss.backward()

        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), 50)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.Eoptimizer.step()
        self.Doptimizer.step()

        return (sum(print_losses) / n_totals)
    
    def load(self, checkpoint, force: bool = False):
        
        if not self._load_checkpoint and not force:
            logger.log_warn("skipped loading model parameters")
            self.encoder.to(device)
            self.decoder.to(device)
            return self
        
        # make sure we have data
        if not checkpoint:
            logger.log_error("tried loading checkpoint data but it was null!")
            self.encoder.to(device)
            self.decoder.to(device)
            return self
        
        self.encoder.load(checkpoint)
        self.decoder.load(checkpoint)
        if not checkpoint or not checkpoint.get("encoder_optimizer", None):
            logger.log_warn("  Eoptimizer failed to load parameters!")
        else:
            self.Eoptimizer.load_state_dict(checkpoint["encoder_optimizer"])
            logger.log_info("  Eoptimizer loaded parameters")
            
        if not checkpoint or not checkpoint.get("decoder_optimizer", None):
            logger.log_warn("  Doptimizer failed to load parameters!")
        else:
            self.Doptimizer.load_state_dict(checkpoint["decoder_optimizer"])
            logger.log_info("  Doptimizer loaded parameters")
        
        if not checkpoint or not checkpoint.get("tokenizer", None):
            logger.log_warn("  tokenizer failed to load parameters!")
        else:
            self.tokenizer = checkpoint["tokenizer"]
            logger.log_info("  tokenizer loaded")

        self.checkpoint = checkpoint
        
        self.encoder.to(device)
        self.decoder.to(device)
        
        return self

    def save_model(self, data_directory: str, model_name: str, dataset_name: str, iteration: int, **kwargs) -> None:
        """
        Save the model's parameters to a file
        Pass in optional kwargs to save more parameters to the model
        """
        # compute save directory path
        parameters_str: str = f"{self.encoder.n_layers}-{self.decoder.n_layers}-{self.encoder.hidden_size}"
        dataset_name_stripped = os.path.splitext(dataset_name)[0] # assumed filepath in form: "foo.bar"
        dirpath = os.path.join(data_directory, model_name, dataset_name_stripped, parameters_str)

        # create directory if it does not exist
        if not os.path.exists(dirpath):
            logger.log_warn(f"'{dirpath}' does not exists, creating directories...")
            os.makedirs(dirpath)
        
        # dictionary of parameters to save
        # TODO(Sean) save token list
        params_dict = {
            "iteration": iteration,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "encoder_optimizer": self.Eoptimizer.state_dict(),
            "decoder_optimizer": self.Doptimizer.state_dict(),
            "embedding": self.embedding.state_dict(),
            "attention": self.decoder.attn.state_dict(),
            "tokenizer": self.tokenizer
        }
        # add kwargs to the save dict
        params_dict.update(kwargs)
        
        checkpoint_filepath = os.path.join(dirpath, f"{iteration}-checkpoint.tar")
        logger.log_info(f"  saving model to '{checkpoint_filepath}'")
        for key in params_dict.keys():
            logger.log_info(f"    saved {key}")
        torch.save(params_dict, checkpoint_filepath)
    
    @staticmethod
    def load_checkpoint(data_directory: str, model_name: str, dataset_name: str, iteration: int, n_layers: int, hidden_size: int, cont: bool = False):
        # compute saved file directory
        parameters_str: str = f"{n_layers}-{n_layers}-{hidden_size}"
        dataset_name_stripped = os.path.splitext(dataset_name)[0] # assumed filepath in form: "foo.bar"
        dirpath = os.path.join(data_directory, model_name, dataset_name_stripped, parameters_str)
        
        if not os.path.isdir(dirpath):
            logger.log_error(f"tried loading checkpoint(s) from '{dirpath}', but the directory does not exist! Aborting!")
            return None


        if cont: # if we want to continue from where we left off
            checkpoint = 0
            for path in os.listdir(dirpath): # count the ammount of files in directory to get the last checkpoint
                if os.path.isfile(os.path.join(dirpath, path)):
                    checkpoint = max(int(os.path.basename(path).split('-')[0]), checkpoint) # assumes filename in format "{iteration}-checkpoint.tar"
            
            logger.log_info(f"latest checkpoint found = {checkpoint}")
            checkpoint_filepath = os.path.join(dirpath, f"{checkpoint}-checkpoint.tar") # get the last checkpoints file path
        else:
            checkpoint_filepath = os.path.join(dirpath, f"{iteration}-checkpoint.tar") # otherwise just get the one checkpoint to load
        
        if not os.path.isfile(checkpoint_filepath):
            logger.log_warn(f"trying to load checkpoint '{checkpoint_filepath}' but file does not exist!")
            raise FileNotFoundError
        
        # load file
        logger.log_info(f"loading model '{checkpoint_filepath}'")
        checkpoint = None
        # if device == torch.device("cuda"):
        checkpoint = torch.load(checkpoint_filepath, map_location=torch.device("cpu"))
        # else:
        #     checkpoint = torch.load(checkpoint_filepath)
        # TODO(Sean) load token list and compare against current one. if they differ, force recomputing of model...
        return checkpoint

    def train(self, dataset, evaluation_pairs,iteration, batch_size, data_directory: str, dataset_name: str, model_name: str = "Dummy3", epsilon: float = 0.000001):
        """epsilon is the minimum tolerance for a change in loss. training will stop if delta loss < epsilon. set epsilon to None for no early stopping"""
        logger.log_info("training...")
        training_loss_over_iteration = []
        testing_loss_over_iteration = []
        checkpoint = 10
        print_every = 1
        total_loss = 0
        prev_loss = 0
        delta_loss = 0
        #testing
        evX_train,evX_length,evY_train,evmask,evtarget_ml = data_to_sequence(evaluation_pairs)
        # start from specific iteration if we are loading a model
        start_iteration = 1 if not self._load_checkpoint or self.checkpoint is None else self.checkpoint["iteration"] + 1
        
        t = time.time()
        for iter in range(start_iteration,iteration+1):
            #mini batch training
            #load data from dataset
            training_iteration_loss = 0
            number_of_chunck = 0
            for large_chunck in dataset.load_data():
                number_of_chunck += 1
                #for each chunck of the data from file, convert it to pair using data.Data.getPairs()
                pairs = data.Data(large_chunck,self.tokenizer).get_pairs()
                #split batch and do it normally
                batches = [pairs[i:i+batch_size] for i in range(0,len(pairs),batch_size)]
                # print(batches)
                chunck_loss = 0
                for batch in batches:
                    input_v,input_l,target_v,mask,target_ml = data_to_sequence(batch)
                    batch_loss = self.train_one_iteration(input_v,input_l,target_v,target_ml,mask,len(batch))
                    chunck_loss += batch_loss
                chunck_loss/= len(batches)
                
                training_iteration_loss += chunck_loss

            training_iteration_loss /= number_of_chunck
            total_loss += training_iteration_loss
            
            delta_loss = abs(training_iteration_loss - prev_loss)
            prev_loss = training_iteration_loss
            training_loss_over_iteration.append(training_iteration_loss)
            testing_loss_over_iteration.append(self.compute_loss(len(evaluation_pairs),evX_train,evX_length,evY_train,evmask,evtarget_ml))

            #print average every print_every iteration
            if iter%print_every == 0:
                logger.log_info(f"iteration {iter} average loss: {total_loss/print_every} percentage {100*iter/iteration:.2f}%, took {time.time() - t} seconds")
                logger.log_info(f"  delta_loss={delta_loss}, prev_loss={prev_loss}") # TODO(Sean) remove
                # testing_loss_over_iteration.append(self.compute_loss(self.testing_set))
                total_loss = 0
                t = time.time()

                
            # save a checkpoint
            if iter % checkpoint == 0 and self._save_checkpoints:
                self.save_model(data_directory=data_directory, model_name=model_name, dataset_name=dataset_name, iteration=iter)

            # check if delta loss for this iteration is < epsilon, and break early if true
            if epsilon is not None and delta_loss < epsilon:
                logger.log_info(f"iteration {iter}: delta_loss={delta_loss} < epsilon={epsilon}, stopping training early!")
                
                if self._save_checkpoints:
                    self.save_model(data_directory=data_directory, model_name=model_name, dataset_name=dataset_name, iteration=iter)
                break
                

        logger.log_info("done training")
        # TODO(Sean) iteration axis not based on start_iteration...

        plt.plot(training_loss_over_iteration,label="training_loss",color="red")
        plt.plot(testing_loss_over_iteration,label="testing_loss",color="green")
        plt.xlabel(f"Iterations")
        plt.ylabel("Loss value")
        plt.legend()
        plt.show()
            
    def predict(self, X_test, max_length: int, topk_choices: int = 1):
        input_v, input_l = get_X_train(X_test)
        input_v = input_v.to(device=device)
        input_l = input_l.to(torch.device("cpu"))
        encoder_output,encoder_hidden = self.encoder(input_v,input_l)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1,1, device=device, dtype=torch.long)
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        
        for _ in range(max_length):
            decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_output)
            
            # non-greedy decoding
            if topk_choices > 1:
                topv,topi = torch.topk(decoder_output, topk_choices, dim=1) #non greedy predict
                topv = F.softmax(topv).cpu().detach().numpy().squeeze()
                
                index_range = np.array(range(len(topv)))
                index = np.random.choice(index_range,p=topv,size=1)
                decoder_input = topi.squeeze()[index]
            else:
                # greedy decoding
                _, topi = decoder_output.topk(1)
                decoder_input = topi.reshape(-1)
                
            
            if decoder_input == 0 or decoder_input == 2:
                break
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens

logger = Logger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_data_directories(dataset_name: str, data_root_dir_path: str, raw_data_dir_path):
    """Create data directories if they do not exist."""
    
    full_path: str = os.path.join(data_root_dir_path, raw_data_dir_path, dataset_name)# f"{data_root_dir_path}/{raw_data_dir_path}/{dataset_name}"
    
    if not os.path.exists(full_path):
        print(f"creating directory '{full_path}'")
        os.makedirs(full_path)

def pre_process_dataset(dataset_name: str, data_root_dir_path: str, raw_data_dir_path: str, force: bool = False):
    """Run pre-processing code to download and aggregate breaking bad transcripts."""
    
    create_data_directories(dataset_name, data_root_dir_path=data_root_dir_path, raw_data_dir_path=raw_data_dir_path)
    store_directory = os.path.join(data_root_dir_path, raw_data_dir_path)
    scrape_transcripts_from_website(raw_data_directory=store_directory, dataset_name=dataset_name, force=force)
    aggregate_raw_transcripts(
        data_directory=data_root_dir_path, 
        raw_transcript_dir_name=raw_data_dir_path, 
        dataset_name=dataset_name, 
        Filter_data=True, 
        force=force
    )
    
def output_to_sentence_str(output: typing.Union[list[int], torch.Tensor], my_tokenizer, include_padding: bool = False):
    if output is torch.tensor:
        print(output.dtype)
    
    if include_padding:
        return " ".join(my_tokenizer.decode(output))
    else:
        return " ".join([token for token in my_tokenizer.decode(output) if token != tokenizer.Tokens.PAD])

def bleu(ref, gen):
    ''' 
    calculate pair wise bleu score. uses nltk implementation
    Args:
        references : a list of reference sentences 
        candidates : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    '''
    logger.log_info("starting to calculate blue score...")
    ref_bleu = []
    gen_bleu = []
    for l in gen:
        gen_bleu.append(l.split())
    for i,l in enumerate(ref):
        ref_bleu.append([l.split()])
    cc = SmoothingFunction()
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    logger.log_info("finished calculating blue score")
    return score_bleu

def evaluate(model,test_pairs,my_tokenizer,max_word):
    logger.log_info("starting to evaluate model...")
    X_train,Y_train = [pair[0] for pair in test_pairs], [pair[1] for pair in test_pairs]
    #sentence as int for now
    predicted = [model.predict([x],max_word,1) for x in X_train]
                
    predicted = [output_to_sentence_str(p, my_tokenizer=my_tokenizer, include_padding=True) for p in predicted]
    Y_train = [output_to_sentence_str(y,my_tokenizer=my_tokenizer) for y in Y_train]
    logger.log_info("finished evaluating model")
    logger.log_info(f"  score: {bleu(Y_train,predicted)}")


def main(args, argv):

    data_directory = "../data/"
    dataset_name_with_extension = "breaking_bad_trim.txt"
    dataset_name = os.path.splitext(os.path.basename(dataset_name_with_extension))[0] # name of the dataset without extension
    # data_csv_name = "BreakingCSV.csv"
    dataset_path = os.path.join(data_directory, dataset_name_with_extension)
    # data_csv_path = os.path.join(data_directory,data_csv_name)
    
    #create a dataset that handle large data file
    my_dataset = data.DataSet(dataset_path)
    evaluation_dataset = data.DataSet("../data/breaking_bad_evaulation.txt")
    #create a tokenizer for tokenization, encoding and decoding
    my_tokenizer = tokenizer.Tokenizer()
    #first initialize the tokenize with some document or use previous computed value
    # my_tokenizer.tokenize(dataset_path)
    # my_tokenizer.save("Pre_computed_breaking_bad.pickle") #if you wanna save the data
    my_tokenizer.load("../data/cache/Pre_computed_breaking_bad.pickle") #if you wanna load the data 
    number_of_tokens = len(my_tokenizer)
    
    
    print(my_dataset)
    print(my_tokenizer)
    my_tokenizer.plotDistribution(10)

    # pre_process_dataset(
    #     dataset_name=dataset_name, 
    #     data_root_dir_path=data_directory, 
    #     raw_data_dir_path=raw_data_directory, 
    #     force=False
    # )
    # gtp3_model_setup(dataset_filepath=dataset_path,json_filepath=data_json_path,force=False)

    # sentences_as_tokens = tokenize(dataset_path, force_compute=False)

    checkpoint = None
    if LOAD_MODEL_CHECKPOINT:
        checkpoint = Dummy3.load_checkpoint(
            data_directory=data_directory, 
            model_name=MODEL_NAME, 
            dataset_name=dataset_name, 
            iteration=ITERATION, 
            n_layers=N_LAYERS, 
            hidden_size=HIDDEN_SIZE,
            cont=CONTINUE_CHECKPOINT  # type: ignore
        )
        


    # #split the pairs into training and testing  60% training, 20% evaluate, 20% testing
    # train_pairs, evaluate_pairs, test_pairs = pairs[:int(numbers_of_pairs*0.6)],pairs[int(numbers_of_pairs*0.6):int(numbers_of_pairs*0.8)],pairs[int(numbers_of_pairs*0.8):]
    

    #network setting
    embedding = nn.Embedding(number_of_tokens, HIDDEN_SIZE)
    if LOAD_MODEL_CHECKPOINT and checkpoint:
        if checkpoint.get("embedding", None) is None:
            logger.log_warn("  embedding failed to load parameters!")
        else:
            embedding.load_state_dict(checkpoint["embedding"])
            logger.log_info(f"  embedding loaded parameters")
    encoder = EncoderRNN(HIDDEN_SIZE,embedding, n_layers=N_LAYERS)
    decoder = DecoderRNN(embedding, HIDDEN_SIZE, number_of_tokens, n_layers=N_LAYERS)
    model = Dummy3(
        HIDDEN_SIZE,
        LEARNING_RATE, 
        encoder=encoder, 
        decoder=decoder, 
        embedding=embedding,
        tokenizer=my_tokenizer,
        save_checkpoints=SAVE_MODEL_CHECKPOINTS, 
        load_checkpoint=LOAD_MODEL_CHECKPOINT,
        cont_checkpoint=CONTINUE_CHECKPOINT,
    ).load(checkpoint=checkpoint).send_optimizers_to_device()
    
    raw_data = evaluation_dataset.load_data(int(len(evaluation_dataset)))
    evaluation_pairs = data.Data(next(raw_data),my_tokenizer).get_pairs()
    numbers_of_pairs = len(evaluation_pairs)
    raw_data = my_dataset.load_data(int(len(my_dataset)))
    training_pairs = data.Data(next(raw_data),my_tokenizer).get_pairs()

    #train the model and predict
    # logger.log_info(f"training on {len(my_dataset)} pairs")
    # model.train(
    #     my_dataset,
    #     evaluation_pairs,
    #     ITERATION,
    #     BATCH_SIZE,
    #     data_directory=data_directory,
    #     dataset_name=dataset_name,
    #     model_name=MODEL_NAME
    # )


    topk_choices = 3 if use_non_greedy_decoding else 1
    #predict the top 3 sentences
    for i in range(3):
        result = model.predict([evaluation_pairs[i][0]],MAX_LENGTH, topk_choices=topk_choices)
        logger.log_info(f"Query: {output_to_sentence_str(output=evaluation_pairs[i][0], my_tokenizer=my_tokenizer, include_padding=True)}")
        logger.log_info(f"  Predict: {output_to_sentence_str(result, my_tokenizer=my_tokenizer, include_padding=True)}")
        logger.log_info(f"  True Y: {output_to_sentence_str(output=evaluation_pairs[i][1], my_tokenizer=my_tokenizer, include_padding=True)}")
        print()

    #evaluting using BLEU, return a float value from 0-1, the higher the better
    if EVALUATE_MODEL:
        logger.log_info("Training BLEU: ")
        evaluate(model,training_pairs,my_tokenizer,MAX_LENGTH)
        logger.log_info("Evaluation BLEU: ")
        evaluate(model,evaluation_pairs,my_tokenizer,MAX_LENGTH)
        # logger.log_info("Testing BLEU")
        # evaluate(model,evaluation_pairs,token_list,MAX_LENGTH)
    
    #real time prediction
    logger.log_info("Lets do some real predictions, type 'quit' to end")
    while True:
        query = input(">")
        # tokenize input query
        if(len(query) == 0 or query == "quit"):
            break
        tokenized_query = my_tokenizer.encode(query)
        # convert tokenized sentence to indices
        result = model.predict([tokenized_query], MAX_LENGTH, topk_choices=topk_choices)
        logger.log_info(output_to_sentence_str(output=result, my_tokenizer=my_tokenizer, include_padding=True))
    
        

if __name__ == "__main__":
    main(len(sys.argv),sys.argv)
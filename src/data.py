from logger import Logger
import tokenizer
import os
import sys

logger = Logger()

class Data:
    def __init__(self, raw_sentence:[str], tokenizer) -> None:
        self.raw_sentence = raw_sentence
        self.tokenizer = tokenizer
        self.size = len(raw_sentence)

    def get_pairs(self,delimiter="\t"):
        """
        get the encoded version of the sentence as pairs
        """
        result = []
        for i in range(self.size):
            _ = self.raw_sentence[i].split(delimiter)
            if len(_) != 2 or len(_[0]) == 0 or len(_[1]) == 0:
                continue
            encoded = [self.tokenizer.encode(_[0]), self.tokenizer.encode(_[1])]
            if len(encoded[0]) <= 0 or len(encoded[1]) <= 0:
                continue 
            result.append(encoded)
        return result
   
class DataSet:
    def __init__(self,filename):
        self.filename = filename
        if not os.path.exists(filename):
            print(f"path {filename} not found")
            return
        self.file = open(filename,"r",encoding="utf-8")
        self.number_of_line = 0
        for line in self.file:
            self.number_of_line += 1
        self.file.seek(0)

    
    def load_data(self,number:int=32768):
        """
        read number of lines from the datafile
        """
        #set the file pointer to the begining
        self.file.seek(0)
        lines = []
        #iterate through the entire file
        for i in range(1,self.number_of_line+1):    
            line = self.file.readline()
            lines.append(line)
            #return this chunck of data if it is at an end or reach the number
            if line=="" or i % number == 0:
                yield lines
                #make sure to clean the list for next big chunck
                lines.clear()
        yield lines
        #terminated

    def switch_datasete(self,filename):
        if not os.path.exists(filename):
            print(f"path {filename} not found, switch failed, used the old one")
            return
        self.file.close()
        self.filename = self.filename
        self.file = open(filename,"r")

    def __len__(self):
        return self.number_of_line

    def __str__(self):
        return f"-----DataSet {self.filename} number_of_line {self.number_of_line}"



def trim(filename,output_name,min_count,max_count):
    original_line_count = 0
    new_line_count = 0
    with open(filename,"r",encoding="utf-8",errors="replace") as f:
        output = open(output_name,"w",encoding="utf-8")
        for line in f:
            sentences = line.strip().split("\t")
            word_count1 = len(sentences[0].split(" "))
            word_count2 = len(sentences[1].split(" "))
            original_line_count +=1
            if word_count1 >= min_count and word_count2 >= min_count and word_count1 <= max_count and word_count2 <= max_count:
                output.write(sentences[0]+"\t"+sentences[1]+"\n")
                new_line_count +=1
            if original_line_count %10000 == 0:
                print(f"{original_line_count} lines processed")
        print(f"number of sentence trimmed from {original_line_count} to {new_line_count}, removed percentage {(original_line_count-new_line_count)/original_line_count:.3f}")





if __name__ == "__main__":
    # trim("./data/reddit.txt","./data/reddit_trim.txt",3,10)
    filename = "../data/breaking_bad_trim.txt"
    if len(sys.argv) == 2:
        filename = "../data/" +sys.argv[1]
    my_dataset = DataSet(filename)
    

    my_tokenizer = tokenizer.Tokenizer()
    my_tokenizer.tokenize(filename)
    # my_tokenizer.load("Pre_computed.pickle")
    print(my_dataset)
    print(my_tokenizer)
    for i,chunck in enumerate(my_dataset.load_data(4096)):
        print(f"loading a chunk of data {len(chunck)} chunck {i}")
        datas = Data(chunck,my_tokenizer).get_pairs()
        for i,pairs in enumerate(datas):
            if(len(pairs[0]) <= 0 or len(pairs[1]) <= 0):
                print(f"Error {i} {pairs} {len(pairs[0]),len(pairs[1])}")
                exit()
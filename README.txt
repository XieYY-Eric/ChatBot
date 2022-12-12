This is the Monkey's Brain

To run a small demo, run the `project.ipynb` file in the src directory.

	Seq2Seq.py:
To run the Seq2Seq model, execute Seq2Seq.py
"python Seq2Seq.py"
You can edit the global values inside the seq2seq.py file to decide if you want to show the BLUE evaulation result
For non-greedy prediction, set the variable to True, by default, it choose the top 3 highest word, 
or you can also change the topk_choices value to however you want

    jes-1.py:
To run the transformer model, execute jes-1.py
`python jes-1.py`
You can edit the hyperparameters at the top of jes-1.py file, though this may stop the pre-trained model from loading (when changing hyperparameters that affect the architure).
Non-greedy prediction is not implemented for the transformer.

	tokenizer.py:
Contains a demo for how to tokenize a datafile inside data directory, also generate an word percentage distribution
the internal state of tokenizer will be saved to a pickle file
"python tokenizer reddit_trim.txt save_filename"


	data.py:
Loading stream of data from a txt file, used to handle a large dataset that can't fit in the memory at once
"python data.py reddit_trim.txt"


	gpt3.ipynb:
Need to fill in envirenment variable of API_key. Then to run fine tuned models, run these lines 

"!openai api completions.create -m curie:ft-personal-2022-11-17-07-40-44 -p "*Test String Here* ->"

or

"!openai api completions.create -m curie:ft-personal-2022-11-17-01-49-03 -p "How do you know my name?"

where the first one was tested with an input structure requirement while the second one does not. If you want to see
the graphs then all the other lines would need to executed except for the training related cells.
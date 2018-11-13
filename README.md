Accompanying code for Violence Rating Prediction from Movie Scripts
====================================================================

## Folder Structure
 * experiments: Contains the code to run the experiments
 * notebooks: Model error analysis
 * lexicons: Contains the folders to store the lexicons
 

## Instructions:
### Download Lexicons:
	1. AFINN-111 into lexicons/AFINN from [AFINN](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010)
	2. categories.tsv into lexicons/empath from [Empath](https://github.com/Ejhfast/empath-client)
	3. hatebase_dict.csv and refined_ngram_dict.csv into lexicons/hatespeech_davidson from <https://github.com/t-davidson/hate-speech-and-offensive-language>
	4. expandedLexicon.txt into lexicons/lexicon_abusive_words from <https://github.com/uds-lsv/lexicon-of-abusive-words>
	5. vader_lexicon.txt into lexicon/vader from <https://github.com/cjhutto/vaderSentiment>

### Experiments
The experiments folder contains the scripts to replicate the experiments presented in the paper. All models were trained on cross-validated fashion with the folds pre-calculated and stored in hard drive. We added a bash script to ease the run of the experiments.


#### RNN models
The script RNN_CV runs the RNN model on k-fold CV. It takes as arguments the following:
	- fold_dir: directory with the cross validation folds
	- outf: output file for the cross validation predictions
	- (opt) model_name: name for the output model
	- (opt) max_len: number of utterances to consider (between 500 and 1000, defaults to 500).
	- (opt) epochs: number of epochs to run the model (defaults to 30)
	- (opt) batch_size: batch size (defaults to 16)
	- (opt) FEAT: list of names of features (defaults to 'ngrams', 'w2v', 'afinn', 'vader', 'hatebase', 'empath_192', 'abusive', 'empath_2')
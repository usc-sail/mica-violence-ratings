#!/usr/bin/env python3
import os, subprocess

FEAT=[
	  (['w2v', 'afinn', 'vader', 'hatebase', 'empath_192', 'abusive', 'empath_2'], 'ngrams'),
	  (['ngrams', 'afinn', 'vader', 'hatebase', 'empath_192', 'abusive', 'empath_2'], 'w2v'),
	  (['ngrams', 'w2v', 'hatebase', 'empath_192', 'abusive'], "sentiment"),
	  (['ngrams', 'w2v', 'afinn', 'vader', 'empath_192', 'empath_2'], "abusive"),
	  (['ngrams', 'w2v', 'afinn', 'vader', 'hatebase', 'abusive', 'empath_2'], 'linguistic'),
	 ]

for i in range(len(FEAT)):
	feats, wo = FEAT[i]
	command = ["./runCV.sh",
			   "RNN_CV.py",
			    "2",
			    "/work/victor/abusive-lang-workshop/data/batches/reversefolds",
			    "ABL_16GRU_500_loss_wo{}".format(wo),
			    " ".join(feats)]
	subprocess.check_call(" ".join(command), shell=True)
	print("ABL w/o {} returned".format(wo))
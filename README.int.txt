The sense-tagged interest data is taken from the pos-tagged portion of the 
WSJ that appears in the ACL/DCI version of the Penn Treebank (this is the 
1990 preliminary version). Each instance of interest is tagged with one of
six possible LDOCE senses. A brief key to the sense tags used in this text 
follows:

Sense 1 =  361 occurrences (15%) - readiness to give attention
Sense 2 =   11 occurrences (01%) - quality of causing attention to be given to
Sense 3 =   66 occurrences (03%) - activity, etc. that one gives attention to
Sense 4 =  178 occurrences (08%) - advantage, advancement or favor
Sense 5 =  500 occurrences (21%) - a share in a company or business
Sense 6 = 1252 occurrences (53%) - money paid for the use of money
	  ----
	  2368 occurrences in the sense tagged corpus, where each
	occurrence is a single sentence that contains the word 'interest'.

The tagging was done by Rebecca Bruce and Janyce Wiebe and is described in 
their ACL-94 paper. Please give them credit for creating this data! 

@inproceedings{BruceW94B,
        author = {Bruce, R. and Wiebe, J.},
        title = {Word-Sense Disambiguation using Decomposable Models},
        booktitle={Proceedings of the 32nd Annual Meeting of the
                  Association for Computational Linguistics},
        year = {1994},
        pages = {139-146}}
                            
This data has also been used in a number of other papers, where it is
described in varying degrees of detail. Most of these papers are
available from the cmp-lg server http://xxx.lanl.gov/find/cs:

@inproceedings{Pedersen00b,
        author = {Pedersen, T.},
        title = {A Simple Approach to Building Ensembles of Naive Bayesian
Classifiers for Word Sense Disambiguation},
        booktitle = {Proceedings of the North American Chapter of the
Association for Computational Linguistics},
        year = {2000},
        month ={May},
        address = {Seattle, WA}}      

@inproceedings{PedersenBW97,
        author = {Pedersen, T. and Bruce, R. and Wiebe, J.},
        title = {Sequential Model Selection for Word Sense
                  Disambiguation},
        year = {1997},
        month = {April},
        address = {Washington, DC},
        pages = {388--395},
        booktitle = {Proceedings of the Fifth Conference on Applied
           Natural Language Processing}}

@inproceedings{PedersenB97A,
        author = {Pedersen, T. and Bruce, R.},
        title = {A New Supervised Learning Algorithm for Word Sense
Disambiguation},
        year = {1997},
        booktitle= {Proceedings of the Fourteenth National Conference on
                  Artificial Intelligence},
        month = {July},
        pages = {604--609},
        address = {Providence, RI}} 

@inproceedings{NgL96,
        author = {Ng, H.T. and Lee, H.B.},
        title ={Integrating Multiple Knowledge Sources to Disambiguate
                  Word Sense: An Exemplar-Based Approach},
        booktitle ={Proceedings of the 34th Annual Meeting of the
                  Society for Computational Linguistics},
        pages = {40--47},
        year = {1996}}
                              

@inproceedings{BruceWP96,
        author = {Bruce, R. and Wiebe, J. and Pedersen, T.},
        title = {The Measure of a Model},
        booktitle={Proceedings of the Conference on Empirical Methods
                  in Natural Language Processing},
        pages = {101-112},
        year = {1996}}     

What follows is the original documentation provided by Bruce and Wiebe.
========================================================================

The attached data file is composed of sentences containing the noun
"interest" or "interests" that have been automatically extracted from
the Penn Treebank Wall Street Journal corpus.  The data includes the
part-of-speech tags and phrase bracketing provided in the original
corpus.

Each sentences in the data file contains one sense-tagged occurrence
of the word "interest" (or "interests").  The sense tags correspond
the six non-idiomatic noun senses of "interest" defined in the
electronic version of the first edition of Longman's Dictionary of
Contemporary English.  The sense tags are appended to the end of the
word prior to the part-of-speech tag as shown in the following
example:

	interest_6/NN

In the example above, "interest" is identified as having the sixth
sense (i.e. "_6") listed in the Longman's Dictionary of Contemporary
English.  All sense tags were manually assigned.

Each sentence in the data file is delineated by a line containing the
symbol "$$".  In total, there are 2,369 sentences.  For a more
complete description of the data and its original purpose see:

	Bruce, R. and Wiebe, J. (1994).
	Word-Sense Disambiguation Using Decomposable Models
	{\it Proceedings of the 32nd Annual Meeting of the Association
	for Computational Linguistics (ACL-94)}.



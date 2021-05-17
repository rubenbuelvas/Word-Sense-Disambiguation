Word Sense Disambiguation
==============

Rubén Buelvas

Université de Montréal
IFT 3335 - Artificial Intelligence
Winter 2021

*Python Version 3.0+*

All functionality is contained in the main.py file. The list of requirements can be found in the file requeriments.txt.


Generate datasets
----------------------------

We can use the program to generate datasets from the original source (both interest-original.txt and interest.acl94.txt files have to be present in the same folder). The program does the feature extraction and generates 4 .csv files in the same folder with the preprocessed data.

To generate these files, use the following command: 

	python main.py generate_datasets


Performance tests
----------------------------------

To test the performance of different configurations for the decision tree and multi-layer perceptron models, the program has two functions. The program tests a pre-configured range of settings for each model and then saves the performance results in .xlsx files. 

The decision tree function generates results for all datasets at once. To test the configurations for the decision tree model, use the following command:

	python main.py test_dt_config

IN the case of the multi-layer perceptron, testing can take some time, so the function can generate results for only one dataset at a time.  To test the configurations for the multi-layer perceptron model, use the following command:

	python main.py test_mlp_config <dataset name>

We can replace <dataset> with the words “gc”, “nw”, “ws”, and “ws_w_gc”.
  
If nevertheless, we want to test all configurations for all datasets at once, we simply use the command without the second argument:

	python main.py test_mlp_config


Run with optimal configurations
----------------------------------

Finally, we can test the models with their optimal configuration for each dataset. These configurations can be changed by editing the dictionary at the top to the main.py file. Running these commands will also generate weight files for the MLP model.

To run all the models use the following commands:

	python main.py run_models

Using arguments, we can run a certain model with a certain dataset. We can replace <model> with the words “nb”, “dt”, and “mlp” and <dataset> with the words “gc”, “nw”, “ws”, and “ws_w_gc”:

	python main.py run_model <model> <dataset>
  

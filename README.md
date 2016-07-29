## Architectures

cf /src

- Phone\_Emb4 : Input NN : two one hot vectors for left and right context. 
                Output NN : central phoneme 
                One hidden layer in two copies constrained to be the same for the left and the right phoneme. 
                The outputs of the hiddenlayer are concatenated before softmax.

- Phone_Emb5: Input NN : two one hot vectors for left and right context + central phoneme for autoencoder
                Output NN : central phoneme (also output of the autoencoder)
                2 hiddenlayers: _one hidden layer in two copies constrained to be the same for the left,the right and the central phoneme.
                                _the second one transform the concatenation of the left and right context from the first hidden layer into a vector of the size of the first hidden layer. 
- Phone_Emb6: Idem Phone_Emb5 exept input is FIVE one hot vectors, 2 for left and 2 for right context + central phoneme for autoencoder

- Phone_Emb7: Idem Phone_Emb4 exept input is FOUR one hot vectors, 2 for left and 2 for right context.
 
- phoneCBOW2.py
    Inputs : Vecteurs one-hot représentant les phonèmes à gauche et à droite sont additionnés
    Architecture de idem article T. Mikolov
    
- PhonesSkipgram1.py
    Inputs : vecteur correspondant à un phonème
    Target : phonème suivant seulement

## Tools

cf /src

-Dico_dictio_emb : Input: dictionary + trained model
                    Ouput: coordinates of the phonemes in the embedding space

-GetStatistics2: Input: Buckeye corpus
                    Output: 3 cvs files containing the counts of occurrences of contexts vs phonemes  

-Use_CBOW2_Model : Input: .word file. 
                    Ouput: error rate of the model

## Results

cf /results

- Saved models 
- Dictionnary representation in the embedding space (first hidden layer)
- CVS files of the error rates for various hyperparameters or embedding space sizes
- ods files putting together the results for each architecture for embedding evaluation with Dunbar method
- /reproductibility : runs to evaluate the reproducibility of the embedding with Phone_Emb4. Differents example shuffling and parameter initialization.

## Data

Buckeye corpus as been cleaned of typos. Find cleaned corpus in /data/data_cleaned


- GetPhones_dictio.py  (/data)
    Formating the corpus from the .word files. Output list of phone in the same order. See options at the top

- BuckeyeDictionary 
    Builds the dictionary of phonems from a corpus. 

## Evaluation

TRAINING: 
-parameters.Rmd: Evaluation of hyperparameters 
-count_error_rate.Rmd: fixes a baseline for the optimal error rate choosing the most probable phone in each context (one phoneme left and right)

EMBEDDING:
With https://github.com/bootphon/suprvenr package

Used test pairs at /test_pairs.csv




<!--
                       _   _                       _       _   _             
__      _____  _ __ __| | | |_ _ __ __ _ _ __  ___| | __ _| |_(_) ___  _ __  
\ \ /\ / / _ \| '__/ _  | | __| '__/ _` | '_ \/ __| |/ _` | __| |/ _ \| '_ \ 
 \ V  V / (_) | | | (_| | | |_| | | (_| | | | \__ \ | (_| | |_| | (_) | | | |
  \_/\_/ \___/|_|  \__,_|  \__|_|  \__,_|_| |_|___/_|\__,_|\__|_|\___/|_| |_|
-->                                                                  

# Word translation using PLSR 
This project is based on the supervised learning tutorial by Aurelie Herbelot available at https://github.com/ml-for-nlp/word-translation.

## Dependencies
The only requirements are docopt and scikit-learn, which are included in ``requirements.txt``.

## Arguments
The main file ``plsr_regression.py`` can take different arguments when called from the terminal.
```
python3 -W ignore plsr_regression.py --lang LANGUAGES ((-a | --auto) | --ncomps N –nns N) [-v | --verbose]
```
where
- --langs (ENGCAT | CATITA | ENGITA) needs to be one of the three possible pairs of languages to translate between.
- [-a | --auto] activates automatic mode for hyperparameter search, instead of manually inputting --ncomps and --nns.
- --ncomps N is the number of principal components to use, which defaults to 30.
- --nns N is the number of nearest neighbours used for the evaluation, which defaults to 10.
- [-h | --help] shows the help screen.
- --version show the version of the code.
- [-v | --verbose] shows verbose output, i.e. it prints the gold standard and the list of k nearest neighbours. It will not work in automatic mode, as it implies too much printing.

For example, to run the program for Catalan-Italian word translation using 15 principal components and 5 nearest neighbours and checking the verbose answer, we would write
```
python3 -W ignore plsr_regression.py --lang CATITA --ncomps 15 --nns 5 -v
```
and the output should look somewhat like this
```
muller moglie ['donna', 'persona', 'lei', 'madre', 'ragazza'] 0
Precision PLSR: 0.3125
```

## The data
All data can be found in the data folder. This includes three semantic spaces and three lists of pairs. 
- The three semantic spaces files were provided in the original tutorial by Herbelot, as well as the English-Catalan pairs list. 
- The Catalan-Italian pairs list was handmade by me using the aforementioned semantic spaces.
- The English-Italian pairs list was provided by my colleague Sam, and it is available at https://github.com/samueleantonelli/PLSR-words-translation.

## About the automatic mode and results
The main difference between this project and the original tutorial is the implementation of an automatic mode. When [-a | --auto] is activated, the program will perform an hyperparameter search, meaning that the user does not need to input any value for --ncomps or --nns, as they model will try out a list of possible values and return all results in the form of a table. The results table will be printed in the terminal, but also stored as a .csv file so it can be easily manipulated afterwards. The results folder in this repository includes these three .csv files and an excel file containing the same three tables for an easier visualisation. 

## Visualising the semantic spaces
The three semantic spaces can be visualised in 2D plots which are found in the plots folder. The code used to generate these images is commented at the end of the main file and needs to be uncommented in order to run it again.

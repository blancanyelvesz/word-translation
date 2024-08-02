"""Word translation

Usage:
  plsr_regression2.py [options]
  plsr_regression2.py (-h | --help)
  plsr_regression2.py --version

Options:
  --langs (ENGCAT|CATITA|ENGITA)  Choose one of the three pairs to translate between.
  -a --auto         Uses a loop instead of --ncomps and --nns.
  --ncomps <n>      Number of principal components. [default: 1]
  --nns <n>         Number of nearest neighbours for the evaluation. [default: 1]
  -h --help         Show this screen.
  --version         Show version.
  -v --verbose      Show verbose output. Will not work in automatic mode, as it implies too much printing.

"""

from docopt import docopt
import numpy as np
import utils
from utils import *
from sklearn.cross_decomposition import PLSRegression
import csv


def mk_training_matrices(pairs, dimension1, dimension2, subspace1, subspace2):
    matrix1 = np.zeros((len(pairs), dimension1)) 
    matrix2 = np.zeros((len(pairs), dimension2))
    c = 0
    for p in pairs:
        word1, word2 = p.split()
        matrix1[c] = subspace1[word1]   
        matrix2[c] = subspace2[word2]   
        c+=1
    return matrix1, matrix2


def PLSR(matrix1, matrix2, ncomps):
    plsr = PLSRegression(n_components = ncomps)
    plsr.fit(matrix1, matrix2)
    return plsr 


def evaluate(plsr, testpairs, subspace1, subspace2, nns):
    score = 0
    for p in testpairs:
        lang1, lang2 = p.split()
        predicted_vector = plsr.predict(subspace1[lang1].reshape(1,-1))[0]
        # print(predicted_vector[:20])
        nearest_neighbours = utils.neighbours(subspace2, predicted_vector, nns)
        if lang2 in nearest_neighbours:
            score += 1
            if verbose:
                print(lang1, lang2, nearest_neighbours, "1")
        else:
            if verbose:
                print(lang1, lang2, nearest_neighbours, "0")
    precision = score / len(testpairs)
    return precision


if __name__ == '__main__':
    args = docopt(__doc__, version = 'PLSR regression for word translation 2.0 ~ edited by Blanca 😊')
    languages = args["--langs"]
    verbose = False
    if args["--verbose"]:
        verbose = True
    auto = False
    if args["--auto"]:
        auto = True
        verbose = False

    ''' Read semantic spaces of all three languages '''
    english_space = utils.readDM("data/english.subset.dm")
    catalan_space = utils.readDM("data/catalan.subset.dm")
    italian_space = utils.readDM("data/italian.subset.dm")

    ''' Read word pairs of the chosen pair of languages '''
    all_pairs = []
    f = open(f"data/pairs_{languages}.txt")
    for l in f:
        l = l.rstrip('\n')
        all_pairs.append(l)
    f.close()

    ''' Make training/test fold using 70% split '''
    split = int(len(all_pairs) * 0.70)
    training_pairs = all_pairs[:split]
    test_pairs = all_pairs[split+1:]

    ''' Make training/test matrices and get PLSR model for the chosen pair of languages '''
    if languages == "ENGCAT":
        space1, space2 = english_space, catalan_space
        dimspace1, dimspace2 = 400, 300
    elif languages == "CATITA":
        space1, space2 = catalan_space, italian_space
        dimspace1, dimspace2 = 300, 300
    elif languages == "ENGITA":
        space1, space2 = english_space, italian_space
        dimspace1, dimspace2 = 400, 300

    mat1, mat2 = mk_training_matrices(training_pairs, dimspace1, dimspace2, space1, space2)
    
    ''' Make lists of possible values to assign to ncomps and nns '''
    ncomps_list = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100] 
    nns_list = [1, 2, 3, 5, 10, 15, 20]

    ''' Predict with PLSR '''
    
    def create_table(ncomps_values, nns_values):
        results = []
        cols = ['' , 1, 2, 3, 5, 10, 15, 20]
        print(*cols, sep = '\t')
        for ncomps in ncomps_values:
            plsr = PLSR(mat1, mat2, ncomps)
            results_ncomps = [ncomps]
            for nns in nns_values:
                result_nns = round(float(evaluate(plsr, test_pairs, space1, space2, nns)), 5)
                results_ncomps.append(result_nns)
            results.append(results_ncomps)
            print(*results_ncomps, sep = '\t')
        with open(f'./results/{languages}_results.csv', 'w') as file:
            write = csv.writer(file)
            write.writerow(cols)
            write.writerows(results)
        return results

    if auto:
        print(f"Table with precision results:")
        results = create_table(ncomps_list, nns_list)
        print(f"rows: ncomps, columns: nns")
    else:
        ncomps = int(args["--ncomps"])
        nns = int(args["--nns"])
        plsr = PLSR(mat1, mat2, ncomps)
        result = evaluate(plsr, test_pairs, space1, space2, nns)
        print(f"Precision PLSR: {result}")

    ''' Create PNGs of the semantic spaces: UNCOMMENT TO USE!!'''
    # create_pngs()
# coding: utf-8
import os
import pandas as pd
import numpy as np
import pickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
stop_words = stopwords.words('english')
import nltk
from gensim.similarities import WmdSimilarity
import gzip

root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

MODEL_PATH = root_dir_path + '/data/GoogleNews-vectors-negative300.bin.gz'
model = None
norm_model = None

def wmd(s1, s2):

    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)

def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

def load_models():
    """" """
    global model
    if model == None:
        printer('loading model')
        model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

    global norm_model
    if norm_model == None:
        printer('loading normalized model')
        norm_model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
        norm_model.init_sims(replace=True)

def vectorize(question, xDim):
    printer('generating sent2vec represnetation for Question')
    question_vectors = np.zeros((xDim, 300))
    for i, q in enumerate(questions):
        question_vectors[i, :] = sent2vec(q)
    return question_vectors

def printer(line): print ('\x1b[2K{}'.format(line), end='\r')

def engineer_features(q1=None, q2=None, data=None, w2v=True, save=True):
    """

    :w2v: - boolean - flag indicating if we should load models and perform word2vec operations
    """
    if data is None:
        if q1 == None or q2 == None:
            print ('must pass either non None q1, q2 values or provide a dataset as the data parameter.')
            return
        if not (type(q1) is list):
            q1 = [q1]
        if not (type(q2) is list):
            q2 = [q2]

        data = pd.DataFrame({'question1': q1, 'question2': q2})

    if not os.path.exists('data'):
        os.mkdir('data')

    printer ('generating length of Question 1')
    data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
    printer ('generating length of Question 2')
    data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
    printer ('generating length differences')
    data['diff_len'] = data.len_q1 - data.len_q2
    printer ('generating character length of Question 1')
    data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    printer ('generating character length of Question 2')
    data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    printer ('generating number of words for Question 1')
    data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
    printer ('generating number of words for Question 2')
    data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
    printer ('generating common words between the questions')
    data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
    printer ('generating fuzz_qratio')
    data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    printer ('generating fuzz_WRatio')
    data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    printer ('generating fuzz partial ratio')
    data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    printer ('generating fuzz partial token set ratio')
    data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    printer ('generating fuzz partial token sort ratio')
    data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    printer ('generating fuzz token set ratio')
    data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    printer ('generating fuzz token sort ratio')
    data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

    outfile = 'data/data_pt1.csv'
    printer('generated features, writing to file {}'.format(outfile))
    data.to_csv(outfile)

    data = pd.read_csv(outfile)
    data = data.drop(data.columns[0], axis=1)


    if w2v:
        load_models()

        printer('generating wmd')
        data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
        printer('generating normalized wmd')
        data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

        question1_vectors  = np.zeros((data.shape[0], 300))
        for i, q in enumerate(data.question1.values):
            question1_vectors[i, :] = sent2vec(q)

        question2_vectors  = np.zeros((data.shape[0], 300))
        for i, q in enumerate(data.question2.values):
            question2_vectors[i, :] = sent2vec(q)

        printer('generating cosine distance')
        data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                  np.nan_to_num(question2_vectors))]

        printer('generating cityblock_distance')
        data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                  np.nan_to_num(question2_vectors))]

        printer('generating jaccard distance')
        data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                  np.nan_to_num(question2_vectors))]

        printer('generating canberra distance')
        data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                  np.nan_to_num(question2_vectors))]

        printer('generating euclidean distance')
        data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                  np.nan_to_num(question2_vectors))]

        printer('generating minkowski distance')
        data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                  np.nan_to_num(question2_vectors))]

        printer('generating braycurtis distance')
        data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                  np.nan_to_num(question2_vectors))]

        printer('generating skew vector for Question 1')
        data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
        printer('generating skew vector for Question 2')
        data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
        printer('generating kur vector for Question 1')
        data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
        printer('generating kur vector for Question 1')
        data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

        # print ('\nshape of q1v is {}'.format(question1_vectors.shape))
        q1v = pd.DataFrame(question1_vectors, columns=[str(i) for i in range(question1_vectors.shape[1])])
        q2v = pd.DataFrame(question2_vectors, columns=[str(i)+'_q2' for i in range(question2_vectors.shape[1])])

        # Write data to pickle
        data = pd.concat([data, q1v, q2v], axis=1)
        data.to_pickle(root_dir_path + '/data/Quora_featured')
        data.to_csv(root_dir_path + '/data/full_data.csv')
        return data

    print ('\x1b[2KSuccessfully generated features.')


# Test
# q1 = 'Is piped text compatible with Web Services?'
# q2 = 'Can you used piped text in a Web Service?'
# engineer_features(q1, q2)

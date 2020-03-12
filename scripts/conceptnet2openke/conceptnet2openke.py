"""
@author: Rehan Ahmed
This is an end-to-end script to convert the assertions of ConceptNet to 
train/test input consumable by OpenKE models for a given language.
Steps:
    1) Download the assertions file into provided cache folder.
    2) Unzip the the file to get the assertions.csv file
    3) Read and create intermediate assertions-lang.csv file which
        contains the assertions for the given language at input
    4) Extract all the entities that are part of at least n assertions
    5) Create train2id.txt file
"""

import sys
from statistics import mean
import os.path
import csv
from collections import defaultdict
from random import shuffle
from tqdm import tqdm
import requests
import gzip
import shutil


def download(url, save_path):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")

def download_assertions(cache_folder):
    if not os.path.exists(cache_folder):
        try:
            os.makedirs(cache_folder)
        except OSError:
            print("Creation of the directory %s failed" % cache_folder)
        else:
            print("Successfully created the directory %s" % cache_folder)

    print("Downloading conceptnet-assertions-5.7.0.csv.gz")
    assertion_gz_file = "%s/conceptnet-assertions-5.7.0.csv.gz"%cache_folder
    if not os.path.exists(assertion_gz_file):
        url = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
        download(url, assertion_gz_file)

    assertion_csv_file = "%s/assertions.csv"%cache_folder
    print('Extracting conceptnet.csv.gz file')
    if not os.path.exists(assertion_csv_file):
        with gzip.open(assertion_gz_file, 'rb') as f_in:
            with open(assertion_csv_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def extract_lang_assertions(language, cache_folder):
    assertions_file = '%s/assertions.csv'%cache_folder
    if not os.path.isfile(assertions_file):
        download_assertions(cache_folder)

    print("Extracting assertions for language: %s"%language)

    lang_str='/c/%s/'%language
    assertions_lang_file = "%s/conceptnet-%s.csv"%(cache_folder, language)

    with open(assertions_file) as af, open(assertions_lang_file, 'w') as cf:
        ass_reader = csv.reader(af, delimiter='\t')
        for row in ass_reader:
            r = row[1]
            c1 = row[2]
            c2 = row[3]

            if (c1.startswith(lang_str) and c2.startswith(lang_str)):
                cf.write('\t'.join([c1, r, c2]))
                cf.write('\n') 


def generate_topn_ent_rel(output_folder, assertion_lang_file, topn_ents=20000):
    '''
    Step 4: Extract all the entities that are part of at least n assertions
    :param output_folder:
    :param assertion_lang_file:
    :param topn_ents:
    :return:
    '''
    n_ent_rel_dict_head = defaultdict(int)
    n_ent_rel_dict_tail = defaultdict(int)
    relations = set([])

    with open(assertion_lang_file) as rf:
        for line in rf:
            row = line.strip().split('\t')
            relations.add(row[1])
            n_ent_rel_dict_head[row[0]] += 1
            n_ent_rel_dict_tail[row[-1]] += 1

    head_ents = set(n_ent_rel_dict_head.keys())
    tail_ents = set(n_ent_rel_dict_tail.keys())

    entities = list(head_ents.union(tail_ents))

    n_ent_avg_head_tail = {ent: mean([n_ent_rel_dict_head[ent], n_ent_rel_dict_tail[ent]]) for ent in entities}

    ordered_entities = sorted(n_ent_avg_head_tail.items(), \
                            key=lambda x: x[1], \
                            reverse=True)[:topn_ents]

    entity2id_file = "%s/entity2id.txt"%output_folder
    with open(entity2id_file, 'w') as ef:
        ef.write('%d\n'%len(ordered_entities))
        ef.write('\n'.join([ '\t'.join([v[0], str(i)]) for i, v in enumerate(ordered_entities)]))

    relation2id_file = "%s/relation2id.txt"%output_folder
    relations = sorted(list(relations))
    with open(relation2id_file, 'w') as rf:
        rf.write('%d\n'%len(relations))
        rf.write('\n'.join([ '\t'.join([v, str(i)]) for i, v in enumerate(relations)]))

def create_train2id(output_folder, assertion_lang_file):
    '''
    Step 5: Create train2id.txt file
    :param output_folder:
    :param assertion_lang_file:
    :return:
    '''
    entity2id_file = "%s/entity2id.txt"%output_folder
    relation2id_file = "%s/relation2id.txt"%output_folder
    with open(entity2id_file) as ef, open(relation2id_file) as rf:
        e_rows = [line.strip().split('\t') for line in ef.readlines()[1:]]
        r_rows = [line.strip().split('\t') for line in rf.readlines()[1:]]
        entity2id = {row[0]:row[1] for row in e_rows}
        relation2id = {row[0]:row[1] for row in r_rows}

    with open(assertion_lang_file) as af, open('%s/train2id.txt'%output_folder, 'w') as tf:
        training_rows = []
        for line in af:
            row = line.strip().split('\t')
            if (row[0] in entity2id and row[-1] in entity2id):
                training_rows.append(row)
        
        shuffle(training_rows)
        tf.write('%d\n'%len(training_rows))
        for row in training_rows:
            tf.write(' '.join([entity2id[row[0]], entity2id[row[-1]], relation2id[row[1]]]))
            tf.write('\n')

        
if __name__=='__main__':
    if len(sys.argv) != 5:
        print('error. usage: python conceptnet2openke.py cache_folder_path language topn_ents output_folder')
    else:
        # folder to download ConceptNet assertions file in and store intermediate files
        cache_folder = sys.argv[1]
        
        # language of the concepts to be extracted
        language = sys.argv[2]

        # use only topn most popular concepts for generating training set
        topn_ents = int(sys.argv[3])

        # folder to store openke consumable files
        output_folder = sys.argv[4]
        
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except OSError:
                 print ("Creation of the directory %s failed" % output_folder)
            else:
                print ("Successfully created the directory %s" % output_folder)

        assertion_lang_file = "%s/conceptnet-%s.csv"%(cache_folder, language)
        if not os.path.isfile(assertion_lang_file):
            print("Assertions file for the language not found!")
            print("Creating the Assertions file in cache folder")
            extract_lang_assertions(language, cache_folder)

        print('Extracting top n-most popular concepts')
        generate_topn_ent_rel(output_folder, assertion_lang_file, topn_ents)

        print('Creating train2id.txt file')
        create_train2id(output_folder, assertion_lang_file)

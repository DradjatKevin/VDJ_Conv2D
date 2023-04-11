import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def parse_fasta(filename, a3m=False, stop=10000):
  '''function to parse fasta file'''
  
  if a3m:
    # for a3m files the lowercase letters are removed
    # as these do not align to the query sequence
    rm_lc = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    
  header, sequence = [],[]
  lines = open(filename, "r")
  for line in lines :
    line = line.rstrip()
    if len(line) > 0:
      if line[0] == ">":
        if len(header) == stop:
          break
        else:
          header.append(line[1:])
          sequence.append([])
      else:
        if a3m: line = line.translate(rm_lc)
        else: line = line.upper()
        sequence[-1].append(line)
  lines.close()
  sequence = [''.join(seq) for seq in sequence]
  
  return header, sequence


def keep_genes_in_dict(header, sequence, allele=False) :
  new_header, new_sequence = [], []
  if allele :
    v_genes_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()
    d_genes_dict = np.load('data/dict/d_alleles_37.npy', allow_pickle='True').item()
    j_genes_dict = np.load('data/dict/j_alleles_11.npy', allow_pickle='True').item()
  else :
    v_genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()
    d_genes_dict = np.load('data/dict/d_genes_30.npy', allow_pickle='True').item()
    j_genes_dict = np.load('data/dict/j_genes_6.npy', allow_pickle='True').item()
  # keep V, D, J genes that are in the dict
  for i in range(len(header)) :
    v_gene_allele = header[i].split('|')[1]
    d_gene_allele = header[i].split('|')[2]
    j_gene_allele = header[i].split('|')[3]
    if allele :
      v_gene = v_gene_allele
      d_gene = d_gene_allele
      j_gene = j_gene_allele
    else : 
      v_gene = v_gene_allele.split('*')[0]
      d_gene = d_gene_allele.split('*')[0]
      j_gene = j_gene_allele.split('*')[0]
    if v_gene in v_genes_dict.keys() and d_gene in d_genes_dict.keys() and j_gene in j_genes_dict.keys() :
      new_header.append(header[i])
      new_sequence.append(sequence[i])
  return new_header, new_sequence

  
def keep_genes_in_dict_v2(header, sequence, allele=False) :
  new_header, new_sequence = [], []
  v_genes_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()
  d_genes_dict = np.load('data/dict/d_alleles_37.npy', allow_pickle='True').item()
  j_genes_dict = np.load('data/dict/j_alleles_11.npy', allow_pickle='True').item()

  # keep V, D, J genes that are in the allele dict (we want same alleles on test and train)
  for i in range(len(header)) :
    v_gene_allele = header[i].split('|')[1]
    d_gene_allele = header[i].split('|')[2]
    j_gene_allele = header[i].split('|')[3]

    if v_gene_allele in v_genes_dict.keys() and d_gene_allele in d_genes_dict.keys() and j_gene_allele in j_genes_dict.keys() :
      new_header.append(header[i])
      new_sequence.append(sequence[i])

  return new_header, new_sequence


def parse_name(name, allele=False):
  V, D, J = name.split("|")[1], name.split("|")[2], name.split("|")[3]
  #V = int(V[4])
  #D = int(D[4])
  #J = int(J[4])

  if allele :
    v_genes_dict = np.load('data/dict/v_alleles_276.npy', allow_pickle='True').item()
    d_genes_dict = np.load('data/dict/d_alleles_37.npy', allow_pickle='True').item()
    j_genes_dict = np.load('data/dict/j_alleles_11.npy', allow_pickle='True').item()
    
    V = v_genes_dict[V]
    D = d_genes_dict[D]
    J = j_genes_dict[J]
  else :
    v_genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()
    d_genes_dict = np.load('data/dict/d_genes_30.npy', allow_pickle='True').item()
    j_genes_dict = np.load('data/dict/j_genes_6.npy', allow_pickle='True').item()

    V = v_genes_dict[V.split('*')[0]]
    D = d_genes_dict[D.split('*')[0]]
    J =  j_genes_dict[J.split('*')[0]]

  return V, D, J

def preprocess(data_path, nb_seq_max, allele, max_len, type='V') :
    # preprocessing
    print('Preprocess data...')
    headers, sequences = parse_fasta(data_path, stop=nb_seq_max)
    headers, sequences = keep_genes_in_dict_v2(headers, sequences, allele=allele)

    # Encoding sequences
    integer_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(categories='auto')
    input_features = []
    input_labels = []

    if max_len != 'max' :
        max_len = int(max_len)
    else :
        max_len = np.max([len(seq) for seq in sequences])
    V, D, J = [], [], []
    for seq, name in zip(sequences, headers) :
        if 'N' not in list(seq) :
            # sequences
            integer_encoded = integer_encoder.fit_transform(list(seq))
            integer_encoded = np.array(integer_encoded).reshape(-1, 1)
            one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
            if max_len-len(seq) >= 0 :
              one_hot_encoded = np.pad(one_hot_encoded.toarray(), [(0, max_len-len(seq)), (0,0)], mode='constant')
            else : 
              one_hot_encoded = one_hot_encoded.toarray()
              one_hot_encoded = one_hot_encoded[:max_len, :]
            one_hot_encoded = np.expand_dims(one_hot_encoded, axis=-1)
            input_features.append(one_hot_encoded)

            # classes
            v, d, j = parse_name(name, allele=allele)
            V.append(v)
            D.append(d)
            J.append(j)
            
    input_features = np.stack(input_features)
    if type == 'V' :
      input_labels = np.array(V)
    elif type == 'D' :
      input_labels = np.array(D)
    elif type == 'J' :
      input_labels = np.array(J)
    # Encoding classes
    """one_hot_encoder = OneHotEncoder(categories='auto')
    V = np.array(V).reshape(-1,1)
    input_labels = one_hot_encoder.fit_transform(V).toarray()"""

    return input_features, input_labels



def reverse_one_hot(seq) :
  nt_list = ['A', 'C', 'G', 'T']
  str_seq = []
  for i in range(seq.shape[0]) :
    if list(np.where(seq[i]==1.)[0]) != [] :
      str_seq.append(nt_list[np.where(seq[i]==1.)[0][0]])
  str_seq = ''.join(str_seq)
  return str_seq
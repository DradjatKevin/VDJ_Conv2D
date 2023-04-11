import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import argparse 
from src.data_vdj_cnn_conv2D import *
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from Bio import pairwise2
from Bio.Seq import Seq
import functools

def main() :
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument(
        "--train_dir", 
        type=str, 
        default=None, 
        help="Train data dir. "
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Batch size"
    )
    parser.add_argument(
        "--epoch", 
        type=int, 
        default=10, 
        help="Number of epoch"
    )
    parser.add_argument(
        "--allele", 
        action="store_true", 
        help="Whether to consider allele or not"
    )
    parser.add_argument(
        "--nb_classes", 
        type=int, 
        default=76, 
        help="Number of classes (output nodes)"
    )
    parser.add_argument(
        "--nb_seq_max", 
        type=int, 
        default=10000, 
        help="Number of maximum sequence to consider"
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=6, 
        help="size of the kernel. Should be a tuple."
    )
    parser.add_argument(
        "--filters", 
        type=int, 
        default=64, 
        help="Number of filters for Conv2D"
    )
    parser.add_argument(
        "--padding",  
        default='max', 
        help="apply specific padding"
    )
    parser.add_argument(
        "--wandb", 
        action="store_true", 
        help="Follow run on wandb"
    )
    parser.add_argument(
        "--type", 
        type=str, 
        default='V' ,
        help="Which type of gene to identify : V, D or J"
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.5, 
        help="Drouput rate"
    )
    parser.add_argument(
        "--pool_size", 
        type=int, 
        default=2, 
        help="Pool size"
    )
    parser.add_argument(
        "--verbose", 
        type=int, 
        default=0, 
        help="Define keras loading bars"
    )
    parser.add_argument(
        "--error_file", 
        type=str, 
        default="misclassification.txt", 
        help="Name of the misclassification report"
    )
    parser.add_argument(
        "--test_data", 
        type=str,
        default='igscueal'
    )
    parser.add_argument(
        "--draw", 
        action="store_true",
        help="make acc and loss figures"
    )
    parser.add_argument(
        "--draw_file",
        type=str,
        default="model",
        help="name of the output loss and accuracy files"
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Apply alignment to precise alignment"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the model at the end of the training"
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default="models/model"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="prob threshold for alignment"
    )
    args = parser.parse_args()

    # run wandb
    if args.wandb :
        import wandb
        from wandb.keras import WandbMetricsLogger
        wandb.init(
            project='vdj_detection_cnn', 
            config={"architecture":"CNN",
                    "epochs":args.epoch, 
                    "dataset":args.train_dir,
                    "kernel_size":args.kernel_size, 
                    "filters":args.filters}
        )

    # import data
    data_path = args.train_dir

    #data_test_path1 = 'data/airrship/all_alleles_shmflat_notrim_0mut.fasta'
    #data_test_path2 = 'data/airrship/all_alleles_shmflat_notrim_5mut.fasta'
    #data_test_path3 = 'data/airrship/all_alleles_shmflat_notrim_10mut.fasta'
    #data_test_path4 = 'data/airrship/all_alleles_shmflat_notrim_20mut.fasta'
    #data_test_path5 = 'data/airrship/all_alleles_shmflat_notrim_40mut.fasta'
    #data_test_path6 = 'data/airrship/all_alleles_shmflat_notrim_80mut.fasta'

    if args.test_data == 'AS' :
        data_test_path1 = 'data/airrship/all_alleles_shmflat_indels_0mut.fasta'
        data_test_path2 = 'data/airrship/all_alleles_shmflat_indels_5mut.fasta'
        data_test_path3 = 'data/airrship/all_alleles_shmflat_indels_10mut.fasta'
        data_test_path4 = 'data/airrship/all_alleles_shmflat_indels_20mut.fasta'
        data_test_path5 = 'data/airrship/all_alleles_shmflat_indels_40mut.fasta'
        data_test_path6 = 'data/airrship/all_alleles_shmflat_indels_80mut.fasta'
    else :
        data_test_path1 = 'data/nika_data/simple_plus_indels.fasta'
        data_test_path2 = 'data/nika_data/simple_plus_indels_5Mut_out.fasta'
        data_test_path3 = 'data/nika_data/simple_plus_indels_10Mut_out.fasta'
        data_test_path4 = 'data/nika_data/simple_plus_indels_20Mut_out.fasta'
        data_test_path5 = 'data/nika_data/simple_plus_indels_40Mut_out.fasta'
        data_test_path6 = 'data/nika_data/simple_plus_indels_80Mut_out.fasta'

    input_features, input_labels = preprocess(data_path, args.nb_seq_max, args.allele, args.padding, type=args.type)
    
    input_features_test_0mut, input_labels_test_0mut = preprocess(data_test_path1, args.nb_seq_max, args.allele, max_len=input_features.shape[1], type=args.type)
    input_features_test_5mut, input_labels_test_5mut = preprocess(data_test_path2, args.nb_seq_max, args.allele, max_len=input_features.shape[1], type=args.type)
    input_features_test_10mut, input_labels_test_10mut = preprocess(data_test_path3, args.nb_seq_max, args.allele, max_len=input_features.shape[1], type=args.type)
    input_features_test_20mut, input_labels_test_20mut = preprocess(data_test_path4, args.nb_seq_max, args.allele, max_len=input_features.shape[1], type=args.type)
    input_features_test_40mut, input_labels_test_40mut = preprocess(data_test_path5, args.nb_seq_max, args.allele, max_len=input_features.shape[1], type=args.type)
    input_features_test_80mut, input_labels_test_80mut = preprocess(data_test_path6, args.nb_seq_max, args.allele, max_len=input_features.shape[1], type=args.type)
    
    # train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(input_features, input_labels, test_size = 0.2, random_state=42)
    print(X_train.shape)
    #print(Y_train.shape)


    # Define the model 
    model = Sequential() 
    
    #model.add(Conv1D(filters=args.filters, kernel_size=args.kernel_size, input_shape=(X_train.shape[1], 4), padding='same', activation='relu'))
    model.add(Conv2D(filters=args.filters, kernel_size=(args.kernel_size, 4), input_shape=(X_train.shape[1], 4, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=args.pool_size))
    #model.add(Conv1D(filters=2*args.filters, kernel_size=args.kernel_size, padding='same', activation='relu'))
    model.add(Conv2D(filters=2*args.filters, kernel_size=(args.kernel_size, 4), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=args.pool_size))
    model.add(Dropout(args.dropout))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(args.dropout))
    model.add(Dense(args.nb_classes, activation='softmax'))

    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    top3_acc = functools.partial(keras.metrics.sparse_top_k_categorical_accuracy, k=2)
    top3_acc.__name__ = 'top2_acc'
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', top3_acc])
    model.summary()


    # Train
    if args.wandb :
        history = model.fit(X_train, Y_train, epochs=args.epoch, validation_split=0.2, callbacks=[WandbMetricsLogger("epoch")], verbose=args.verbose)
    else :
        history = model.fit(X_train, Y_train, epochs=args.epoch, validation_split=0.2, verbose=args.verbose)

    if args.save_model :
        model.save(args.save_file)

    # Test
    print('Test set :')
    model.evaluate(X_test, Y_test)
    print('Other simulator dataset :')
    results0 = model.evaluate(input_features_test_0mut, input_labels_test_0mut)
    results5 = model.evaluate(input_features_test_5mut, input_labels_test_5mut)
    results10 = model.evaluate(input_features_test_10mut, input_labels_test_10mut)
    results20 = model.evaluate(input_features_test_20mut, input_labels_test_20mut)
    results40 = model.evaluate(input_features_test_40mut, input_labels_test_40mut)
    results80 = model.evaluate(input_features_test_80mut, input_labels_test_80mut)

    # misclassification report
    # import genes_dict
    if args.type == 'V' :
        genes_dict = np.load('data/dict/v_genes_75.npy', allow_pickle='True').item()
    elif args.type == 'D' :
        genes_dict = np.load('data/dict/d_genes_30.npy', allow_pickle='True').item()
    else :
        genes_dict = np.load('data/dict/j_genes_6.npy', allow_pickle='True').item()

    # get predictions for 0mut    
    predictions = model.predict(input_features_test_0mut)
    with open(args.error_file, 'w') as file :
        c = 0
        file.write('misclassification\tpredicted_class\ttrue_class\n')
        for i in range(len(predictions)) :
            if np.argmax(predictions[i]) != input_labels_test_0mut[i] :
                #print(predictions[i], np.argmax(predictions[i]), predictions[i][np.argpartition(predictions[i],-2)[-2:][0]]-predictions[i][np.argpartition(predictions[i],-2)[-2:][1]])
                pred_class = [k for k, v in genes_dict.items() if v == np.argmax(predictions[i])]
                true_class = [k for k, v in genes_dict.items() if v == input_labels_test_0mut[i]]
                file.write(f'{c}\t{pred_class[0]}\t{true_class[0]}\n')
                c += 1
    #print(predictions[122], np.argmax(predictions[122]), predictions[122][np.argpartition(predictions[122],-2)[-2:][0]]-predictions[122][np.argpartition(predictions[122],-2)[-2:][1]])
    
    # classifications with alignment
    if args.align :
        # load sequences dict
        if args.allele :
            seq_dict = np.load('data/v_alleles_seq.npy', allow_pickle='True').item()
        else : 
            seq_dict = np.load('data/v_genes_seq.npy', allow_pickle='True').item()
        predictions = model.predict(input_features_test_0mut)
        new_pred = []
        for vect, onehot in zip(predictions, input_features_test_0mut) :
            seq = reverse_one_hot(onehot)
            ind = np.argpartition(vect, -3)[-3:]
            # condition : consider only top-2
            ind_cond = np.argpartition(vect, -2)[-2:]
            if np.abs(vect[ind_cond[0]] - vect[ind_cond[1]]) < args.threshold :
                # get gene names
                fam = []
                for i in ind :
                    fam.append([[k for k, v in v_genes_dict.items() if v == i]][0][0])
                # convert to sequences
                #print(fam)
                for i in range(len(fam)) :
                    fam[i] = Seq(seq_dict[fam[i]].upper())
                # convert to Seq
                seq = Seq(seq)
                # align 
                scores = []
                for i in range(len(fam)) :
                    #print(i)
                    alignment = pairwise2.align.localxx(seq, fam[i])
                    #print(alignment)
                    scores.append(alignment[0][2])
                # new prediction
                name = [k for k, v in seq_dict.items() if v == fam[int(np.argmax(scores))].lower()][0]
                new_pred.append(v_genes_dict[name])
            else :
                new_pred.append(np.argmax(vect))
        # accuracy 
        s = 0
        for i, j in zip(new_pred, input_labels_test_0mut) :
            if i == j :
                s += 1
        s = s / len(new_pred)
        print(f'Accuracy with alignment : {s}')


    if args.wandb :
        wandb.log({"acc_0mut":results0[1], 
                "acc_5mut":results5[1], 
                "acc_10mut":results10[1], 
                "acc_20mut":results20[1], 
                "acc_40mut":results40[1], 
                "acc_80mut":results80[1]})
        wandb.finish()
    
    if args.draw :
        # accuracy
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy', fontsize=15)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(args.draw_file+'_acc.png')
        # accuracy
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss', fontsize=15)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(args.draw_file+'_loss.png')


if __name__ == "__main__" :
    main()




# python vdj_cnn.py --train_dir 'data/airrship/all_alleles_expMut.fasta' --epoch 30 --batch_size 32 --nb_classes 76 


"""
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=21, input_shape=(train_features.shape[1], 4), 
                 padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=200, kernel_size=21, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
"""
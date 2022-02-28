# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike


# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """

    ## A will be encoded in 0th column of matrix, T in 1th column, etc. This matrix will then be flattened
    AA_map = {
        "A": 0,
        "T": 1,
        "C": 2,
        "G": 3
    }
    one_hot_encodings = []
    #loop through each sequence in seq_arr and produce a flattened one-hot-encoding
    for seq in seq_arr:

        shape = (len(seq), 4)
        one_hot_matrix = np.zeros(shape)  # initialize matrix filled with 0s. 4 columns and one row per nucleotide in the sequence
        for row, nucleotide in enumerate(seq):
            one_hot_matrix[row, AA_map[nucleotide]] = 1
        one_hot_encodings.append(one_hot_matrix.flatten())
    return one_hot_encodings


def sample_seqs(
        seqs: List[str]
        labels: List[bool]) -> Tuple[List[seq], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """



        pos_class_indices = np.where(labels)
        neg_class_indices = np.where(~np.array(labels))
        desired_class_size = abs(len(labels) - sum(labels))

    #if True/positive class is the minority class, upsample this class (i.e, with replacement) to compensate for imbalance
    #Otherwise, sample the negative class with replacement
    if sum(labels) / len(labels) <= 0.5:
        upsampled_indices = np.random.choice(a = pos_class_indices,size = desired_class_size,replace=True) #sample pos class with replacement
        downsampled_indices = np.random.choice(a = neg_class_indices,size = desired_class_size,replace=False) #downsample neg class

    else:
        upsampled_indices = np.random.choice(a = neg_class_indices,size = desired_class_size,replace=True) #sample neg class with replacement
        downsampled_indices = np.random.choice(a = pos_class_indices,size = desired_class_size,replace=False) #downsample pos class

    sampled_indices = list(upsampled_indices) + list(downsampled_indices)
    return seqs[sampled_indices],labels[sampled_indices]
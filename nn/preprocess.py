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


def clip_sample_seqs(
        pos_seqs: List[str],
        neg_seqs: List[str])-> Tuple[List[str], List[str]]:
    """
    Takes two lists of sequences, where the sequences in list A are longer
    than those in list B. This function will clip the sequences in list A to more smaller sequences
    of the same size as in list B
    Args:
        pos_seqs: List[str]
            List of sequences from the positive class
        neg_seqs: List[str]
            List of sequences from the negative class

    Returns:
        pos_seqs: List[str]
            List of sequences from the positive class (each seq is of same length as neg_seqs)
        neg_seqs: List[str]
            List of sequences from the negative class (each seq is of same length as pos_seqs)

    """

    len_pos_seqs = len(pos_seqs[0]) #return length of first element in pos_seqs. Assume rest of elements are same length
    len_neg_seqs = len(neg_seqs[0]) #return length of first element in neg_seqs. Assume rest of elements are same length


    if len_pos_seqs < len_neg_seqs:
        clipped_neg_seqs = []
        for neg_seq in neg_seqs:
            clipped_seq = [neg_seq[i:i + len_pos_seqs] for i in range(0, len(neg_seq), len_pos_seqs)]
            clipped_neg_seqs.append([x for x in clipped_seq if len(x) ==len_pos_seqs ])
        flat_clipped_neg_seqs = [item for sublist in clipped_neg_seqs for item in sublist] #flatten list
        return pos_seqs, flat_clipped_neg_seqs
    else:
        clipped_pos_seqs = []
        for pos_seq in pos_seqs:
            clipped_seq = [pos_seq[i:i + len_neg_seqs] for i in range(0, len(pos_seq), len_neg_seqs)]
            clipped_pos_seqs.append([x for x in clipped_seq if len(x) == len_neg_seqs])
        flat_clipped_pos_seqs = [item for sublist in clipped_pos_seqs for item in sublist]  # flatten list
        return flat_clipped_pos_seqs, neg_seqs

def sample_seqs(
        seqs: List[str],
        labels: List[bool],
        random_state: int) -> Tuple[List[str], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels
        random_state: int
            Numpy random seed to be used

    Returns:
        sampled_seqs: ArrayLike
            List of sampled sequences which reflect a balanced class size
        sampled_labels: Array[bool]
            List of labels for the sampled sequences
    """
    #get

    np.random.seed(random_state)
    pos_class_indices = np.where(labels)[0]
    neg_class_indices = np.where(~np.array(labels))[0]
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
    sampled_seqs = [seqs[i] for i in sampled_indices] #list of seqs after upsampling
    sampled_seqs = np.stack(sampled_seqs,axis=0)  #convert list of np arrays back into a 2D array

    sampled_labels = [labels[i] for i in sampled_indices] #list of corresponding labels after upsampling
    sampled_labels = np.array(sampled_labels)
    return sampled_seqs,sampled_labels
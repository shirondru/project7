# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
import pytest
from nn import NeuralNetwork, one_hot_encode_seqs, sample_seqs,clip_sample_seqs
from sklearn.model_selection import train_test_split

# TODO: Write your test functions and associated docstrings below.

def test_forward():
    """
    Beacuse a random seed has been assigned, the weight and bias initialization should be reproducible
    And therefore the results of each individual forward pass should be reproducible. Test that is the case by
    comparing the activation matrix generated from each layer by the class object to what I calculated manually.

    """
    np.random.seed(42)
    nn = NeuralNetwork(nn_arch=[{'input_dim': 10, 'output_dim': 5, 'activation': 'sigmoid'},
                                {'input_dim': 5, 'output_dim': 3, 'activation': 'relu'},
                                {'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'}],
                       lr=0.01, batch_size=10, seed=42, epochs=5, loss_function=
                       'mean_squared_error')

    X = np.random.randn(5, 10)  # 5 observations, 10 features
    y = np.eye(1)[np.random.choice(1, 5)]  # one hot encoded 1 dim y vector with 5 obs

    ## compare result of each Activation Matrix with manually generated outcome
    output, cache = nn.forward(X)
    expected_activation_matrices = { 'A1': np.array([[0.43547758, 0.44119264, 0.48349452, 0.55064977, 0.42868517],
                                                [0.52517522, 0.48727809, 0.50595399, 0.59094964, 0.47200604],
                                                [0.44440897, 0.51979499, 0.5489964 , 0.55032532, 0.48897861],
                                                [0.52444771, 0.50747399, 0.47981625, 0.58426236, 0.51532172],
                                                [0.44602643, 0.50216959, 0.44358769, 0.3911953 , 0.60201282]]),

                                     'A2': np.array([[0.08478   , 0.04021755, 0.09307999],
                                            [0.09413245, 0.03127832, 0.10582984],
                                            [0.08286107, 0.03602137, 0.09639724],
                                            [0.09718215, 0.03814938, 0.09992652],
                                            [0.09350582, 0.07615239, 0.07311864]]),
                                     'A3': np.array([[0.50477087],
                                            [0.505984  ],
                                            [0.50503877],
                                            [0.50553204],
                                            [0.50234866]])}

    #assert the activation matrices generated by class object is the same as what I calculated manually
    for activation_matrix in expected_activation_matrices.keys():
        assert np.allclose(expected_activation_matrices[activation_matrix],cache[activation_matrix]), f"{activation_matrix} Activation Matrix is different than expected!"




def test_single_forward():
    """
    Beacuse a random seed has been assigned, the weight and bias initialization should be reproducible
    And therefore the results of a single forward pass should be reproducible. Test that is the case by
    comparing results of NN single forward pass to manual results


    """
    np.random.seed(42)
    ### initialize NN and small X and y matrices
    nn = NeuralNetwork(nn_arch=[{'input_dim': 4, 'output_dim': 1, 'activation': 'sigmoid'}],
                       lr=0.01, batch_size=10, seed=42, epochs=5, loss_function=
                       'cross_entropy')

    X = np.random.randn(3, 4) # 3 observations, 4 features
    A, Z = nn._single_forward(W_curr=nn._param_dict['W1'],
                              b_curr=nn._param_dict['b1'],
                              A_prev=X,
                              activation=nn.arch[0]['activation'])

    manual_Z = np.array([[-0.07867661],
                         [ 0.01662859],
                         [-0.28527714]])
    manual_A = np.array([[0.48034099],
                         [0.50415705],
                         [0.42916049]])

    assert np.allclose(A,manual_A),"Activation Matrix from single forward pass different from expected!"
    assert np.allclose(Z, manual_Z), "Z Matrix from single forward pass different from expected!"


def test_forward_dimensions():
    """
    This test takes the matrices produced during a forward pass and tests the dimensions of the output matrix from each layer
    is the same as defined when the matrix was initialized. This also tests the dimensions of y_hat are identical to y_true

    """
    nn = NeuralNetwork(nn_arch=[{'input_dim': 64, 'output_dim': 49, 'activation': 'sigmoid'},
                                {'input_dim':49,'output_dim':32,'activation':'relu'},
                                {'input_dim':32,'output_dim':1,'activation':'sigmoid'}],
                       lr=0.01, batch_size=10, seed=42, epochs=5, loss_function=
                       'cross_entropy')
    X = np.random.randn(100, 64) #instantiate 100 observations with 64 features
    y = np.eye(1)[np.random.choice(1, 100)] #instantiate 100 observations with 1 feature
    output, cache = nn.forward(X)
    for idx in range(len(nn.arch)):
        layer_idx = idx + 1
        expected_output_dim = nn.arch[idx]['output_dim'] #the number of neurons expected after a forward pass through this layer
        true_output_dim = cache['Z' + str(layer_idx)].shape[1] #the number of neurons/features after a forward pass through this layer
        assert expected_output_dim == true_output_dim, f"Dimensions of linear transformed matrix in layer {layer_idx} is wrong!"
    assert output.shape == y.shape, "Dimensions of y_hat are different from y_true"
def test_single_backprop():
    """
    Test single_backprop by comparing this functions output to manually calculated results
    for a very simple case
    """
    np.random.seed(42)
    X = np.array([[1,1,1],
                  [2,2,2],
                  [3,3,3]])

    nn = NeuralNetwork(nn_arch=[{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                        {'input_dim': 2, 'output_dim': 3, 'activation': 'relu'}],
                       lr=0.001, batch_size=1, seed=42, epochs=25,
                       loss_function='mean_squared_error')
    #set weights and bias manually to easy values to make hand calculation easy
    nn._param_dict = {'W1': np.array([[0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5]]),
                        'b1': np.array([[1.0],
                                        [1.0]]),
                        'W2': np.array([[0.5, 0.5],
                                        [0.5, 0.5],
                                        [0.5, 0.5]]),
                        'b2': np.array([[1.0],
                                        [1.0],
                                        [1.0]])}
    #get necessary values for calculating single_backprop.
    output, cache = nn.forward(X)
    Z_curr = cache["Z2"]
    dZdW = cache["A1"]
    dAdZ = nn._activation_function_backprop(Z_curr, 'relu')
    dJdA = nn._loss_function_backprop(X, output)
    delta = np.multiply(dJdA, dAdZ)

    #manually calculate the necessary arrays for calculating single_backprop
    expected_dAdZ = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]])

    expected_dJdA = np.array([[0.27777778, 0.27777778, 0.27777778],
                               [0.33333333, 0.33333333, 0.33333333],
                               [0.38888889, 0.38888889, 0.38888889]])
    expected_delta = np.array([[0.27777778, 0.27777778, 0.27777778],
                           [0.33333333, 0.33333333, 0.33333333],
                           [0.38888889, 0.38888889, 0.38888889]])
    assert np.allclose(expected_dAdZ,dAdZ),"dAdZ different than expected"
    assert np.allclose(expected_dJdA,dJdA),"dJdA different than expected"
    assert np.allclose(expected_delta,delta),"delta different than expected"

    # manually calculated partial derivatives with respect to loss for weights and biases in final layer
    expected_dW = np.array([[4.16666667, 4.16666667],
                           [4.16666667, 4.16666667],
                           [4.16666667, 4.16666667]])
    expected_dB = np.array([[1.],
                           [1.],
                           [1.]])
    #partial derivatives with respect to loss for weights and biases in final layer generated by NN
    dW, dB = nn._single_backprop(dZdW, delta)

    assert np.allclose(expected_dW,dW) and np.allclose(expected_dB,dB), "Backpropagation from the final layer did not lead to correct gradients"
def test_predict():
    """
    The predict method calls forward, which has been tested separately, and returns the predictions
    Since the forward method has already been tested, the calculations don't need to be tested again.
    Instead I test that predict returns a single array of the correct shape
    """
    nn = NeuralNetwork(nn_arch=[{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
                                {'input_dim': 16, 'output_dim': 1, 'activation': 'sigmoid'}],
                       lr=0.01, batch_size=20, seed=42, epochs=50, loss_function=
                       'cross_entropy')
    X = np.random.randn(1000, 64)
    y = np.random.randint(low=0, high=2, size=1000)  # one hot encoded 8 dim y vector with 10000 obs
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    train_loss, _ = nn.fit(X_train, y_train, X_test, y_test)  # get mean training loss per training epoch

    preds = nn.predict(X)
    assert preds.shape[0] == y.shape[0], "predict method return an array of unexpected size"


def test_binary_cross_entropy():
    """
    Test binary_cross_entropy is working as expected by comparing results of a simple case
    to manual calculation
    """
    nn = NeuralNetwork(nn_arch=[{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                       {'input_dim': 2, 'output_dim': 1, 'activation': 'relu'}],
                            lr = 0.001, batch_size = 1, seed = 42, epochs = 25,
                            loss_function = 'cross_entropy')
    y_hat = np.array([0.6, 0.3, 0.4])
    y = np.array([1, 0, 0])

    expected_bce = 0.4594420638235713 #manually calculated for this simple case
    real_bce = nn._binary_cross_entropy(y,y_hat)
    assert np.allclose(real_bce,expected_bce), "Binary Cross entropy different than expected"


def test_binary_cross_entropy_backprop():
    """
    Test binary_cross_entropy_backprop is working as expected by comparing results of a simple case
    to manual calculation
    """
    #instantiate arbitrary NN to test _binary_cross_entropy_backprop
    nn = NeuralNetwork(nn_arch=[{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                                {'input_dim': 2, 'output_dim': 1, 'activation': 'relu'}],
                       lr=0.001, batch_size=1, seed=42, epochs=25,
                       loss_function='cross_entropy')
    y_hat = np.array([0.6, 0.3, 0.4])
    y = np.array([1, 0, 0])

    expected = np.array([-1.66666667,  1.42857143,  1.66666667]) #from manual calculation
    real = nn._binary_cross_entropy_backprop(y,y_hat)
    assert np.allclose(expected,real),"Binary cross entropy derivative different than expected"


def test_mean_squared_error():
    """
    Test mean squared error method returns same value as manually calculated for a simple case
    """
    #instantiate arbitrary NN to use its _mean_squared_error method
    nn = NeuralNetwork(nn_arch=[{'input_dim': 4, 'output_dim': 1, 'activation': 'sigmoid'}],
                       lr=0.01, batch_size=10, seed=42, epochs=5, loss_function=
                       'mean_squared_error')
    y = np.array([0.5,1.5,3.7])
    y_hat = np.array([1.0,3.1,1.2])

    class_MSE = nn._mean_squared_error(y,y_hat)
    expected_MSE = 1.51 #manually calculated

    assert class_MSE == expected_MSE, "MSE calcualted by NN different than expected!"
def test_mean_squared_error_backprop():
    # instantiate random NN in order to call it's _mean_squared_error method
    nn = NeuralNetwork(nn_arch=[{'input_dim': 68, 'output_dim': 16, 'activation': 'relu'},
                                {'input_dim': 16, 'output_dim': 1, 'activation': 'sigmoid'}],
                       lr=0.01, batch_size=1, seed=42, epochs=5, loss_function=
                       'mean_squared_error')

    y = np.array([0, 1, 2])
    y_pred = np.array([3, 4, 5])
    expected_output = np.array([1.0,1.0,1.0]) #manually calculated
    real_output = nn._mean_squared_error_backprop(y,y_pred)

    assert np.allclose(expected_output,real_output),"Mean Squared Error Derivative giving unexpected result!"
def test_one_hot_encode():
    """
    Test one hot encode func is working as expected by asserting it gives the right output
    for a very simple case
    """
    expected_output = np.array([1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.])
    actual_output = one_hot_encode_seqs(['AGA'])
    assert np.allclose(np.stack(actual_output,axis=0), expected_output), "one hot encode giving unexpected result"


def test_sample_seqs():
    """
    sample_seqs will upsample the minority class until it has been resampled with replacement
    until the length of the minority class equals the length of the majority class.
    Assert that is true here with a very simple test case.
    """
    labels = [True,False,False,False]
    seqs = ["ATGC","AAAA","TTTT","CCCC"]

    X, y = sample_seqs(seqs, labels, random_state=42)

    #sample_seqs will upsample the minority class 3x
    #the expected output will therefore have the single minority class element repeated 3x
    # and all elements of the majority class once.
    #assert that is the case
    expected_output = [seqs[0]] * 3 + seqs[1:]
    assert all(expected_output==X),"sample_seqs gave wrong result!"

def test_clip_sample_seqs():
    """
    Test clip_sample_seqs properly turns longer sequences from one class
    to smaller sequences of equal size to those in the other class with a simple case.
    Note that clip_sample_seqs assumes that all sequences from a given class are of the same length
    And this unit test does too
    """
    seq1 = ["ATGC"]
    seq2 = ["ATGCATGCAAAA"]

    expected_seq1 = ["ATGC"]
    expected_seq2 = ["ATGC","ATGC","AAAA"]

    clipped_seq1, clipped_seq2 = clip_sample_seqs(seq1,seq2)

    assert expected_seq1 == clipped_seq1 and expected_seq2 == clipped_seq2, "clip_sample_seqs returned different values than expected!"

def test_fit():
    """
    Because the loss is being minimized via gradient descent it is expected that the loss (with respect to the training data)
    will approach 0 as number of gradient descent iterations increases. Test that is the case by asserting the loss from the 1st third of the data
    is greater than the loss from the middle third of the data, which is greater than the loss from the last third of the data.

    """

    nn = NeuralNetwork(nn_arch=[{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
                                {'input_dim': 16, 'output_dim': 1, 'activation': 'sigmoid'}],
                       lr=0.01, batch_size=20, seed=42, epochs=50, loss_function=
                       'cross_entropy')
    X = np.random.randn(1000, 64)
    y = np.random.randint(low=0, high=2, size=1000)  # y vector with binary 0/1 values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    train_loss, _ = nn.fit(X_train, y_train, X_test, y_test) #get mean training loss per training epoch

    #test that training loss decreases over epochs during gradient descent
    early_losses = train_loss[0:(nn._batch_size // 3)]  # loss history from first 1/3 of data
    middle_losses = train_loss[(nn._batch_size // 3): 2 * (nn._batch_size // 3)]  # loss history from second 1/3 of data
    late_losses = train_loss[2 * (nn._batch_size // 3):]  # loss history from last 1/3 of data
    assert np.mean(early_losses) > np.mean(middle_losses) > np.mean(late_losses), "Training loss is not approaching 0!"

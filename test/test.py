# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
import pytest
from nn import NeuralNetwork, one_hot_encode_seqs, sample_seqs
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
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    """
    Test mean squared error method returns same value as manually calculated for a simple case
    """
    nn = NeuralNetwork(nn_arch=[{'input_dim': 4, 'output_dim': 1, 'activation': 'sigmoid'}],
                       lr=0.01, batch_size=10, seed=42, epochs=5, loss_function=
                       'mean_squared_error')
    y = np.array([0.5,1.5,3.7])
    y_hat = np.array([1.0,3.1,1.2])

    class_MSE = nn._mean_squared_error(y,y_hat)
    expected_MSE = 1.51

    assert class_MSE == expected_MSE, "MSE calcualted by NN different than expected!"


def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    pass


def test_sample_seqs():
    pass


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
    y = np.random.randint(low=0, high=2, size=1000)  # one hot encoded 8 dim y vector with 10000 obs
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    train_loss, _ = nn.fit(X_train, y_train, X_test, y_test) #get mean training loss per training epoch

    #test that training loss decreases over epochs during gradient descent
    early_losses = train_loss[0:(nn._batch_size // 3)]  # loss history from first 1/3 of data
    middle_losses = train_loss[(nn._batch_size // 3): 2 * (nn._batch_size // 3)]  # loss history from second 1/3 of data
    late_losses = train_loss[2 * (nn._batch_size // 3):]  # loss history from last 1/3 of data
    assert np.mean(early_losses) > np.mean(middle_losses) > np.mean(late_losses), "Training loss is not approaching 0!"

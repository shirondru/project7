# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32 dimension hidden layer
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function. If "cross_entropy" is in the string, binary cross entropy is implemented if the output is binary (i.e, if the y used during training is 1-dimensional). If the output has
            multiple classes (i.e, the y used during training is 2-dimensional) multi-class cross entropy is implemented, even if loss_function == "binary_cross_entropy" and vice-versa.
        reduction: str
            Either "mean" or "sum". The aggregation method for the loss function.

    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch: List[Dict[str, Union[int, str]]],
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 loss_function: str):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs

        assert loss_function == 'mean_squared_error' or "cross_entropy" in loss_function, "loss_function must be either 'mean_squared_error' or contain the string 'cross_entropy'"
        self._loss_func = loss_function

        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()


    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(self,
                        W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """


        Z_curr = A_prev.dot(W_curr.T) + b_curr.T
        A_curr = self._activation_function(Z_curr,activation)


        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing X,Z, and A matrices from `_single_forward` for use in backprop.
        """
        # initialize cache with X. X will be used to calculate dZ/dW1
        cache = {"X":X}
        for idx in range(len(self.arch)):
            layer_idx = idx + 1

            # use X as input layer
            if layer_idx == 1:
                A_prev = X
            else:
                A_prev = A_curr
            A_curr, Z_curr = self._single_forward(W_curr=self._param_dict['W' + str(layer_idx)],
                                                  b_curr=self._param_dict['b' + str(layer_idx)],
                                                  A_prev=A_prev,
                                                  activation=self.arch[idx]['activation'])
            cache['Z' + str(layer_idx)] = Z_curr
            cache['A' + str(layer_idx)] = A_curr

        # A_curr at the end of the for loop is output layer (y_hat)
        return A_curr, cache

    def _single_backprop(self,
                         dZdW: ArrayLike,
                         current_delta: ArrayLike,
                         ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            dZdW: ArrayLike
                Previous layer activation matrix. For the very first layer (between input and first hidden layer neurons),
                this value equal to the input X beacuse Z1 = W1T @X + b. Therefore dZ/dW1 = X
            current_delta: ArrayLike
                Backpropagation delta up to and including the current layer


        Returns:

            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        #partial derivative of current activation matrix with respect to current layer linear transform matrix. dAL/dZ


        dW_curr = current_delta.T.dot(dZdW)
        db_curr = np.sum(current_delta,axis = 0,keepdims=True).T


        return dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {}
        for idx in reversed(range(len(self.arch))):
            layer_idx = idx + 1
            Z_curr = cache["Z" + str(layer_idx)]

            # When backpropagating through the first layer (i.e the layer between the input and the first hidden layer neurons)
            # there is no A(layer=0) because the input matrix does not go through an activation function. Instead,
            #dZdW = X
            #
            if layer_idx > 1:
                dZdW = cache["A" + str(layer_idx - 1)]
            else:
                dZdW = cache['X']
            activation_curr = self.arch[idx]['activation']
            dAdZ = self._activation_function_backprop(Z_curr, activation_curr)

            #if on the final layer in the network, calculate dJdA and dAdZ
            if layer_idx == len(self.arch):
                dJdA = self._loss_function_backprop(y,y_hat)
                delta = np.multiply(dJdA,dAdZ)
                dW_curr, db_curr = self._single_backprop(dZdW,delta)

            else:

                #Get backpropagation delta for current layer by matrix multiplying
                #previously calculated delta with W from layer + 1, and element-wise multiplying result
                #with derivative of current activation matrix with respect to current linear transformed matrix
                W_LPlusOne = self._param_dict["W" + str(layer_idx + 1)]
                delta = np.multiply(delta.dot(W_LPlusOne),dAdZ)
                dW_curr, db_curr = self._single_backprop(dZdW, delta)
            grad_dict["dW" + str(layer_idx)] = dW_curr
            grad_dict["db" + str(layer_idx)] = db_curr

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """
        for param in self._param_dict.keys():
            self._param_dict[param] = self._param_dict[param] - self._lr * grad_dict["d" + param]

    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        assert self.arch[-1]['output_dim'] == y.shape[1],f"Number of output layer neurons does not equal number of classes implied by (i.e, number of columns in) y."
        num_batches = int(X_train.shape[0] / self._batch_size)
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        for epoch in range(self._epochs):

            #shuffle data for each epoch
            ind_list = np.array(list(range(X_train.shape[0])))
            np.random.shuffle(ind_list)
            X_train = X_train[ind_list,:]
            y_train = y_train[ind_list,:]

            #create array containing all batches
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)

            #Iterate through each batch until all batches in epoch
            batch_train_loss = []
            batch_val_loss = []
            for X_batch_train, y_batch_train in zip(X_batch, y_batch):
                y_hat, cache = self.forward(X_batch_train) #get prediction with training data from this batch
                batch_train_loss.append(self._loss_function(y_batch_train,y_hat)) #save training loss from this batch
                grad_dict = self.backprop(y_batch_train,y_hat,cache) #get partial derivative of loss with respect to each weight and bias term
                self._update_params(grad_dict) #update weights and biases


                y_hat, _ = self.forward(X_val) #get prediction with validation data
                batch_val_loss.append(self._loss_function(y_val, y_hat)) #sore validation loss

            #save mean train and val loss across batches for this epoch
            per_epoch_loss_train.append(np.mean(batch_train_loss))
            per_epoch_loss_val.append(np.mean(batch_val_loss))
        return per_epoch_loss_train, per_epoch_loss_val


    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)
        return y_hat


    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """


        return 1/(1+np.exp(-Z))

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0,Z)

    def _sigmoid_backprop(self, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return np.multiply(self._sigmoid(Z),(1-self._sigmoid(Z)))


    def _relu_backprop(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:

            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        conditions = [Z < 0, Z > 0]
        choicelist = [0, 1]
        return np.select(conditions, choicelist)

    def _activation_function_backprop(self, Z: ArrayLike, activation: str) -> ArrayLike:
        """
        Calls the correct activation function in accordance with self.arch['activation']

        Args:
            Z: ArrayLike
                Output of layer linear transform.
            activaton: str
                Name of activation function being used in this layer. Defined by self.arch['activation']

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """

        if activation.upper() == "SIGMOID":
            return self._sigmoid_backprop(Z)
        elif activation.upper() == "RELU":
            return self._relu_backprop(Z)

    def _activation_function(self, Z: ArrayLike, activation: str) -> ArrayLike:
        """
        Calls the correct activation function in accordance with self.arch['activation']

        Args:
            Z: ArrayLike
                Output of layer linear transform.
            activaton: str
                Name of activation function being used in this layer. Defined by self.arch['activation']

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        viable_activations = ["SIGMOID", "RELU"]
        assert activation.upper() in ["SIGMOID",
                                      "RELU"], f"Desired activation function {activation.upper()} must be one of {viable_activations}"
        if activation.upper() == "SIGMOID":
            return self._sigmoid(Z)
        elif activation.upper() == "RELU":
            return self._relu(Z)

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """

        #return mean (across observations) binary cross entropy loss
        return np.mean(-(y*np.log(y_hat) + ((1-y) * np.log(1-y_hat)))) #scalar

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """

        #not taking sum here as this derivative will be matrix multiplied to dZcurr/dWcurr
        #which takes care of the summation across observations
        return -(y*(1/y_hat) - ((1-y)*(1/(1-y_hat)))) #same shape as y or yhat

    def _multi_class_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Categorical cross entropy loss function, for when the number of classes >2.
        This expects labels to be provided in a one-hot representation

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """

        # sum the loss across each class to get one value per observation
        loss_vector = np.sum(-(y*np.log(y_hat) + ((1-y) * np.log(1-y_hat))),axis = 1) #dim: [batch_size,]
        return np.mean(loss_vector) #return mean loss across observations

    def _multi_class_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        multi class cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # not taking sum here as this derivative will be matrix multiplied to dZcurr/dWcurr
        # which takes care of the summation across observations
        return -(y*(1/y_hat) - ((1-y)*(1/(1-y_hat)))) #same shape as y or yhat


    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """

        #multiply by 0.5 so derivative is simpler
        return 0.5* np.mean((y_hat - y)**2)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # not taking sum here as this derivative will be matrix multiplied to dZcurr/dWcurr
        # which takes care of the summation across observations
        #but dividing by len(y) so the summation via matrix multiplication becomes a mean
        return (y_hat - y) /len(y)

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Calls the correct loss function in accordance with self._loss_func

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """

        if "cross_entropy" in self._loss_func and y.shape[1] == 1:
            return self._binary_cross_entropy(y, y_hat)
        elif "cross_entropy" in self._loss_func and y.shape[1] > 1:
            return self._multi_class_cross_entropy(y, y_hat)
        elif "mean_squared_error" == self._loss_func:
            return self._mean_squared_error(y, y_hat)

    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Returns the partial derivative of the correct loss function with respect to the output layer matrix, in accordance
        with self._loss_func
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to output matrix.
        """
        if "cross_entropy" in self._loss_func and y.shape[1] == 1:
            return self._binary_cross_entropy_backprop(y, y_hat)
        elif "cross_entropy" in self._loss_func and y.shape[1] > 1:
            return self._multi_class_cross_entropy_backprop(y, y_hat)
        elif "mean_squared_error" == self._loss_func:
            return self._mean_squared_error_backprop(y, y_hat)

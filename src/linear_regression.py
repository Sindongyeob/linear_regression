import numpy as np
import random


class LinearRegression():
    def __init__(self):
        self.w = random.gauss(0.0, 1.0)
        self.b = random.gauss(0.0, 1.0)

    ## function predict()
    ## Input: x => the input variable
    ## Process : predicting the ground truth y by using linear regression model
    ## Return : pred => the predicted value
    def predict(self, x):
        pred = 0.0
        # write your function body here - begin

        # write your function body here - end
        return pred

    ## function SE()
    ## Input:
    ## x => the input variable
    ## y => the ground truth value
    ## Process : Calculating Square Error (SE) between the prediction and the ground truth
    ## Return : SE
    def SE(self, x, y):
        SE = 0.0 # Calculate Square error
        # write your function body here - begin

        # write your function body here - end
        return SE

    ## function gradient_of_SE()
    ## Input:
    # x => the input variable
    # y => the ground truth value
    # Process: calculate the gradient of SE for parameter w => grad_for_w
    #          calculate the gradient of SE for parameter b => grad_for_b
    # Return: [grad_for_w, grad_for_b]
    def gradient_of_SE(self, x, y):
        grad_for_w = 0.0
        grad_for_b = 0.0
        # write your function body here - begin

        # write your function body here - end
        return np.array([grad_for_w, grad_for_b])

    ## function update_params()
    ## Input:
    ## grad_for_w => Derivative of the Objective Function (SE) for parameter w
    ## grad_for_b => Derivative of the Objective Function (SE) for parameter b
    ## Process: update self.w and self.b using their gradients
    ## Return: None
    def update_params(self, grad_for_w, grad_for_b, alpha):

        return

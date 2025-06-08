
import sys
import numpy as np
from optweights.utils import fast_xtdx, get_p_dict, update_DRO_weights, round_p_dict, clip_p_dict_per_group, normalize_p_dict
from optweights.model import Model
from optweights.metrics import calc_loss_for_model, calc_worst_group_loss, calc_BCE
from optweights.weights import Weights
from sklearn.metrics import  mean_squared_error
from sklearn.linear_model import LogisticRegression
import copy
import time

class WeightSearcher():
    def __init__(self,  X_train, y_train, g_train, X_val, y_val, g_val, p_ood, sklearn_model=None,  GDRO=False, subsample_weights=False, k_subsamples=1, weight_rounding=4, p_min=10e-4, l1_penalty=0, l2_penalty=0):
        """

        Arguments:
            X_train: np.array of (n, d) shape, training data
            y_train: np.array of (n, k) shape, training labels
            g_train: np.array of (n, 1) shape, group labels for training
            X_val: np.array of (n, d) shape, validation data
            y_val: np.array of (n, k) shape, validation labels
            g_val: np.array of (n, 1) shape, group labels for validation
            p_ood: dict, probability of each group in the OOD data
            sklearn_model: sklearn model, if None, use LogisticRegression
            GDRO: bool, if True, use optimize the worst-group loss
            subsample_weights: bool, if True, use subsample weights 
            k_subsamples: int, number of subsamples
            weight_rounding: int, number of decimals to round the weights
            p_min: float, minimum value for the weights
            l1_penalty: float, l1 penalty
            l2_penalty: float, l2 penalty
        
        
        """

        # set the attributes
        if sklearn_model is None:

            # if l1_penalty is greater than 0, add the l1 penalty
            # check; if both l1 and l2 penalty are > 0, raise an error
            if l1_penalty>0 and l2_penalty>0:
                sys.exit('Both l1 and l2 penalty are greater than 0. Please choose only one penalty type')
            # if l1 penalty is greater than 0, add the l1 penalty
            elif l1_penalty>0:
                penalty_type = 'l1'
                penalty = l1_penalty
            # if l2 penalty is greater than 0, add the l2 penalty
            elif l2_penalty>0:
                penalty_type = 'l2'
                penalty = l2_penalty
            # if both are 0, no penalty
            else:
                penalty_type = None
                penalty = 1
            
            # create a standard logistic regression model
            model_param  = {'max_iter': 100,
                    'penalty': penalty_type,        
                    'C':1/penalty,                
                    'solver': 'liblinear',  # standard lib to solve
                    'tol': 1e-4,            # standard tolerance
                    'verbose': 0,
                    'random_state': 0,
                    'fit_intercept': True, 
                    'warm_start': False}
        
            # create model and add
            self.sklearn_model = LogisticRegression(**model_param)

        else:
            self.sklearn_model = sklearn_model
   

        # set attributes
        self.X_train = X_train
        self.y_train = y_train
        self.g_train = g_train
        self.X_val = X_val
        self.y_val = y_val
        self.g_val = g_val
        self.GDRO = GDRO
        self.weight_rounding = weight_rounding
        self.p_min = p_min
        self.subsample_weights = subsample_weights
        self.k_subsamples = k_subsamples
        self.g_prob = None
        

        # given the g_train, g_val, calculate the p_train, p_val
        self.p_train = get_p_dict(g_train)
        self.p_val = get_p_dict(g_val)
        self.p_ood = p_ood
        self.groups = list(self.p_train.keys())
        self.G = len(self.groups)

        # initialize the weights object for train, val
        self.weights_obj_tr = Weights(p_w=None, p_train=self.p_train, weighted_loss_weights=not subsample_weights)
        self.weights_obj_val = Weights(p_w=p_ood, p_train=self.p_val, weighted_loss_weights=True)

        # initialize the model object
        self.model = Model(weights_obj=self.weights_obj_tr, sklearn_model=self.sklearn_model, subsampler=self.subsample_weights, k_subsamples=k_subsamples)

        # check: are the groups in p_train the same as in p_val?
        if set(self.p_train.keys()) != set(self.p_val.keys()):
            KeyError('The groups in p_train are not the same as in p_val')

        # based on the model class, define the loss function
        # if sklearn.linear_model.LogisticRegression, use the log_loss
        if self.sklearn_model.__class__.__name__ == 'LogisticRegression':
            # from sklearn use log_loss
            self.loss = calc_BCE
        elif self.sklearn_model.__class__.__name__ == 'LinearRegression':
            self.loss = mean_squared_error
        else:
            sys.exit('The model class is not supported. Please use sklearn.linear_model.LogisticRegression or sklearn.linear_model.LinearRegression')

    def set_g_prob(self, g_prob):
        self.g_prob = g_prob

    def calc_Hessian_weighted_logistic_loss(self, X,  w, Beta, l1_penalty, l2_penalty, p , p_train, eps=1e-6, divide_by_n=True):
        """
        Calculate the Hessian of the logistic loss function

        Arguments:
            X: np.array of (n, d) shape, input data
            w: np.array of (n, 1) shape, weights
            Beta: np.array of (d+1, 1) shape, parameters
            l1_penalty: float, l1 penalty
            l2_penalty: float, l2 penalty
            eps: float, threshold for the l1 penalty
            divide_by_n: bool, if True, divide the Hessian by the number of samples
        
        Returns:
            H: np.array of (d+1, d+1) shape, Hessian matrix
        """

        # add the intercept to X, if the column dim of X is one less than the row dim of Beta
        if (X.shape[1]+1) == Beta.shape[0]:
            X =  np.c_[np.ones(X.shape[0]), X]
        
        # create a diagonal matrix with inputs sigmoid(x_i^T beta) * (1 - sigmoid(x_i^T beta))
        X_t_Beta = np.matmul(X, Beta).squeeze()
        sigmoid_X_t_Beta = 1/(1 + np.exp(-X_t_Beta))
        
        p_vec = np.array([p[g] for g in self.groups])  # shape (G,)
        p_train_vec = np.array([p_train[g] for g in self.groups])  # shape (G,)
        group_weight = p_vec / p_train_vec  # shape (G,)
        sample_weights = self.g_prob @ group_weight  # shape (n,)

        # Combine with external weights (e.g., DRO or subsample weighting)
        full_weights = sample_weights * w.squeeze()  # shape (n,)

        # calculate the diagonal matrix
        diag_H = ((sigmoid_X_t_Beta * (1 - sigmoid_X_t_Beta)) * full_weights) 
        
        # calculate the Hessian
        H = fast_xtdx(X, diag_H)

        # divide by the number of samples
        if divide_by_n:
            H /= X.shape[0]

        # add the l_2 penalty
        if l2_penalty>0:
            added_term = np.eye(H.shape[0])*l2_penalty *2

            # add the term
            if divide_by_n:
                H+= (added_term/X.shape[0])
            else:
                H += added_term

        if l1_penalty>0:
            # the following is an approximation of the derivative of the l_1 penalty
            beta_squared = (Beta**2).squeeze()
            H_l_1_approx =   eps/((beta_squared + eps)**(3/2))
            H_l_1_approx_diag = np.diag(H_l_1_approx)*l1_penalty
            added_term = H_l_1_approx_diag/X.shape[0]

            # add the term
            if divide_by_n:
                H += (added_term/X.shape[0])
            else:
                H += added_term

        return H

    

    #@classmethod
    def calc_grad_augmented_loss(self, X, Beta, y,  g, subsample_weights, eps=1e-6, m=None):
        """
        Calculate the gradient of the augmented loss, required for the gradient of the validation loss with respect to the weights

        Arguments:
            X: np.array of (n, d) shape, input data
            Beta: np.array of (d+1, 1) shape, parameters
            y: np.array of (n, 1) shape, labels
            g: np.array of (n, 1) shape, group labels
            subsample_weights: bool, if True, use subsample weights
            eps: float, threshold for the l1 penalty
            m: int, number of subsamples
        
        Returns:
            grad: np.array of (d+1, G) shape, gradient of the augmented loss
        """

        # first, calculate the gradient of the loss for each group
        groups = np.unique(g)

        # First, calculat the grad per group
        grad = np.zeros((X.shape[1]+1, len(groups)))

        # go over each group, except the last one
        for i, group in enumerate(groups):
        
            # calculate the gradient
            grad_group = self.calc_grad_BCE(X, Beta, y, 0, 0, eps=eps, w=self.g_prob[:, i])

            # add to the grad
            grad[:, i] = grad_group.squeeze()
        
        # if not subsample weights, deduct the grad of the last group from the other groups
        if not subsample_weights:
            # first, deduct the grad of the last group from the other groups
            grad_G = grad[:, -1].reshape(-1, 1)
            grad[:, :-1] -= grad_G
            
            # second, remove the last col. (the grad of the last group)
            grad = grad[:, :-1]
        else:
            # multiply the grad with the factor
            factor = X.shape[0]/m
            grad *= factor
        
        return grad
        

    
    @classmethod
    def calc_grad_BCE(self, X, Beta,  y, l1_penalty, l2_penalty, w=None, eps=1e-6, divide_by_n=True):
        """
        Calculate the gradient of the BCE

        Arguments:
            X: np.array of (n, d) shape, input data
            Beta: np.array of (d+1, 1) shape, parameters
            y: np.array of (n, 1) shape, labels
            l1_penalty: float, l1 penalty
            l2_penalty: float, l2 penalty
            w: np.array of (n, 1) shape, weights
            eps: float, threshold for the l1 penalty
            divide_by_n: bool, if True, divide the gradient by the number of samples
        
        Returns:
            grad: np.array of (d+1, 1) shape, gradient of the BCE
        """

        # add the intercept to X, if the column dim of X is one less than the row dim of Beta
        if (X.shape[1]+1) == Beta.shape[0]:
            X =  np.c_[np.ones(X.shape[0]), X]
        
        
        # create a diagonal matrix with inputs sigmoid(x_i^T beta) * (1 - sigmoid(x_i^T beta))
        X_t_Beta = np.matmul(X, Beta)
        sigmoid_X_t_Beta = 1/(1 + np.exp(-X_t_Beta))
               

        # calculate the gradient in two steps:
        # 1. first term: (sigmoid - y)
        # 2. second term: multiply X^T with the first term
        if w is not None:
            # check the shape of w
            if len(w.shape) == 1:
                w = w.reshape(-1, 1)
                
            # multiply element wise with w
            weighted_sigmoid_X_t_Beta = (sigmoid_X_t_Beta - y) * w
            grad =np.matmul(X.T, weighted_sigmoid_X_t_Beta)
        else:
            grad =np.matmul(X.T, (sigmoid_X_t_Beta - y))
        # divide by the number of samples
        if divide_by_n:
            grad /= X.shape[0]

        # add the l_2 penalty
        if l2_penalty>0:
            added_term = 2 * l2_penalty * Beta

            # add the term
            if divide_by_n:
                added_term /= X.shape[0]
            grad += added_term

        elif l1_penalty>0:
            # the following is an approximation of the derivative of the l_1 penalty
            beta_squared = (Beta**2)
            sqrt_beta_squared = np.sqrt(beta_squared + eps)
            added_term =  ((Beta / sqrt_beta_squared) * l1_penalty)
            
            # add the term
            if divide_by_n:
                added_term /= X.shape[0]
            grad += added_term

        return grad
    

    
    def weight_grad_via_ift(self, model, p, X_train, y_train, g_train, X_val, y_val, g_val, weights_obj_val, eps=1e-6,   subsample_weights=False):
        """
        Gets the gradient of the validation loss with respect to the weights, using the IFT

        Arguments:
            model: model object, model object
            p: dict, weights
            X_train: np.array of (n, d) shape, training data
            y_train: np.array of (n, k) shape, training labels
            g_train: np.array of (n, 1) shape, group labels for training
            X_val: np.array of (n, d) shape, validation data
            y_val: np.array of (n, k) shape, validation labels
            g_val: np.array of (n, 1) shape, group labels for validation
            weights_obj_val: weights object, weights object for the validation data
            eps: float, threshold for the l1 penalty
            subsample_weights: bool, if True, use subsample weights
        
        Returns:
            grad_ift_dict: dict, gradient of the validation loss with respect to the weights
        
        """

        # create a copy of the starting weights
        groups = list(p.keys())
        last_group = self.G

        # get the w_train, w_val
        w_train = model.weights_obj.assign_weights(g_train)
        w_val = weights_obj_val.assign_weights(g_val)

        # if the weights are subsampled, calculate the m - e.g. how many 
        if subsample_weights:
            self.m = model.m

        # calculate the hessian
        H = self.calc_Hessian_weighted_logistic_loss(X_train, w_train, model.Beta, model.l1_penalty, model.l2_penalty, p, self.p_train, eps=1e-6)


        # use multiplication factor if subsample_weights
        if subsample_weights:
            n_train = X_train.shape[0]
            factor = n_train/model.m
            H *= factor
    
        # ensure the Hessian is positive definite
        H += np.eye(H.shape[0])*eps

        # if the d > n, use the moores-penrose inverse
        if H.shape[0] > H.shape[1]:
            H_inv = np.linalg.pinv(H)
        else:
            H_inv =np.linalg.inv(H)

        # Calculate the gradient of the augmented loss with respect to the parameters
        J_augmented_w = self.calc_grad_augmented_loss(X_train, model.Beta, y_train, g_train, subsample_weights=subsample_weights, eps=1e-4, m=model.m)
       
        # third, calc the jacobian with respect to the weighted validation loss - take the average over the validation set
        J_val_w = self.calc_grad_BCE(X_val, model.Beta, y_val, 0, 0, w=w_val, eps=1e-4, divide_by_n=True)
        
        # calculate the derivative of the parameters with respect to w
        partial_deriv_param_w = np.matmul(-H_inv,  J_augmented_w)
       
        # now, calculate the derivative of the validation loss with respect to w
        grad_ift = np.matmul(J_val_w.T, partial_deriv_param_w)

        # now, calculate the derivative
        if subsample_weights:
            grad_ift = grad_ift.squeeze()
            grad_ift_dict =  {g: grad_ift[g-1].item() for g in groups}

        # for the last group, sum the changes in the other groups and taking the negative
        else:

            # squeeze the grad_ift
            grad_ift = grad_ift.squeeze()   

            # if single group, return the grad_ift
            if self.G-1 == 1:
                grad_ift_dict = {1: grad_ift.item()}
            else:
                grad_ift_dict = {g:grad_ift[g-1].item() for g in groups[:-1]}            

            # calculate the change for the last group
            grad_last_group = -np.sum([grad_ift_dict[g] for g in groups[:-1]]).item()

            # set the change for the last group based on change in all other groups
            grad_ift_dict[last_group] = grad_last_group
        
      

        return grad_ift_dict
    
    
    
    def return_weights(self, p_hat, g_train):
        """
        Function to return the weights based on the p_hat and the g_train

        Arguments:
            p_hat: dict, weights
            g_train: np.array of (n, 1) shape, group labels for training

        Returns:
            w_train: np.array of (n, 1) shape, weights
        """

        # use the self.weights_obj to return the weights
        # first, reset the weights using phat
        self.weights_obj_tr.reset_weights(p_hat)

        # then, return the weights based on the g_train
        w_train = self.weights_obj_tr.assign_weights(g_train)

        return w_train

      

    def optimize_weights(self, T,  lr,  momentum, start_p=None, eps=0, patience=None,  save_trajectory=False,  verbose=True,  eta_q=0.1, decay=0.9, lr_schedule='constant',stable_exp=True,   lock_in_p_g = None):
        """
        Optimize the weights using exponentiated gradient descent
        
        Arguments:
            T: int, number of iterations
            lr: float, learning rate
            momentum: float, momentum
            eps: float, threshold for the stopping criterion
            patience: int, number of iterations to wait for improvement
            save_trajectory: bool, if True, save the trajectory of the weights
            verbose: bool, if True, print the loss at each iteration
            eta_q: float,used for optimizing worst-group loss of the weights
            lr_schedule: str, learning rate schedule
            stable_exp: bool, if True, use stable exponentiation
            p_min: float, minimum value for the weights
            subsample_weights: bool, if True, use subsample weights
            lock_in_p_g: int, if not None, lock in the weights for group g
        
        Returns:
            best_p: dict, the optimized weights
        """

        # Check: if start_p is None, define it
        if start_p is None:
            start_p = self.p_ood

        # Check: are the groups in start_p the same as in p_train?
        if set(start_p.keys()) != set(self.p_train.keys()):
            KeyError('The groups in start_p are not the same as in p_train')

        # Check: are the entries in start_p floats?
        if not all(isinstance(value, float) for value in start_p.values()):
            TypeError('The values in start_p are not floats')
        
        # Initialize the gradient and the current p
        grad = dict.fromkeys(start_p, 999)
        p_t = round_p_dict(copy.deepcopy(start_p), self.weight_rounding)

        # if the trajectory is saved, initialize the trajectory
        if save_trajectory:

            # save the p trajectory
            p_at_t_traj = np.zeros((T, self.G))
            p_at_t_traj[0] = np.array(list(p_t.values()))
            
            # save the loss trajectory
            loss_at_t_traj = np.zeros(T-1)

        # check if momentum is not None
        if momentum is not None:
            prev_update = dict.fromkeys(self.groups, 0)
        
        # initialize the iteration
        t = 0
        best_loss = np.inf
        best_p = start_p
        stop_GD = False
        patience_count = patience
        
        # create weight obj. for initial weights, set the weights in the model
        self.model.reset_weights(p_t)
        self.model.fit(self.X_train, self.y_train, self.g_train)
 
         # if GDRO, save the p_at_t
        if self.GDRO:
            q_t = {g: 1/self.G for g in self.groups}
            best_worst_group_loss = np.inf


        # start the gradient descent
        while not stop_GD and (t < T):

            # if GDRO, calculate the worst group loss
            if self.GDRO:
                worst_group_loss_t, loss_per_group_t = calc_worst_group_loss(self.model, self.loss, self.X_val, self.y_val, self.g_val)
            else:
                  # calculate the loss at the current weights, using the validation data and the validation weights
                loss_t = calc_loss_for_model(self.model, self.loss, self.X_val, self.y_val, self.g_val, weights_obj = self.weights_obj_val, type_pred='probabilities')

             # save the loss at t
            if save_trajectory:
                if self.GDRO:
                    loss_at_t_traj[t-1] = worst_group_loss_t
                else:
                    loss_at_t_traj[t-1] = loss_t

            # if GDRO, this is done based on the worst group loss
            if self.GDRO:
                if worst_group_loss_t < best_worst_group_loss:
                    best_worst_group_loss = worst_group_loss_t
                    patience_count = patience
                    best_p = p_t.copy()
                else:
                    patience_count -= 1
            
            # if not GDRO, this is done based on the overall loss
            else:
                # check if the loss is less than the best loss minus eps
                if loss_t < (best_loss - eps):
                    best_loss = loss_t
                    patience_count = patience
                    best_p = p_t.copy()
                else:
                    patience_count -= 1
            
            # check if the patience count is 0
            if patience_count == 0:
                stop_GD = True
            

            # if GDRO, change the weights based on the loss
            if self.GDRO:

                # update the weights
                q_t =  update_DRO_weights(q_t, loss_per_group_t, eta_q, p_min=self.p_min, p_max=1.0)

                # set the weights for the validation set
                self.weights_obj_val.reset_weights(p_w=q_t)


            # calculate the grad
            grad = self.weight_grad_via_ift(self.model, p_t, self.X_train, self.y_train, self.g_train, self.X_val, self.y_val, self.g_val, self.weights_obj_val, eps=1e-6,   subsample_weights=self.subsample_weights)

            # provide information about the process
            if verbose:
                if self.GDRO:
                    loss_format = worst_group_loss_t
                else:
                    loss_format = loss_t
                
                # format the p_t, and the gradients to print
                p_t_format = {g: round(p_t[g], 4) for g in p_t}
                grad_format = {g: round(grad[g], 4) for g in grad}
                print('At step {}, the loss is {:.4f}, we have {} patience left, and the probabilities are {}, which sum to {:.4f} with gradients {}.'.format(t, loss_format, patience_count, p_t_format,  sum(p_t.values()), grad_format))

                if self.GDRO:
                    q_t_format = {g: round(q_t[g], 4) for g in q_t}
                    loss_per_group_format = {g: round(loss_per_group_t[g], 4) for g in loss_per_group_t}
                    print('The GDRO probabilities are updated to {}, based on this loss per group: {}'.format(q_t_format, loss_per_group_format))

            # make a copy of the weights
            p_t_plus_1 = p_t.copy()

            # determine the learning rate at time t
            if lr_schedule == 'constant':
                lr_t = lr
            elif lr_schedule == 'exponential':
                lr_t = lr * np.exp(-decay*t)
            elif lr_schedule == 'linear':
                lr_t = lr * decay
            else:
                Exception('The learning rate schedule is not recognized')
            

            # calculate the updates
            updates = dict.fromkeys(self.groups, 0)

            # update the weights per group
            for g in self.groups:
                # get the grad
                update =  (grad[g])
                
                # if locked in, do not update
                if g == lock_in_p_g and lock_in_p_g is not None:
                    continue
                
                # check if momentum is not None
                if momentum is not None:
                    update = (1-momentum)*update + (momentum * prev_update[g])
                    
                    # save the update
                    prev_update[g] = update
              
                # add to dict of updates via the learning rate
                updates[g] = -lr_t*update

            # if stable, then deduct the max update
            if stable_exp:
                max_update = max([updates[g] for g in self.groups])
                updates = {g: updates[g] - max_update for g in self.groups}
            
            
            # update the p
            for g in self.groups:
                p_t_plus_1[g] = (p_t_plus_1[g] * np.exp(updates[g])).item()


            # round the p_at_t to the specified number of decimal places
            p_t= round_p_dict(p_t_plus_1, self.weight_rounding)

            # clip the p_at_t
            p_t =  clip_p_dict_per_group(p_t, p_min=self.p_min, p_max=1.0)

            # if normalize, clip again
            if not self.subsample_weights:
                p_t = normalize_p_dict(p_t)

            # after the p_at_t is determined, update the model
            time_before = time.time()
            self.model.reset_weights(p_w=p_t)
            self.model.fit(self.X_train, self.y_train, self.g_train)
            if verbose:
                print('The model is updated in {} seconds'.format(time.time()-time_before))

            # save the trajectory if needed
            if save_trajectory:
                p_at_t_traj[t] = np.array(list(p_t.values()))
            t += 1
        
        # return the best p
        if self.GDRO:
            print('Returning the p={}, for which loss is {}'.format(best_p, best_worst_group_loss))
        else:
            print('Returning the p={}, for which loss is {}'.format(best_p, best_loss))

        # return the weight
        if save_trajectory:
            return best_p, p_at_t_traj[:t-1], loss_at_t_traj[:t-1]
        else:
            return best_p

                



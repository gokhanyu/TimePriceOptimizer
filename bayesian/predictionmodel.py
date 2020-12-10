# Authors: R. Chandra, T. Lovelock

# MCMC Random Walk for Feedforward Neural Network for One-Step-Ahead Chaotic Time Series Prediction

# --------------------
# Libaries
# --------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import math
import seaborn as sb
from statsmodels.graphics.tsaplots import plot_acf

# --------------------
# Neural Network Setup
# --------------------
class Network:
    def __init__(self, Topo, Train, Test, learn_rate):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        self.lrate = learn_rate

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = (np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])) # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = (np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])) # bias second layer

        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        return sqerror

    def ForwardPass(self, X):

        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1) # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = np.copy(z2) #Linear output for second layer.

    def BackwardPass(self, Input, desired):

        #Out Layer - linear activation function used
        out_delta = (desired - self.out)

        #Hidden Layer - sigmoid activation function used
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        #Update Gradients
        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

    def decode(self, w):
        #Convert weights from an array to matrix form
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]
        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))
        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = (w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]][None, :])
        self.B2 = (w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size
                                                               + self.Top[1] + self.Top[2]][None, :])

    def encode(self):
        #Convert weights to an ordered array
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        B1 = self.B1.ravel()
        B2 = self.B2.ravel()
        w = np.concatenate([w1, w2, B1, B2])
        return w

    def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for i in range(0, depth):
            for i in range(0, size):
                pat = i
                Input = data[pat, 0:self.Top[0]][None, :]
                Desired = data[pat, self.Top[0]:][None, :]
                self.ForwardPass(Input)
                self.BackwardPass(Input, Desired)

        w_updated = self.encode()

        return w_updated

    def evaluate_proposal(self, data, w):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        fx = np.zeros((size, self.Top[2]))

        for pat in range(0, size):
            Input[:] = data[pat, 0:self.Top[0]]
            self.ForwardPass(Input)
            fx[pat] = self.out

        return fx

    def squared_error(self, prediction, actual):
        #Used for SGD model
        return np.sum(np.square(prediction - actual)) / prediction.shape[0]  # to cater more in one output/class

    def test_sgd_model(self, data, tolerance):

        num_instances = data.shape[0]

        class_perf = 0
        sum_sqer = 0
        for s in range(0, num_instances):
            input_instance = self.TrainData[s, 0:self.Top[0]]
            actual = self.TrainData[s, self.Top[0]:]
            self.ForwardPass(input_instance)
            sum_sqer += self.squared_error(self.out, actual)
            index = np.argmax(prediction)

        rmse = np.sqrt(sum_sqer / num_instances)
        print(rmse, rmse)

        return rmse

    def SGD(self, max_epoch):
        epoch = 0
        shuffle = True

        while epoch < max_epoch:
            sum_sqer = 0
            for s in range(0, self.TrainData.shape[0]):
                if shuffle == True:
                    i = random.randint(0, self.TrainData.shape[0] - 1)
                input_instance = self.TrainData[i, 0:self.Top[0]][None, :]
                actual = self.TrainData[i, self.Top[0]:][None, :]
                self.ForwardPass(input_instance)
                sum_sqer += self.squared_error(self.out, actual)
                self.BackwardPass(input_instance, actual)  # major difference when compared to GD
            epoch = epoch + 1

        rmse_train = self.test_sgd_model(self.TrainData, 0.3)
        rmse_test = self.test_sgd_model(self.TestData, 0.3)

        return rmse_train, rmse_test

# --------------------
# MCMC Traing Model Class
# --------------------
class MCMC:
    def __init__(self,  use_langevin_gradients , l_prob,  learn_rate,  samples, traindata, testdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        self.use_langevin_gradients = use_langevin_gradients
        self.l_prob = l_prob  # likelihood prob
        self.learn_rate = learn_rate

        #Store Bulk Data Package After Run
        self.data_package = ""
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2)).mean(axis=0)

    def likelihood_func(self, neuralnet, data, w, tausq):
        y = data[:, self.topology[0]:]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        loss = np.sum(-0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq)
        return [loss, fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def update_weights(self, weights, loops, step_w, accept_log):
    #This module samples weights based on a step size.
    #Sampling was made independent for each of the layers increase different sampling methods are used in the future.

        #Proposal weights
        w_proposal = np.copy(weights)

        #Layer sizes and weights
        w_layer1size = self.topology[0] * self.topology[1]
        w_layer2size = self.topology[1] * self.topology[2]
        w_layer1 = w_proposal[0:w_layer1size]
        w_layer2 = w_proposal[w_layer1size:w_layer1size + w_layer2size]
        B1 = w_proposal[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.topology[1]]
        B2 = w_proposal[w_layer1size + w_layer2size + self.topology[1]:w_layer1size + w_layer2size
                                                                       + self.topology[1] + self.topology[2]]
        #Adjsuted step size to target 50% acceptence ratio.
        if (loops>100) and (loops%25 ==0):

            #This formula will adjusted the step size by a factor of [0.5, 2].
            accept_ratio_temp = sum(accept_log) / len(accept_log)
            step_w = max(step_w * 0.5* np.exp(2 * accept_ratio_temp), self.org_w_limt * 0.1)

        # Layer 1 Weights
        w_layer1 = np.random.normal(w_layer1, step_w, w_layer1size)
        W1 = np.reshape(w_layer1, (self.topology[0], self.topology[1]))

        # Bias 1 Weights
        B1 = np.random.normal(B1, step_w, B1.shape[0])

        # Layer 2 Weights
        w_layer2 = np.random.normal(w_layer2, step_w, w_layer2size)
        W2 = np.reshape(w_layer2, (self.topology[1], self.topology[2]))

        # Bias 2 Weights
        B2 = np.random.normal(B2, step_w, B2.shape[0])

        #Encode Weights
        W1 = W1.ravel()
        W2 = W2.ravel()
        w_proposal = np.concatenate([W1, W2, B1, B2])

        return w_proposal, step_w

    def sampler(self, w_limit, tau_limit):

        self.sgd_depth = 1
        self.org_w_limt = w_limit

        # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        #Store Data
        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)
        Likelihood = pd.DataFrame(columns=["Iteration", "Value"])

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]:]
        y_train = self.traindata[:, netw[0]:]

        #num of all weights and bias
        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]

        #posterior of all weights and bias over all samples
        pos_w = np.ones((samples, w_size))
        pos_tau = np.ones((samples, 1))

        # fx of train & test data over all samples
        fxtrain_samples = np.ones((samples, trainsize, netw[2]))
        fxtest_samples = np.ones((samples, testsize, netw[2]))
        rmse_train = np.zeros((samples, netw[2]))
        rmse_test = np.zeros((samples, netw[2]))

        '###########'

        step_w = w_limit # defines how much variation you need in changes to w
        step_eta = tau_limit  # exp 1
        w = np.random.randn(w_size) * 0.5 #Draw initial weights

        # --------------------- Declare FNN and initialize
        neuralnet = Network(self.topology, self.traindata, self.testdata, self.learn_rate)
        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        sigma_diagmat = np.zeros((w_size, w_size))  # for Equation 9 in Ref [Chandra_ICONIP2017]
        np.fill_diagonal(sigma_diagmat, step_w)
        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients
        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

        print(likelihood, ' Initial likelihood')

        naccept = []
        step_w_log = []

        langevin_count = 0

        #Sampling procedure
        for i in range(samples - 1):

            lx = np.random.uniform(0, 1, 1)
            if (self.use_langevin_gradients is True) and (lx < self.l_prob):
                w_gd = neuralnet.langevin_gradient(self.traindata, w.copy(), self.sgd_depth)  # Eq 8
                w_proposal = np.random.normal(w_gd, step_w, w_size)  # Eq 7
                w_prop_gd = neuralnet.langevin_gradient(self.traindata, w_proposal.copy(), self.sgd_depth)
                wc_delta = (w - w_prop_gd)
                wp_delta = (w_proposal - w_gd)

                sigma_sq = step_w

                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq

                diff_prop = first - second
                langevin_count = langevin_count + 1

            else:
                w_proposal, step_w = self.update_weights(w, i, step_w, naccept[-25:])
                diff_prop = 0

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w_proposal,
                                                                            tau_pro)

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood

            try:
                mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood + diff_prop))

            except OverflowError as e:
                mh_prob = 1

            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                naccept.append(1)
                step_w_log.append(step_w)
                likelihood = likelihood_proposal
                prior_likelihood = prior_prop
                w = w_proposal
                eta = eta_pro

                #Store Likelihood results
                Likelihood = Likelihood.append({'Iteration':i, 'Value': likelihood}, ignore_index=True)

                #Print limited output
                if i % 50 == 0:
                    print(i, likelihood, rmsetrain, rmsetest, '- accepted')

                #Record Data
                pos_w[i + 1,] = w_proposal
                pos_tau[i + 1,] = tau_pro
                fxtrain_samples[i + 1,] = pred_train
                fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest

            else:
                naccept.append(0)
                step_w_log.append(step_w)
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]

        print(sum(naccept), ' num accepted')
        print(sum(naccept) / (samples * 1.0) * 100, '% was accepted')
        accept_ratio = sum(naccept) / (samples * 1.0) * 100

        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test,
                Likelihood, naccept, step_w_log)


    def mcmc_manager(self, file, w_limit, tau_limit, sample_cols):
    #Controls all aspects of the sampling process and saves the results to the class.

        # --------------------
        # MCMC Sampling
        # --------------------
        timer = time.time()
        [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, Likelihood,
         accept_ratio, step_log] = self.sampler(w_limit, tau_limit)
        timetotal = (timer - time.time()) / 60

        # --------------------
        # Process Results
        # --------------------
        if file != "Predict":

            # --------------------
            # Remove Burnin Data
            # --------------------
            burnin = 0.25 * self.samples  # use post burn in samples
            pos_w = pos_w[int(burnin):, ]
            pos_tau = pos_tau[int(burnin):, ]

            # --------------------
            # Prepare Data for Analysis
            # --------------------
            rmse_tr = np.mean(rmse_train[int(burnin):])
            rmsetr_std = np.std(rmse_train[int(burnin):])
            rmse_tes = np.mean(rmse_test[int(burnin):])
            rmsetest_std = np.std(rmse_test[int(burnin):])
            acceptance_ratio = sum(accept_ratio)/len(accept_ratio)

            # Save Data for Covergence Analysis
            data = np.concatenate((pos_w, pos_tau), axis=1)[:, sample_cols]
            outres = open(file + '.txt', 'w')
            np.savetxt(outres, data, fmt='%1.5f');
            outres.close()

            # Save Simulation Results
            outres_db = open('result.txt', "a+")
            np.savetxt(outres_db, (self.use_langevin_gradients, self.learn_rate, rmse_tr, rmsetr_std, rmse_tes,
                                   rmsetest_std, acceptance_ratio, timetotal), fmt='%1.5f')
            outres_db.close()

        else:
            # Return raw data, burin period will be removed latter
            self.data_package = {'pos_w': pos_w, 'pos_tau': pos_tau, 'rmse_tr': rmse_train, 'rmse_test': rmse_test,
                                 'Likelihood': Likelihood, 'accept_ratio': accept_ratio, 'step_w_log': step_log,
                                 'timetotal': timetotal}

# --------------------
# Support Functions
# --------------------
def gelman_rubin(data):
    """
    Apply Gelman-Rubin convergence diagnostic to a bunch of chains.
    :param data: np.array of shape (Nchains, Nsamples, Npars)
    """
    Nchains, Nsamples, Npars = data.shape
    B_on_n = data.mean(axis=1).var(axis=0)  # variance of in-chain means
    W = data.var(axis=1).mean(axis=0)  # mean of in-chain variances

    # simple version, as in Obsidian
    sig2 = (Nsamples / (Nsamples - 1)) * W + B_on_n
    Vhat = sig2 + B_on_n / Nchains
    Rhat = Vhat / W

    # advanced version that accounts for ndof
    m, n = np.float(Nchains), np.float(Nsamples)
    si2 = data.var(axis=1)
    xi_bar = data.mean(axis=1)
    xi2_bar = data.mean(axis=1) ** 2
    var_si2 = data.var(axis=1).var(axis=0)
    allmean = data.mean(axis=1).mean(axis=0)
    cov_term1 = np.array([np.cov(si2[:, i], xi2_bar[:, i])[0, 1]
                          for i in range(Npars)])
    cov_term2 = np.array([-2 * allmean[i] * (np.cov(si2[:, i], xi_bar[:, i])[0, 1])
                          for i in range(Npars)])
    var_Vhat = (((n - 1) / n) ** 2 * 1.0 / m * var_si2
                + ((m + 1) / m) ** 2 * 2.0 / (m - 1) * B_on_n ** 2
                + 2.0 * (m + 1) * (n - 1) / (m * n ** 2)
                * n / m * (cov_term1 + cov_term2))
    df = 2 * Vhat ** 2 / var_Vhat

    Rhat *= df / (df - 2)

    if max(Rhat) < 1.1:
        print('++++++++++++++++++++++++++++++++++++++++++')
        print('Passed Gelman-Rubin Convergence Diagnostic')
    else:
        print('++++++++++++++++++++++++++++++++++++++++++')
        print('Failed Gelman-Rubin Convergence Diagnostic')

    print(Rhat, ' Rhat')
    print('++++++++++++++++++++++++++++++++++++++++++')

    return Rhat

def prediction(data, w_org, topology, scaler, org_data, type):

    # -----------------------
    #This function accepts:
    # data - New prediction data as log change
    # w-org - the sampled posterior weights from the neural network
    # topology - the specification of the neural network
    # scaler - details of the values used to scale the original log change changed to [-1,1]
    # original stock price data
    # If the prediction is for a bayes posterior or SGD

    #The function returns the 5 day prediction.
    # -----------------------

    #Rows in weight matrix
    rows_w = w_org.shape[0]

    #Bayes prediction
    if type == "Bayes":

        #Matrix to store predicted values
        predictions = pd.DataFrame(np.ones((rows_w, topology[2]+1)), columns = np.linspace(0,topology[2],topology[2]+1))

        #For each of the drawn posterior values
        for i in range(0, rows_w):

            #Choose a random row
            w = w_org[np.random.randint(low=0, high=rows_w-1),:]

            #Convert weights to matrix representation
            w_layer1size = topology[0] * topology[1]
            w_layer2size = topology[1] * topology[2]
            w_layer1 = w[0:w_layer1size]
            W1 = np.reshape(w_layer1, (topology[0], topology[1]))
            w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
            W2 = np.reshape(w_layer2, (topology[1], topology[2]))
            B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + topology[1]]
            B2 = w[w_layer1size + w_layer2size + topology[1]:w_layer1size + w_layer2size + topology[1] + topology[2]]

            #Forward Pass
            z1 = data.dot(W1) - B1
            hidout = 1 / (1 + np.exp(-z1))

            # Output of second hidden layer with linear activation function
            z2 = hidout.dot(W2) - B2
            predictions.iloc[i,1:topology[2]+1] = z2

    else:

        #SGD Prediction
        predictions = pd.DataFrame(np.ones((1, topology[2]+1)), columns = np.linspace(0,topology[2],topology[2]+1))

        # Choose a random row
        w = w_org
        w_layer1size = topology[0] * topology[1]
        w_layer2size = topology[1] * topology[2]
        w_layer1 = w[0:w_layer1size]
        W1 = np.reshape(w_layer1, (topology[0], topology[1]))
        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        W2 = np.reshape(w_layer2, (topology[1], topology[2]))
        B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + topology[1]]
        B2 = w[w_layer1size + w_layer2size + topology[1]:w_layer1size + w_layer2size + topology[1] + topology[2]]

        # Forward Pass
        z1 = data.dot(W1) - B1
        hidout = 1 / (1 + np.exp(-z1))

        # Output of second hidden layer with linear activation function
        z2 = hidout.dot(W2) - B2
        predictions.iloc[0, 1:topology[2] + 1] = z2

    # Transform predictions back to log-scale
    min = float(scaler[0]);
    max = float(scaler[1])

    #Convert to compounding returns
    for c in range(1, topology[2] + 1):
        predictions.iloc[:, c] = np.exp(((predictions.iloc[:, c] + 1) * (max - min)) / 2 + min)
        predictions.iloc[:, c] = predictions.iloc[:, c] * predictions.iloc[:, c - 1]

    # Covert back to share price and melt dataframe for graph
    predictions = predictions * org_data.iloc[-1]

    return predictions

def transform_data(data, lag_amount, forcast, scale):
#This model transform stock price data to log returns and scales the data between [-1,1]. The scaled data is put into
#an appropriate matrix for #neural network timeseries analysis.

    # Transform Data
    log_data = (np.log(data) - np.log(data).shift(1)).iloc[1:].to_frame()

    #If new model, output the scale
    if scale == 'None':
        min = log_data.min(); max = log_data.max();
        log_data = -1 + ((log_data - min)*(2))/(max-min);
        scaler = [min, max]

    #If for an existing model, scale new data
    else:
        log_data = -1 + ((log_data - scale[0])*(2))/(scale[1]-scale[0]);
        scaler = []

    #Lag Data
    for lag in range(1, lag_amount + forcast):
        log_data[str(lag + 1)] = log_data.iloc[:,0].shift(-lag)

    #Obtain last complete line for forecasting
    x_fcst_data = log_data.iloc[-(lag-forcast+1),:][0:(lag-forcast+1)].to_numpy()

    #Return transformed x_data
    temp = log_data.index
    log_data = log_data.iloc[:-(lag_amount + forcast-1), ]
    log_data = log_data.set_index(temp[(lag_amount + forcast-1):]) ##Check the accuracy of this

    return log_data, x_fcst_data, scaler

# --------------------
# Main Module
# --------------------
def main():

    # --------------------
    # Import Raw Data
    # --------------------
    pre_covid = pd.DataFrame(np.reshape(np.loadtxt('data1.csv'),(-1,1)),columns=["Adj Close"])
    post_covid = pd.DataFrame(np.reshape(np.loadtxt('data2.csv'),(-1,1)),columns=["Adj Close"])
    predict_set = pd.DataFrame(np.reshape(np.loadtxt('predict.csv'), (-1, 1)), columns=["Adj Close"])

    #--------------------
    #Neural Network Setup
    #--------------------
    #Topology
    hidden = 50
    input = 10
    output = 5
    topology = [input, hidden, output]

    #Step Sizes
    w_limit = 0.01 # step size for w
    tau_limit = 0.1 # step size for eta

    #Langevin Gradients
    use_langevin_gradients = False
    l_prob = 0.5
    learn_rate = 0.01

    # --------------------
    # Combine Data as per project requirements
    # --------------------

    # Part 1 - Train model using pre-covid and test post Covid
    p1_train, _, p1_scaler = transform_data(pre_covid['Adj Close'], input, output, 'None')
    p1_test, p1_fcst, _ = transform_data(post_covid['Adj Close'], input, output, p1_scaler)
    p1_train = p1_train.to_numpy(); p1_test = p1_test.to_numpy();

    # Part 2 - Train model using pre-covid and first 50% of post covid, train using the remainder of post covid
    p2_raw_train = pd.concat([pre_covid,pd.DataFrame(post_covid.iloc[0:round(post_covid.shape[0]/2)-1,0])])
    p2_raw_test = pd.DataFrame(post_covid.iloc[-round(post_covid.shape[0] / 2):, 0])
    p2_train, _, p2_scaler = transform_data(p2_raw_train['Adj Close'], input, output, 'None')
    p2_test, p2_fcst, _ = transform_data(p2_raw_test['Adj Close'], input, output, p2_scaler)
    p2_train = p2_train.to_numpy(); p2_test = p2_test.to_numpy();

    #--------------------
    #Diagnostic Test
    #--------------------
    print('*************************************')
    print('Starting Gelman Rubin Diagnostic Test')
    print('*************************************')

    #randomly select 5 posterior variables to assess
    data_size = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2] + 1
    sample_cols = random.sample(range(data_size), 5)

    test = 1
    for i in ['Test_1', 'Test_2', 'Test_3', 'Test_4', 'Test_5']:
        #************
        d_test = MCMC(use_langevin_gradients, l_prob, learn_rate, samples=15000, traindata=p2_train, testdata=p2_test, topology=topology)
        d_test.mcmc_manager(i,w_limit,tau_limit,sample_cols)

        print('*************************************')
        print('Test ' + str(test) + ' of 5 completed')
        print('*************************************')
        test += 1

    #Load Results
    file_list = ['Test_1.txt', 'Test_2.txt', 'Test_3.txt', 'Test_4.txt', 'Test_5.txt']
    pos_run1 = np.loadtxt(file_list[0])
    pos_run2 = np.loadtxt(file_list[1])
    pos_run3 = np.loadtxt(file_list[2])
    pos_run4 = np.loadtxt(file_list[3])
    pos_run5 = np.loadtxt(file_list[4])
    data = np.array([pos_run1, pos_run2, pos_run3, pos_run4, pos_run5])

    #Plot ACF of some results
    fig, axes = plt.subplots(1, 1)
    fig = plot_acf(pos_run3[:, 4], lags=500)
    plt.xlabel("Lag")
    plt.savefig('ACF.png')
    plt.close()

    #Generate Gelman_Rubin Score
    gelman_rubin(data)

    #--------------------
    #Part 1 - Experiment
    #--------------------

    #Bayesian

    print('*************************************')
    print('Staring Bayesian Experiment Part 1')
    print('*************************************')

    #Create & Run Sampler
    bp1 = MCMC(use_langevin_gradients, l_prob, learn_rate, samples = 100000, traindata=p1_train, testdata=p1_test, topology=topology)
    bp1.mcmc_manager('Predict',w_limit,tau_limit,sample_cols=0)

    #Plot RMSE for report
    for i in range(0, bp1.data_package.get('rmse_tr').shape[0]):
        if bp1.data_package.get('rmse_tr')[i, 0] != 0:
            plt.plot(bp1.data_package.get('rmse_test')[i:(i + 1500), 0])
            plt.title("Test RMSE", fontsize=12)
            plt.xlabel("First 1500 Samples", fontsize=12)
            plt.savefig('BP1_TST_RMSE Plot.png')
            plt.close()

            plt.plot(bp1.data_package.get('rmse_tr')[i:(i + 1500), 0])
            plt.title("Train RMSE", fontsize=12)
            plt.xlabel("First 1500 Samples", fontsize=12)
            plt.savefig('BP1_TR_RMSE Plot.png')
            plt.close()
            break

    print('*************************************')
    print('Staring Stochastic Gradient Experiment Part 1')
    print('*************************************')

    #SGD
    sgd_p1 = Network(topology, p1_train, p1_test, 0.025)
    sgd_p1.SGD(250)

    #--------------------
    #Part 2 - Experiment
    #--------------------

    #Bayesian

    print('*************************************')
    print('Staring Bayesian Experiment Part 2')
    print('*************************************')

    #Create & Run Sampler
    bp2 = MCMC(use_langevin_gradients, l_prob, learn_rate, samples = 100000, traindata=p2_train, testdata=p2_test, topology=topology)
    bp2.mcmc_manager('Predict',w_limit,tau_limit,sample_cols=0)

    #Plot RMSE for report
    for i in range(0, bp2.data_package.get('rmse_tr').shape[0]):
        if bp2.data_package.get('rmse_tr')[i, 0] != 0:
            plt.plot(bp2.data_package.get('rmse_test')[i:(i + 1500), 0])
            plt.title("Test RMSE", fontsize=12)
            plt.xlabel("First 1500 Samples", fontsize=12)
            plt.savefig('BP2_TST_RMSE Plot.png')
            plt.close()

            plt.plot(bp2.data_package.get('rmse_tr')[i:(i + 1500), 0])
            plt.title("Train RMSE", fontsize=12)
            plt.xlabel("First 1500 Samples", fontsize=12)
            plt.savefig('BP2_TR_RMSE Plot.png')
            plt.close()
            break

    #SGD
    print('*************************************')
    print('Staring Stochastic Gradient Experiment Part 2')
    print('*************************************')
    sgd_p2 = Network(topology, p2_train, p2_test, 0.025)
    sgd_p2.SGD(250)

    #--------------------
    #Prediction - Period 1
    #--------------------

    #Using Period 1 Data
    _, pred_fcst, _ = transform_data(post_covid['Adj Close'], input, output, p1_scaler) #Scale
    bp1_p_s = int(bp1.data_package.get('pos_w').shape[0]*0.25) #Need to remove burin in data
    bp1_p = prediction(pred_fcst, (bp1.data_package.get('pos_w'))[bp1_p_s:,:], topology, p1_scaler,
                       predict_set['Adj Close'], "Bayes")
    sg1_p = prediction(pred_fcst, sgd_p1.encode(), topology, p1_scaler, predict_set['Adj Close'], "SGD")

    #Plot Mean Predictions
    plt.close()
    sb.lineplot(x=np.linspace(-45, 0, 46), y=predict_set[-46:].values.flatten(), legend='brief', label="AMZN")
    fcst = sg1_p.mean(axis=0)
    sb.lineplot(x=fcst.index, y=fcst, markers=True, legend='brief', label="SGD")
    fcst = bp1_p.mean(axis=0)
    sb.lineplot(x=fcst.index, y=fcst, markers=True, legend='brief', label="Bayes")
    plt.legend(loc="lower right")
    plt.fill_between(fcst.index, np.percentile(bp1_p, 1, axis=0), np.percentile(bp1_p, 99, axis=0),alpha=0.25)
    plt.xlabel("Days From Prediction", fontsize=12)
    plt.ylabel("Share Price", fontsize=12)
    plt.title("5-Step Ahead Prediction", fontsize=12)
    plt.savefig('P1_5-Step Ahead Prediction')

    # Plot Distributions of 5 Day Prediction
    for i in range(1,int(max(fcst.index)+1)):
        plt.close()
        plt.hist(bp1_p.iloc[:, i], bins=50, label="Bayes")
        plt.axvline(x=sg1_p[i].values, color='r', linestyle='dashed', linewidth=2, label="SGD")
        plt.legend(prop={'size': 10})
        plt.title('Bayes ' + str(i) + '-Day Prediction Distribution')
        plt.savefig('P1_Dst ' + str(i) + ' day pred.png')

    #--------------------
    #Prediction - Period 2
    #--------------------

    #Using Period 2 Data
    _, pred_fcst, _ = transform_data(post_covid['Adj Close'], input, output, p2_scaler) #Scale
    bp2_p_s = int(bp1.data_package.get('pos_w').shape[0] * 0.25)  # Need to remove burin in data
    bp2_p = prediction(pred_fcst, (bp2.data_package.get('pos_w'))[bp2_p_s:,:], topology, p2_scaler,
                       predict_set['Adj Close'], "Bayes")
    sg2_p = prediction(pred_fcst, sgd_p2.encode(), topology, p2_scaler, predict_set['Adj Close'], "SGD")

    #Plot Mean Predictions
    plt.close()
    sb.lineplot(x=np.linspace(-45, 0, 46), y=predict_set[-46:].values.flatten(), legend='brief', label="AMZN")
    fcst = sg2_p.mean(axis=0)
    sb.lineplot(x=fcst.index, y=fcst, markers=True, legend='brief', label="SGD")
    fcst = bp2_p.mean(axis=0)
    sb.lineplot(x=fcst.index, y=fcst, markers=True, legend='brief', label="Bayes")
    plt.fill_between(fcst.index, np.percentile(bp2_p, 1, axis=0), np.percentile(bp2_p, 99, axis=0),alpha=0.25)
    plt.legend(loc="lower right")
    plt.xlabel("Days From Prediction", fontsize=12)
    plt.ylabel("Share Price", fontsize=12)
    plt.title("5-Step Ahead Prediction", fontsize=12)
    plt.savefig('P2_5-Step Ahead Prediction')

    # Plot Distributions of 5 Day Prediction
    for i in range(1,int(max(fcst.index)+1)):
        plt.close()
        plt.hist(bp2_p.iloc[:, i], bins=50, label="Bayes")
        plt.axvline(x=sg2_p[i].values, color='r', linestyle='dashed', linewidth=2, label="SGD")
        plt.legend(prop={'size': 10})
        plt.title('Bayes ' + str(i) + '-Day Prediction Distribution')
        plt.savefig('P2-Dst ' + str(i) + ' day pred.png')

if __name__ == "__main__": main()

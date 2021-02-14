
import seaborn
# import talos
import talos as ta
from talos.model.normalizers import lr_normalizer
import keras

from keras.activations import relu, elu, softmax, sigmoid, linear
from keras.losses import categorical_crossentropy, logcosh, binary_crossentropy, mean_absolute_error, mean_squared_error
from keras.optimizers import RMSprop, Adam, Nadam

from core.options import Options


talos_params = {
     'lr': (0.00050, 0.01, 1, 5),
     'first_neuron': [12],
     'hidden_layers': [1],
     'batch_size': [32, 64],
     'epochs': [100],
     'first_dropout': (0, 0.8, 4),
     'second_dropout': (0, 0.8, 4),
     'weight_regulizer': [None],
     'emb_output_dims': [None],
     'optimizer': [RMSprop],
     'losses': [mean_absolute_error, mean_squared_error],
     'activation': [sigmoid, relu, elu],
     'last_activation': [linear]
     }



def telos_baseline_model(x_train, y_train, x_val, y_val, params):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, GaussianNoise, BatchNormalization
    from keras.constraints import max_norm
    
    model = Sequential()
    model.add(Dropout(params['first_dropout'], input_shape=(Options.InputFeatureSize,)))
    model.add(Dense(params['first_neuron'], kernel_initializer='normal', activation=params['activation']))
    model.add(Dropout(params['second_dropout']))
    #kernel_constraint=max_norm(3)
    #model.add(BatchNormalization())
    #rms = RMSprop(lr = 0.00050)
    model.add(Dense(1, kernel_initializer='normal', activation=params['last_activation']))

    model.compile(loss=params['losses'],
                  # here we add a regulizer normalization function from Talos
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],params['optimizer'])),
                  metrics=['mae', 'mse'])

    callbacks = []

    if (Options.KerasEarlyStopping) : 
      early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=Options.KerasEarlyStoppingPatience, verbose=Options.KerasVerbose, mode='auto')
      callbacks.append(early_stop)

    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=Options.KerasVerbose,
                    callbacks = callbacks
                    #,validation_data=[x_val, y_val]
                    )

    return out, model




# then define your Keras model EXAMPLE
def example_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation=params['activation']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(y_train.shape[1], activation=params['last_activation']))

    model.compile(optimizer=params['optimizer'],
                  loss=params['losses'],
                  metrics=['acc'])

    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val])

    return out, model



def getData():

  X_train = Options.GlobalData[0]
  Y_train = Options.GlobalData[1]
  
  X_test = Options.GlobalData[2]
  Y_test = Options.GlobalData[3]

  return X_train, Y_train, X_test, Y_test



class NNTelosSearch():
  
  def minimize(self, x_train, y_train, x_val, y_val) :
    Options.GlobalData.append(x_train)
    Options.GlobalData.append(y_train)
    Options.GlobalData.append(x_val)
    Options.GlobalData.append(y_val)


    h = ta.Scan(x_train, y_train,
            params=talos_params,
            dataset_name=Options.RegressionAssetName,
            experiment_no='experiment1',
            model=telos_baseline_model,
            grid_downsample=0.5)
    #we're going to invoke the 'grid_downsample' parameter to 1/100 of the entire permutations. grid_downsample=0.01

    r = ta.Reporting(Options.RegressionAssetName + "_experiment1.csv")

    print(r.best_params('loss'))

    print("Evalutation of best performing model:")

    BITER
    #print(best_run)
    #print("Best model evaluation: ", best_model.evaluate(X_test, Y_test))
    
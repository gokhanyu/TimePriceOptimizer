
from keras.callbacks import Callback 

import signal

#https://github.com/asafh/keras/commit/02513af8c30e95d8dfd4c5d5ea6a65197d8fff1f
class SignalStopping(Callback):
    '''Stop training when an interrupt signal (or other) was received
    # Arguments
        sig: the signal to listen to. Defaults to signal.SIGINT.
        doubleSignalExits: Receiving the signal twice exits the python
            process instead of waiting for this epoch to finish.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
    '''
    def __init__(self, sig=signal.SIGINT, doubleSignalExits=False, verbose=0):

        super(SignalStopping, self).__init__()

        self.signal_received = False
        self.verbose = verbose
        self.doubleSignalExits = doubleSignalExits

        def signal_handler(sig, frame):
            if self.signal_received and self.doubleSignalExits:
                if self.verbose > 0:
                    print('') #new line to not print on current status bar. Better solution?
                    print('Received signal to stop ' + str(sig)+' twice. Exiting..')
                exit(sig)

            self.signal_received = True
            if self.verbose > 0:
                print('') #new line to not print on current status bar. Better solution?
                print('Received signal to stop: ' + str(sig))
        signal.signal(signal.SIGINT, signal_handler)
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        if self.signal_received:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: stopping due to signal' % (self.stopped_epoch))
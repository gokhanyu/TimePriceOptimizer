 
nnloss_epsilon = 1.0e-9


class nnloss(object):
  """Custom Loss functions"""

  @staticmethod
  def mean_absolute_percentage_error(y_true, y_pred): 
      y_true, y_pred = np.array(y_true), np.array(y_pred)
      return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


  #Good for stock market returns
  @staticmethod
  def stock_loss(y_true, y_pred):
      alpha = 100.
      loss = K.switch(K.less(y_true * y_pred, 0), \
          alpha*y_pred**2 - K.sign(y_true)*y_pred + K.abs(y_true), \
          K.abs(y_true - y_pred)
          )
      return K.mean(loss, axis=-1)


  #MSE on log values is better for volatility.
  @staticmethod
  def mse_log(y_true, y_pred):
      y_pred = K.clip(y_pred, nnloss_epsilon, 1.0 - nnloss_epsilon)
      loss = K.square(K.log(y_true) - K.log(y_pred))
      return K.mean(loss, axis=-1)

  @staticmethod
  def qlike_loss(y_true, y_pred):
      y_pred = K.clip(y_pred, nnloss_epsilon, 1.0 - nnloss_epsilon)
      loss = K.log(y_pred) + y_true / y_pred
      return K.mean(loss, axis=-1)


  @staticmethod
  def mse_sd(y_true, y_pred):
      y_pred = K.clip(y_pred, nnloss_epsilon, 1.0 - nnloss_epsilon)
      loss = K.square(y_true - K.sqrt(y_pred))
      return K.mean(loss, axis=-1)    

  @staticmethod
  def hmse(y_true, y_pred):
      y_pred = K.clip(y_pred, nnloss_epsilon, 1.0 - nnloss_epsilon)
      loss = K.square(y_true / y_pred - 1.)
      return K.mean(loss, axis=-1)    

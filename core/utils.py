import datetime as dt
import os, errno



class Timer():

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))

class Utils() :

  @staticmethod
  def ensure_directory(save_dir) :
    try:
      if not os.path.exists(save_dir) :
        os.makedirs(save_dir)
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise
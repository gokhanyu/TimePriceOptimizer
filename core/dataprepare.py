class DataPrepare :

  @staticmethod
  def run(out_futures_path, in_futures_path, train_start_date, future_start_date, future_end_date) :
    try:
      import sys
      import clr
      import os
      cwd = os.getcwd()

      newpath = os.path.join(cwd, "dll")
      sys.path.append(newpath)

      externals = clr.AddReference("externals")

      import externals as extern

      #test1 = extern.FastCsv.test1()

      output_done = extern.FastCsv.prepare_features(out_futures_path, in_futures_path, str(train_start_date), str(future_start_date), str(future_end_date))

      return True
    except:
      print("DataPrepare unexpected error during CLR .dll load:", sys.exc_info()[0])
      return False
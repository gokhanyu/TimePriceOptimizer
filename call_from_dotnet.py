import sys
import os


if __name__ == '__main__':
  cmd = os.getcwd() 
  new_dir = os.path.join(cmd, 'core')
  #sys.path.append(new_dir)
  print("Working directory: ", cmd)
  print("Added dir: ", new_dir)
  import core.run
  from core.options import Options #make sure options are in the sys path

  Options.PrepareFeatures         = False
  Options.PlottingEnabled         = True
  Options.TrainDataSize           = 1
  Options.TestDataSize            = 0
  Options.InputFeaturesPath       = ''
  Options.OutputFeaturesPath      = ''
  Options.PredictionSaveDirectory = ''

  Options.RegressionAssetName     = 'NZDUSD_H4'
  Options.RegressionMethodName    = 'HARM_TRIANGULAR'

  if (len(sys.argv) > 1):
    Options.InputFeaturesPath = sys.argv[1]

  if (len(sys.argv) > 2):
    Options.OutputFeaturesPath = sys.argv[2]
    print(sys.argv[1], sys.argv[2])
    
  if (len(sys.argv) > 3):
      Options.RegressionAssetName = sys.argv[3]

  if (len(sys.argv) > 4):
      Options.RegressionMethodName = sys.argv[4]

  Options.OutputValuesKey         = Options.RegressionAssetName + '_RPO'
  Options.OutputDateValuesKey     = Options.RegressionAssetName + '_RealClose_date'
  Options.RealClosePricesKey      = Options.RegressionAssetName + '_RealClose'

  Options.Init()

  core.run.main()
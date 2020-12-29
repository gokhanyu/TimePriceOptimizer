import time
import datetime
import os
from datetime import (timedelta)


class Options :

  from enum import Enum
  class DataWindowType(Enum):
    OneByOne = 1
    OneByOneTelosSearch = 2
    WindowBatch = 3

  FastTestRead = False #reads 100 rows from file if set to true

  AssetName = 'EURUSD 4H'
  AssetTimeFrameMinutes = 60*4 #4 hour
  AssetForecastMinutes = 60*24*30 #1 month

  PrepareFeatures = False #Cleans directory and prepares features with .NET dll call
  InputFeaturesPath = 'future\\' #read all input features from this folder
  OutputFeaturesPath = 'future\\output\\' #read all output features from this folder
  
  OutputReadyTrainAllPath = '' #init function assigns this value
  OutputReadyFuturesPath  = '' #init function assigns this value
  InputReadyTrainAllPath  = '' #init function assigns this value
  InputReadyFuturesPath   = '' #init function assigns this value

  #ColsToExclude = ['DateTime']
  OutputValuesKey = 'rpo'
  OutputDateValuesKey = 'rpo_date'
  RealClosePricesKey = 'real'

  TrainDataStartDate = datetime.datetime(1800, 1, 1)
  FuturePredictionStartDate = datetime.datetime(2019, 1, 1)
  FuturePredictionDayRange = 60

  
  TrainDataSize = 1
  TestDataSize = 0
  ValidationDataSize = (1 - TrainDataSize - TestDataSize)
  WindowSequenceLength = 56
  WindowShiftStep = 1
  InputFeatureSize = 2 #!important
  MultiplyDataByCustomFactor = 1 #multiplies y vector
  NormaliseData = False #Not implemented

  DataWindow = DataWindowType.OneByOne

  KerasOneByOneBatchSize = 5
  KerasOneByOneEpochs = 12

  FlattenWindowVector = False
  KerasWindowedBatchSize = 50
  KerasWindowedEpochs = 60
  
  KerasModelCheckpointSave = True
  KerasModelCheckpointSaveOnlyBestModel = True
  KerasEarlyStopping = True
  KerasEarlyStoppingPatience = 5
  KerasReduceLearningRateOnPlateu = False
  KerasSignalStopping = True
  KerasVerbose = 1
  KerasCheckPointForNEpoch = 6
  KerasNNSaveDirectory = "data\\saved_models"
  PredictionSaveDirectory = ""

  ReadNumberOfRowsFromFile = 100 if FastTestRead else None

  GlobalData = [] #Used in Telos Optimization


  @staticmethod
  def FuturePredictionEndDate() :
    value = Options.FuturePredictionStartDate + timedelta(days = Options.FuturePredictionDayRange)
    return value

  @staticmethod
  def PredictionSaveDirs() :
    arr = [Options.KerasNNSaveDirectory, os.path.join(Options.PredictionSaveDirectory, Options.AssetName)]
    return arr

  @staticmethod
  def Init() :
    Options.OutputReadyTrainAllPath = os.path.join(Options.OutputFeaturesPath, "ready_all_train")
    Options.OutputReadyFuturesPath  = os.path.join(Options.OutputFeaturesPath, "ready_future")
    Options.InputReadyTrainAllPath  = os.path.join(Options.InputFeaturesPath,  "ready_all_train")
    Options.InputReadyFuturesPath   = os.path.join(Options.InputFeaturesPath,  "ready_future")


   
  ################
  
  #DrawEpochLoss = False #draws a graph of loss and accuracy of each epoch
  #RunCrossValidation = False
  #HyperasSearch = False
  
  #FeatureFilter = []

  #TransformToEmbeddingMapping = True
  #FeaturePermutationImportancesEnabled = False
  #FeaturePermutationImportancesTopCount = 3000


  #CrossValidationFoldCount = 5
  #CrossValidationRandomState = 20180818
  #CrossValidationShowAllFoldGraphs = False
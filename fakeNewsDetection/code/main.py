import numpy as np

from bow_model import run_bow
from bow_model import run_save_data_pickle
from basic_lstm import run_lstm
from lstm_attention import run_lstm_attention
from util import Config

def bagOfWords(ite):
  print('Experiment no ::::: '+str(ite))
  if ite ==0:
    np.random.seed(1)
    config = Config()
    config.n_layers = 1
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Variable'
    config.b_max_len = 600
    config.n_epochs = 40
    # data needs to be saved in pickle form only once.
    # run_save_data_pickle()
    run_bow(config, final = True)
  """
  elif ite ==1:
    #np.random.seed(1)
    config = Config()
    config.n_layers = 2
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Variable'
    config.b_max_len = 75
    result = run_bow(config)
  elif ite ==2:
    # ## Experiment 
    #np.random.seed(1)
    config = Config()
    config.n_layers = 3
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Constant'
    config.b_max_len = 75
    result = run_bow(config)
  elif ite ==3:
    np.random.seed(1)
    config = Config()
    config.n_layers = 0
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Variable'
    config.b_max_len = 150
    result = run_bow(config)

  elif ite ==4:
    #np.random.seed(1)
    config = Config()
    config.n_layers = 1
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Variable'
    config.b_max_len = 150
    result = run_bow(config)

  elif ite ==5:
    #np.random.seed(1)
    config = Config()
    config.n_layers = 3
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Variable'
    config.b_max_len = 150
    result = run_bow(config)

  elif ite ==6:
    config = Config()
    config.n_layers = 0
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Variable'
    config.b_max_len = 300
    result = run_bow(config)

  elif ite ==7:
    #np.random.seed(1)
    config = Config()
    config.n_layers = 1
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Variable'
    config.b_max_len = 300
    result = run_bow(config)

  elif ite ==8:
    #np.random.seed(1)
    config = Config()
    config.n_layers = 3
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Constant'
    config.b_max_len = 300
    result = run_bow(config)
  
  elif ite ==9:
    np.random.seed(1)
    config = Config()
    config.n_layers = 0
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Variable'
    config.b_max_len = 600
    result = run_bow(config)

  elif ite ==10:
    # ## Experiment 
    np.random.seed(1)
    config = Config()
    config.n_layers = 1
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Variable'
    config.b_max_len = 600
    result = run_bow(config)

  elif ite ==11:
    np.random.seed(1)
    config = Config()
    config.n_layers = 3
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Constant'
    config.b_max_len = 600
    result = run_bow(config)

  elif ite ==12:
    ## Experiment 
    np.random.seed(1)
    config = Config()
    config.n_layers = 3
    config.xp = 'layers'
    config.model = 'bow'
    config.lr = 0.005
    config.trainable_embeddings = 'Constant'
    config.b_max_len = 150
    result = run_bow(config)"""
  
def run_lstm_with_parameters(iter):

  print('Experiment no ::::: '+str(iter))

  if iter == 0: 
    np.random.seed(1)
    config0 = Config()
    config0.max_length = 75
    config0.trainable_embeddings = 'Variable'
    config0.hidden_size = 100
    config0.n_epochs = 40
    config0.n_layers = 2
    config0.batch_size = 128
    config0.dropout = 0.8
    config0.lr = 0.001
    config0.model = 'lstm_basic'
    run_lstm(config0, final = True)
  """  
  elif iter==1:
    np.random.seed(1)
    config1 = Config()
    config1.max_length = 150
    config1.trainable_embeddings = 'Variable'
    config1.hidden_size = 100
    config1.batch_size = 128
    config1.dropout = 0.8
    config1.n_layers = 1
    config1.lr = 0.001
    #config1.attention_length = 15
    result = run_lstm(config1)
  elif iter==2:
    np.random.seed(1)
    config2 = Config()
    config2.max_length = 250
    config2.trainable_embeddings = 'Variable'
    config2.hidden_size = 100
    config2.batch_size = 128
    config2.dropout = 0.8
    config2.n_layers = 1
    config2.lr = 0.001
    #config2.attention_length = 15
    result = run_lstm(config2)
  elif iter==3:
    np.random.seed(1)
    config3 = Config()
    config3.max_length = 150
    config3.trainable_embeddings = 'Variable'
    config3.hidden_size = 100
    config3.batch_size = 128
    config3.dropout = 0.8
    config3.n_layers = 1
    config3.downsample = True
    result = run_lstm(config3)
  elif iter==4:
    np.random.seed(1)
    config4 = Config()
    config4.max_length = 150
    config4.trainable_embeddings = 'Variable'
    config4.hidden_size = 100
    config4.batch_size = 128
    config4.dropout = 0.8
    config4.n_layers = 2
    config4.downsample = True
    result = run_lstm(config4)
  elif iter==5:
    np.random.seed(1)
    config5 = Config()
    config5.max_length = 150
    config5.trainable_embeddings = 'Variable'
    config5.hidden_size = 100
    config5.batch_size = 128
    config5.dropout = 0.8
    config5.n_layers = 4
    config5.downsample = True
    result = run_lstm(config5)

  elif iter==6:
    np.random.seed(1)
    config = Config()
    config.max_length = 75
    config.trainable_embeddings = 'Variable'
    config.hidden_size = 100
    config.batch_size = 128
    config.dropout = 0.9
    config.n_layers = 2
    config.downsample = False
    config.lr = 0.005
    result = run_lstm(config)
  elif iter==7:
    np.random.seed(1)
    config = Config()
    config.max_length = 75
    config.trainable_embeddings = 'Variable'
    config.hidden_size = 100
    config.batch_size = 128
    config.dropout = 0.65
    config.n_layers = 2
    config.downsample = False
    config.lr = 0.005
    result = run_lstm(config)
  elif iter==8:
    np.random.seed(1)
    config = Config()
    config.max_length = 75
    config.trainable_embeddings = 'Variable'
    config.hidden_size = 100
    #config.n_epochs = 40
    config.batch_size = 128
    config.dropout = 0.5
    config.n_layers = 2
    config.downsample = False
    config.lr = 0.005
    result = run_lstm(config)

  elif iter==9:
    np.random.seed(1)
    config = Config()
    config.max_length = 50
    config.trainable_embeddings = 'Variable'
    config.hidden_size = 100
    config.batch_size = 128
    config.dropout = 0.8
    config.n_layers = 2
    config.downsample = False
    config.lr = 0.005
    result = run_lstm(config)  
  elif iter==10:
    np.random.seed(1)
    config = Config()
    config.max_length = 30
    config.trainable_embeddings = 'Variable'
    config.hidden_size = 100
    config.batch_size = 128
    config.dropout = 0.8
    config.n_layers = 2
    config.downsample = False
    config.lr = 0.005
    result = run_lstm(config)
  elif iter ==11:
    np.random.seed(1)
    config0 = Config()
    config0.n_layers = 0
    config0.max_length = 75
    config0.trainable_embeddings = 'Variable'
    config0.hidden_size = 100
    config0.n_layers = 1
    config0.batch_size = 128
    config0.dropout = 0.8
    config0.n_layers = 1
    config0.lr = 0.001
    config0.attention_length = 15
    result = run_lstm(config0)"""

def run_lstm_attention_with_parameters(ite):
  print('Experiment no ::::: '+str(ite))
  
  if ite==0:
    np.random.seed(1)
    config0 = Config()
    # print('Running run_lstm_with_parameters')
    config0.max_length = 75
    config0.trainable_embeddings = 'Variable'
    config0.hidden_size = 100
    config0.batch_size = 128
    config0.dropout = 0.8
    config0.n_layers = 2
    config0.lr = 0.001
    config0.xp = 'final_test'
    config0.model = 'lstm_attention'
    # config0.num_samples = 100
    config0.attention_length = 15
    result = run_lstm_attention(config0, final = True)
  """
  elif ite==1:
    np.random.seed(1)
    config0 = Config()
    # # print('Running run_lstm_with_parameters')
    config0.max_length = 150
    config0.trainable_embeddings = 'Variable'
    config0.hidden_size = 100
    config0.n_layers = 2
    config0.batch_size = 128
    config0.dropout = 0.8
    config0.n_layers = 4
    # # config0.downsample = False
    config0.lr = 0.001
    # # config0.num_samples = 
    config0.attention_length = 15
    result = run_lstm_attention(config0)

  elif ite==2:
    #### Testing attention_length # Experiment 1
    ## 1 layer, max_length = 150, attention_length = 10
    np.random.seed(1)
    config0 = Config()
    # # print('Running run_lstm_with_parameters')
    config0.max_length = 150
    config0.trainable_embeddings = 'Variable'
    config0.hidden_size = 100
    config0.n_layers = 1
    config0.batch_size = 128
    config0.dropout = 0.8
    config0.n_layers = 1
    # # config0.downsample = False
    config0.lr = 0.001
    config0.attention_length = 10
    result = run_lstm_attention(config0)
  elif ite==3:
    # #### Testing attention_length # Experiment 2
    # ## 1 layer, max_length = 150, attention_length = 20
    np.random.seed(1)
    config0 = Config()
    # # print('Running run_lstm_with_parameters')
    config0.max_length = 150
    config0.trainable_embeddings = 'Variable'
    config0.hidden_size = 100
    config0.n_layers = 1
    config0.batch_size = 128
    config0.dropout = 0.8
    config0.n_layers = 1
    # # config0.downsample = False
    config0.lr = 0.001
    config0.attention_length = 20
    result = run_lstm_attention(config0)
    
  elif ite==4:
    # #### Testing alearning rate # Experiment 1
    # ## 1 layer, max_length = 150, lr = 0.0005
    np.random.seed(1)
    config0 = Config()
    # # print('Running run_lstm_with_parameters')
    config0.max_length = 150
    config0.trainable_embeddings = 'Variable'
    config0.hidden_size = 100
    config0.n_layers = 1
    config0.batch_size = 128
    config0.dropout = 0.8
    config0.n_layers = 1
    # # config0.downsample = False
    config0.lr = 0.0005
    config0.attention_length = 15
    result = run_lstm_attention(config0)
  elif ite==5:
    # #### Testing alearning rate # Experiment 2
    # ## 1 layer, max_length = 150, lr = 0.0002
    np.random.seed(1)
    config0 = Config()
    # # print('Running run_lstm_with_parameters')
    config0.max_length = 150
    config0.trainable_embeddings = 'Variable'
    config0.hidden_size = 100
    config0.n_epochs = 40
    config0.n_layers = 4
    config0.batch_size = 128
    config0.dropout = 0.8
    config0.n_layers = 1
    # # config0.downsample = False
    config0.lr = 0.0002
    config0.attention_length = 15
    result = run_lstm_attention(config0)"""


if __name__ == "__main__":
  print("Bag of Words Model")
  for i in range(0,13):
    bagOfWords(i)

  print("LSTM Basic")
  for i in range(0,11):
    run_lstm_with_parameters(i)

  print("LSTM Attention")
  for i in range(0,7):
    run_lstm_attention_with_parameters(i)
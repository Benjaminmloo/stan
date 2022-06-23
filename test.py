from stannetflow import STANSynthesizer, STANCustomDataLoader, NetflowFormatTransformer
from stannetflow.artificial.datamaker import artificial_data_generator
from stannetflow.preprocess import user_analysis, user_selection, download_ugr16, prepare_standata
from stannetflow.evaluation.correlation import corr_plot, mse_temporal, mse_same_row
import pandas as pd
import matplotlib.pyplot as plt

def test_artificial():
  adg = artificial_data_generator(weight_list=[0.9, 0.9])
  df_naive = adg.sample(row_num=1000)
  X, y = adg.agg(agg=1)

  stan = STANSynthesizer(dim_in=2, dim_window=1)
  stan.fit(X, y)
  samples = stan.sample(50)
  samples.plot.scatter(x = 0, y = 1)
  #plt.show()
  #print(samples)

def test_ugr16(train_file, load_checkpoint=False):
  train_loader = STANCustomDataLoader(train_file, 6, 16).get_loader()
  ugr16_n_col, ugr16_n_agg, ugr16_arch_mode = 16, 5, 'B'
  # index of the columns that are discrete (in one-hot groups), categorical (number of types)
  # or any order if wanted
  ugr16_discrete_columns = [[11,12], [13, 14, 15]]
  ugr16_categorical_columns = {5:1670, 6:1670, 7:256, 8:256, 9:256, 10:256}
  ugr16_execute_order = [0,1,13,11,5,6,7,8,9,10,3,2,4]

  stan = STANSynthesizer(dim_in=ugr16_n_col, dim_window=ugr16_n_agg, 
          discrete_columns=ugr16_discrete_columns,
          categorical_columns=ugr16_categorical_columns,
          execute_order=ugr16_execute_order,
          arch_mode=ugr16_arch_mode
          )
  
  if load_checkpoint is False:
    stan.batch_fit(train_loader, epochs=10)
  else:
    stan.load_model('ep998') # checkpoint name
    # validation
    # stan.validate_loss(test_loader, loaded_ep='ep998')

  ntt = NetflowFormatTransformer()
  samples = stan.time_series_sample(8640)
  df_rev = ntt.rev_transfer(samples)
  #pd.set_option('max_columns', 20)
  print(df_rev)
  return df_rev

if __name__ == "__main__":
  """#generate artificial data
  print("start: test artificial")
  test_artificial()
  # load model and generate ugr16-format netflow data
  print("start: sample w/ checkpoint")
  test_ugr16('stan_data/ugr16_demo.csv', True)
  # train and generate ugr16-format netflow data
  print("start: sample w/o checkpoint")  
  #test_ugr16('example_data/data_ugr16/testing_ugr.csv')
  test_ugr16('stan_data/ugr16_demo.csv')"""

  print("start: ugr16")
  test_ugr16('/home/ubuntu/Documents/data/stan/preprocessed_data/april_week3_day1_PREPROCESSED.csv')

  # ugr16 netflow user-based analysis
  # user_analysis()
  # user_selection()
  # download_ugr16()

  # correlation plot and metric
  # corr_plot(plot=True, plot_axis='xx1')
  # corr_plot(plot=True, plot_axis='xy')
  # mse_temporal()
  # mse_same_row()

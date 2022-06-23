import configparser
import glob

import pandas as pd

from stannetflow.analyze_functions import analyze, extract, prepare_folders, recover_userlist_from_folder

from stannetflow.synthesizers.stan import NetflowFormatTransformer, STANTemporalTransformer


def user_analysis():
  analyze()

def user_selection():
  config = configparser.ConfigParser()
  config.read('../ugr16_config.ini')
  print({section: dict(config[section]) for section in config.sections()})
  user_list = config['DEFAULT']['userlist'].split(',')
  print('extracting:', user_list)
  prepare_folders()
  # recover_userlist_from_folder()
  extract(user_list)

def download_ugr16():
  print('Visit the following url to download april_week3.csv')  
  print('https://nesg.ugr.es/nesg-ugr16/april_week3.php')

def _prepare(data_path='', output_file='', agg=5):
  """

  Parameters
  ----------
  data_path - glob path with wild cards which will include all the csv files containing data pertinent data
  output_file - the file to save processed data
  agg - rows to aggregate together
  """
  if len(output_file) and len(data_path):
    count = 0
    ntt = NetflowFormatTransformer()
    tft = STANTemporalTransformer(output_file)
    for f in glob.glob(data_path):
      print('user:', f)
      this_ip = f.split("_")[-1][:-4]
      df = pd.read_csv(f)
      tft.push_back(df, agg=agg, transformer=ntt)
      count += 1
    print(count)

def prepare_standata(agg=5, train_folder='stan_data/ugr16/raw_data', train_output='to_train.csv'):
  """

  Parameters
  ----------
  agg - rows to aggregate together
  data_folder - folder containing unprocessed data for training the model, organised in seperate csv partition by local IP
  train_output - path to drop output csv of training dataa
  """
  if len(train_folder):
    print('making train for:')
    _prepare(data_path=train_folder+'**/*.csv', output_file=train_output, agg=agg)

if __name__ == "__main__":
  #download_ugr16()
  #user_analysis()
  #user_selection()
  prepare_standata()
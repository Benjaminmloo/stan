from os.path import exists
from stannetflow import STANSynthesizer, STANCustomDataLoader, NetflowFormatTransformer
from stannetflow.artificial.datamaker import artificial_data_generator
from stannetflow.preprocess import user_analysis, user_selection, download_ugr16, prepare_standata
from stannetflow.evaluation.correlation import corr_plot, mse_temporal, mse_same_row
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm


def test_artificial():
    adg = artificial_data_generator(weight_list=[0.9, 0.9])
    df_naive = adg.sample(row_num=1000)
    X, y = adg.agg(agg=1)

    stan = STANSynthesizer(dim_in=2, dim_window=1)
    stan.fit(X, y)
    samples = stan.sample(50)
    samples.plot.scatter(x=0, y=1)
    # plt.show()
    # print(samples)


def test_ugr16(train_file, checkpoint=0, epochs=2, sample_duration=86400, this_ip=None, model_path='./', save_path=None):
    """

    @param train_file: path to the file containing preprocessed widnowed data
    @param checkpoint: check point to load
    @param epochs: epoch number to stop training at, if epochs == 0 forgo training and just load the model and start generating data
    @param sample_duration: desired sample length of time in seconds
    @param this_ip: ip addressse(s) to create data for if this is a string it will only create data for that one asset, if it is a list of strings it will generate data for all specified assets
    @param model_path: Path to parent of the checkpoint folder where the model will load and save from
    @param save_path: Path to folder where log data for individual assets will be saved as a csv set to 'none' if you don't want to save data to file
    @return: A string indicating the result or a dataframe containing synthesized data
    """
    train_loader = STANCustomDataLoader(train_file, 6, 16).get_loader()
    ugr16_n_col, ugr16_n_agg, ugr16_arch_mode = 16, 5, 'B'
    # index of the columns that are discrete (in one-hot groups)
    # given as a list of columns that are a part of each one-hot group
    # 11, 12 are for in coming and outgoing
    # 13, 14, 15 are transfer protocol TCP, UDP, other respectively
    ugr16_discrete_columns = [[11, 12], [13, 14, 15]]
    #categorical here means whole numbers encoded up to the specified limit (column:max value)
    ugr16_categorical_columns = {5: 1670, 6: 1670, 7: 256, 8: 256, 9: 256, 10: 256}
    #order that the each attribute's model will be trained
    ugr16_execute_order = [0, 1, 13, 11, 5, 6, 7, 8, 9, 10, 3, 2, 4]

    stan = STANSynthesizer(dim_in=ugr16_n_col, dim_window=ugr16_n_agg,
                           discrete_columns=ugr16_discrete_columns,
                           categorical_columns=ugr16_categorical_columns,
                           execute_order=ugr16_execute_order,
                           arch_mode=ugr16_arch_mode,
                           data_path=model_path
                           )

    if checkpoint or epochs == 0:
        print('loading model at epoch: %d' % checkpoint)
        stan.load_model(checkpoint)  # checkpoint name

    if checkpoint < epochs:
        print('Starting training up to epoch: %d' % epochs)
        stan.batch_fit(train_loader, epochs=epochs)

    ntt = NetflowFormatTransformer()
    # validation
    # stan.validate_loss(test_loader, loaded_ep='ep998')
    print('synthesizing asset samples')
    if type(this_ip) is list:
        for ip in tqdm(this_ip):
            #skip any assets that have already been generated
            #enables resuming synthesis
            if not exists(save_path + '%s.csv' % ip):
                samples = stan.time_series_sample(sample_duration)
                df_rev = ntt.rev_transfer(samples, this_ip=ip)
                if save_path is not None:
                    df_rev.to_csv(save_path + '%s.csv' % ip)

        return 'complete'
    elif this_ip is None:
        return 'Model trained, no data generated'
    else:
        samples = stan.time_series_sample(sample_duration)
        df_rev = ntt.rev_transfer(samples, this_ip=this_ip)
        if save_path is not None:
            df_rev.to_csv(save_path + '%s.csv' % this_ip)
        return df_rev

    return 'ERROR'


if __name__ == "__main__":
    checkpoint = 0

    if len(sys.argv) >= 2:
        try:
            checkpoint = int(sys.argv[1])
        except TypeError:
            print("Invalid checkpoint number")

    # old main runs through simple test and data generation
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

    test_ugr16('/home/ubuntu/Documents/data/stan/preprocessed_data/april_week3_day1_PREPROCESSED.csv', \
               checkpoint=checkpoint, epochs=900, sample_duration=86400, \
               model_path='/home/ubuntu/Documents/data/stan/')

    # ugr16 netflow user-based analysis
    # user_analysis()
    # user_selection()
    # download_ugr16()

    # correlation plot and metric
    # corr_plot(plot=True, plot_axis='xx1')
    # corr_plot(plot=True, plot_axis='xy')
    # mse_temporal()
    # mse_same_row()

from depen import *
import datasets
import encoder

#it is just for checking, for training each individual model and loading parameter make a separate function
#this function can be moved to separate file if complicated
def train_encoder(hparams):
    print("training_encoder")
    #save also params etc. in separate folder or in drive


def load_encoder():
    print("load")


def main_check(hparams):
    load_encoder()
    data_module = datasets.MyDataModuleExample(hparams)
    data_module.prepare_data()
    data_module.setup()

    n = 2
    waveform, sample_rate, labels = data_module.yesno_data[n]

    print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))
    print(waveform.shape)
    print(len(labels))

    plt.figure()
    plt.plot(waveform.t().numpy())
    plt.show()

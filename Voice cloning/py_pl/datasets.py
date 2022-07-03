from depen import *


class MyDataModuleExample(pl.LightningDataModule):
  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams
    self.path = self.hparams.path_dataset_encoder

  def prepare_data(self):
    #here you download your dataset, from some site for example
    #or pytorch torchaudio etc.
    #or call torch.utils.data.Dataset type
    self.yesno_data = torchaudio.datasets.YESNO(self.path, download=True)
    self.dataset_size = len(self.yesno_data)

  def setup(self):
    # called on every GPU
    #We should make dataset which will be called depending on a subject
    #For example id 1 for training, id 2,3 for testing
    #or like this but how, I think example above better, so no need of function only torch.utils.data.Dataset, but could make an fuction
    #self.train, self.val, self.test = load_datasets() 
    
    train_and_val, self.test = torch.utils.data.random_split(self.yesno_data, [int(self.dataset_size*0.9), self.dataset_size - int(self.dataset_size*0.9)])
    train_and_val_size = len(train_and_val)
    self.train, self.val = torch.utils.data.random_split(train_and_val, [int(train_and_val_size*0.9), train_and_val_size - int(train_and_val_size*0.9)])

  def train_dataloader(self):
    #transforms = ...
    return DataLoader(self.train) #transforms)

  def val_dataloader(self):
    #transforms = ...
    return DataLoader(self.val) #, transforms)

  def test_dataloader(self):
    #transforms = ...
    return DataLoader(self.test) #, transforms)
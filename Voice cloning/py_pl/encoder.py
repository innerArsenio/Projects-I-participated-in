from depen import *

class Encoder_example(nn.Module):
    def __init__(self, input_size, d):
        super(Encoder_example, self).__init__()
        self.network = nn.Sequential([nn.Linear(in_features = input_size, out_features = 2*input_size),
                                      nn.Tanh(),
                                      nn.Linear(in_features = 2*input_size, out_features = d),
                                      ])
    def forward(self, x):
        return self.network(x)



class Encoder_example_light(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        #model type can also be provided as a hparam, like which model we want
        self.model = Encoder_example(input_size = self.hparams.input_size_encoder, d = self.hparams.d_encoder)
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr_encoder)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def training_step_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss_e': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss_e'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {'test_loss_e': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss_e'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    



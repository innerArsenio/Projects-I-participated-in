import torch
import torch.nn as nn


def cnn_layers(in_channels, batch_norm=False):

    config = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M"
    ]


    layers = []

    for v in config:

        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]


        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [
                    conv2d,
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]


            in_channels = v

    return nn.Sequential(*layers)


def fc_layers(num_classes):


    return nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, num_classes),
    )


class vgg16(nn.Module):
    def __init__(self, num_classes, channels=3):



        super(vgg16, self).__init__()


        self.name = "vgg16"
        self.num_classes = num_classes

        self.features = cnn_layers(channels)
        self.classifier = fc_layers(num_classes)

        self.init_weights()


        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x



    def init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, saved_dict, ignore_keys=[]):


        state_dict = self.state_dict()
        saved_dict = list(saved_dict.items())


        for i, (key, val) in enumerate(state_dict.items()):

            n_val = saved_dict[i][1]

            if (
                key not in ignore_keys
                and val.shape == n_val.shape
            ):
                state_dict[key] = n_val



        self.load_state_dict(state_dict)




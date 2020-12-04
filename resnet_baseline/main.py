import torchvision.models as models
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torch.nn import functional as F

class ImageNetDataModule(LightningDataModule):
    

class ImagenetTransferLearning(LightningModule):

    ###



    ###
    def __init__(self):
        # init a pretrained resnet
        num_target_classes = 3
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.eval()

        # use the pretrained model to classify cifar-10 (10 image classes)
        self.classifier = nn.Linear(2048, num_target_classes)

    def forward(self, x):
        representations = self.feature_extractor(x)
        x = self.classifier(representations)


def main():
    model = ImagenetTransferLearning()
    trainer = Trainer()
    trainer.fit(model)
    trainer.test(model)

if __name__=='__main__':
    main()
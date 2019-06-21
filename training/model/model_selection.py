from model.backbone import Resnet
from model.classifier import Classifier
import torch.nn as nn


class AttributeModel(nn.Module):

    def __init__(self, attributes_number):
        super(AttributeModel, self).__init__()
        self.backbone, self.outnodes = Resnet.get_model("resnet18")
        self.classifier = Classifier(
            self.outnodes, attributes_number)

    def forward(self, x):
        x1 = self.backbone(x)
        # x1 = x1.view(-1, self.outnodes)
        # x1 = self.classifier(x1)
        return x1

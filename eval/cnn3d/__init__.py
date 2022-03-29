import torch
import torch.nn as nn
from torch.autograd import Variable

from metrics import acc_score_tensor


class Evaluator:
    def __init__(self, val_loader, model, device, criterion=nn.CrossEntropyLoss()):
        self.device = device
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        total_loss, total_acc = 0., 0.
        length = 0
        self.model.eval()
        for i, (x, y) in enumerate(self.val_loader, 0):
            bx = Variable(x).to(self.device)
            by = Variable(y).to(self.device)
            output = self.model(bx)
            loss = self.criterion(output, by)
            total_loss += loss.item() * by.shape[0]
            total_acc += acc_score_tensor(output, by)*by.shape[0]
            length += by.shape[0]
            del bx, by
        return {"loss": total_loss/length, "acc": total_acc/length}
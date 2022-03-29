import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from eval.cnn3d import Evaluator
from metrics import acc_score_tensor
from models import weights_init
from utils.torchtools import EarlyStopping


class Trainer:
    def __init__(self,
                 model,
                 train_data,
                 val_data,
                 initialize=True,
                 batch_size=32,
                 epochs=10,
                 optimizer="adam",
                 lr=1e-2,
                 momentum=0.5,
                 step_size=20,
                 interval=1,
                 patience=20,
                 # include_weight=True,
                 path="output/checkpoints/checkpoint.pt"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size)
        self.batch_size=batch_size
        self.epochs = epochs

        if initialize:
            model.apply(weights_init)

        self.model = model.to(device=self.device)
        # Optimizer
        if optimizer == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=step_size, gamma=0.5)
        self.interval = interval
        # self.include_weight = include_weight
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    def __call__(self, *args, **kwargs):
        print("Start training")
        for epoch in range(self.epochs):
            print(f"At epoch {epoch}")
            epoch_loss, epoch_acc = 0., 0.
            length = 0
            for i, (x, y) in enumerate(tqdm(self.train_loader), 0):
                self.model.train()
                b_x = Variable(x).to(self.device)
                b_y = Variable(y).to(self.device)
                output = self.model(b_x)
                loss = self.criterion(output, b_y)
                epoch_loss += loss.item()*b_y.shape[0]
                epoch_acc += acc_score_tensor(output, b_y)*b_y.shape[0]
                length += b_y.shape[0]

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.scheduler.step()
                del b_x, b_y

                if i % self.interval == 0:
                    evaluator = Evaluator(self.val_loader, self.model, self.device)
                    results = evaluator()
                    eval_loss, eval_acc = results["loss"], results["acc"]
                    print(f"train loss: {epoch_loss / length}, train accuracy: {epoch_acc / length}; "
                          f"eval loss: {eval_loss}, eval accuracy: {eval_acc}")
                    self.early_stopping(loss, self.model)
            if self.early_stopping.early_stop:
                break
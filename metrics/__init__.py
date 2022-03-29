import torch


def acc_score_tensor(pred, label):
    _, predicted = torch.max(pred, 1)
    correct = (predicted == label).sum().item()
    return correct/label.shape[0]


def acc_score_binary_tensor(pred, label):
    predicted = torch.round(pred.reshape(-1))
    correct = (predicted == label).sum().item()
    return correct/label.shape[0]
import torch


def accuracy(prediction, target):
    """Computes the precision@k for the specified values of k"""
    target = target / 2 + 0.5
    mask = (target != 0.5)
    acc = ((target == prediction.round()) * mask).sum() / mask.sum()
    return acc

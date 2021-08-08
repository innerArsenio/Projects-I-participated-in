import time
import shutil
import torch

# global variables
best_acc1 = 0


class AvgMet(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def trainEpoch(train_loader, model, criterion, optimizer, epoch):


    losses = AvgMet()
    t1 = AvgMet()
    t5 = AvgMet()

    cuda_exists = torch.cuda.is_available()

    # switch to train mode
    model.train()

    for _, (input, target) in enumerate(train_loader):

        if cuda_exists:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = measureAccuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        t1.update(acc1[0], input.size(0))
        t5.update(acc5[0], input.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        print("accuracy ",t1.avg)

    print("")


def validTest(valid_loader, model, criterion):


    losses = AvgMet()
    t1 = AvgMet()
    t5 = AvgMet()

    cuda_exists = torch.cuda.is_available()

    # switch to evaluate mode
    model.eval()


    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):

            if cuda_exists:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = measureAccuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            t1.update(acc1[0], input.size(0))
            t5.update(acc5[0], input.size(0))

            print("accuracy ",t1.avg)

        print("")

    return t1.avg


def measureAccuracy(output, target, topk=(1,)):

    num_classes = 1
    for dim in output.shape[1:]:
        num_classes *= dim

    with torch.no_grad():
        maxk = max(topk)
        maxk = min(maxk, num_classes)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k < num_classes:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                res.append([0, 0])

        return res


def train(
    model,
    loaders,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    epochs=3,
    checkpoint=None,
):


    global best_acc1

    # create model
    print("=> training", model.name)

    # unpack loaders
    train_loader, valid_loader = loaders
    print(torch.cuda.is_available())
    # find device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("=> found cuda compatible gpu")
    else:
        device = torch.device("cpu")
        print("=> no cuda devices found, using cpu for training")

    # device switches and optimization
    torch.backends.cudnn.benchmark = True

    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device=device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum, weight_decay=weight_decay,
    )

    # resume from a checkpoint
    if checkpoint:
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    else:
        start_epoch = 0

    crtm = time.ctime().split()[1:-1]



    for ep in range(start_epoch, epochs):


        lr_adj = lr * (0.1 ** (ep // 30))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_adj


        trainEpoch(
            train_loader, model, criterion, optimizer, ep,
        )


        acc1 = validTest(valid_loader, model, criterion)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)


        save_dict = {
            "epoch": ep + 1,
            "arch": model.name,
            "best_acc1": best_acc1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(save_dict, "checkpoint.pth")


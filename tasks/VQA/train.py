import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from timeit import default_timer as timer
from time import gmtime, strftime


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(args, model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)

    # change the learning rate here. 
    params = []
    if args.bert:

        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'bert' in key:
                    params += [{'params':[value], 'lr':2e-5}]
                else:
                    params += [{'params':[value]}]
    else:
        params = model.parameters()

    optim = torch.optim.Adamax(params)

    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    start_t = timer()
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        ave_loss = 0
        for i, batch in enumerate(train_loader):
            batch = tuple(t.cuda() for t in batch)
            
            if args.bert:
                v, b, q, a, m, s = batch
                pred = model(v, b, q, a, m, s)
            else:
                v, b, q, a = batch
                pred = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score
            
            ave_loss += loss.item()
            if i % 20 == 0:
                ave_loss = ave_loss / 20
                end_t = timer()  # Keeping track of iteration(s) time
                timeStamp = strftime('%a %d %b %y %X', gmtime())

                printFormat = '[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.5g]'
                printInfo = [
                    timeStamp, epoch, i, end_t - start_t, ave_loss
                ]
                start_t = end_t
                print(printFormat % tuple(printInfo)) 
                ave_loss = 0

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(args, model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def evaluate(args, model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for batch in iter(dataloader):
        batch = tuple(t.cuda() for t in batch)

        if args.bert:
            v, b, q, a, m, s = batch
            pred = model(v, b, q, a, m, s)
        else:
            v, b, q, a = batch
            pred = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound

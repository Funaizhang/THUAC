import torch
import argparse
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torchtext import data, datasets
from model import Model
from utils import LOG_INFO, plot_loss_acc
import cell

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--max_vocab_size", default=25000, type=int, help="vocabulary size.")
parser.add_argument("--n_labels", default=5, type=int, help="Number of labels.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epoch.")
parser.add_argument("--embedding_dim", default=300, type=int, help="Size of word embedding.")
parser.add_argument("--hidden_dim", default=512, type=int, help="Size of each model layer.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=50, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--cell_type", default='GRUCell', type=str,
                    choices=['RNNCell', 'GRUCell', 'LSTMCell'], help="Available rnn cells")

args = parser.parse_args()
print(args)

TEXT = data.Field()
LABEL = data.LabelField(dtype=torch.float)

train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, max_size=args.max_vocab_size,
                 vectors="glove.6B.300d", unk_init=None)

LABEL.build_vocab(train_data)

device = 'cpu'

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=args.batch_size, device=device
)

input_dim = len(TEXT.vocab)
output_dim = args.n_labels

rnncell = cell.__dict__[args.cell_type]
model = Model(rnncell, input_dim, args.embedding_dim, args.hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

model = model.to(device)
criterion = criterion.to(device)

def train(epoch, model, iterator, optimizer, criterion):
    loss_list = []
    acc_list = []

    model.train()

    for i, batch in enumerate(iterator):
        optimizer.zero_grad()
        # every column of batch.text is a sentence
        predictions = model(batch.text)
        loss = criterion(predictions, batch.label.long())
        loss.backward()
        optimizer.step()

        acc = (predictions.max(1)[1] == batch.label.long()).float().mean()
        loss_list.append(loss.item())
        acc_list.append(acc.item())

        if i % args.display_freq == 0:
            msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.4f, train acc = %.4f" % (
                epoch, i, len(iterator), np.mean(loss_list), np.mean(acc_list)
            )
            train_loss = np.mean(loss_list)
            train_acc = np.mean(acc_list)
            LOG_INFO(msg)
            loss_list.clear()
            acc_list.clear()
        
    return train_loss, train_acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)
            loss = criterion(predictions, batch.label.long())

            acc = (predictions.max(1)[1] == batch.label.long()).float().mean()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


train_loss_list = []
train_acc_list = []
valid_loss_list = []
valid_acc_list = []

best_acc = 0
best_epoch = -1
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train(epoch, model, train_iterator, optimizer, criterion)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    valid_loss_list.append(valid_loss)
    valid_acc_list.append(valid_acc)
    
    msg = '...Epoch %02d, val loss = %.4f, val acc = %.4f' % (
        epoch, valid_loss, valid_acc
    )
    LOG_INFO(msg)

    if valid_acc > best_acc:
        best_acc = valid_acc
        best_epoch = epoch
        torch.save(model.state_dict(), 'best-model-%s.pth' % args.cell_type)

LOG_INFO('Test best model @ Epoch %02d' % best_epoch)
model.load_state_dict(torch.load('best-model-%s.pth' % args.cell_type))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
LOG_INFO('Finally, test loss = %.4f, test acc = %.4f' % (test_loss, test_acc))

plot_loss_acc(args.epochs, train_acc_list, valid_acc_list, train_loss_list, valid_loss_list, "GRUCell_None")
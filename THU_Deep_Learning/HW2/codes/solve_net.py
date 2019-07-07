from utils import LOG_INFO, onehot_encoding, calculate_acc
import numpy as np


def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, inputs, labels, batch_size, disp_freq):

    iter_counter = 0
    loss_list = []
    acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size):
        target = onehot_encoding(label, 10)
        iter_counter += 1

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value = loss.forward(output, target)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)

        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

        if iter_counter % disp_freq == 0:
            temp_loss = np.mean(loss_list)
            temp_acc = np.mean(acc_list)
            msg = '  Training iter %d, avg loss %.4f, avg acc %.4f' % (iter_counter, temp_loss, temp_acc)
            loss_list = []
            acc_list = []
            LOG_INFO(msg)

    return temp_loss, temp_acc


def test_net(model, loss, inputs, labels, batch_size, epoch, layer_name):
    loss_list = []
    acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        target = onehot_encoding(label, 10)
        output, output_visualize = model.forward(input, visualize=True, layer_name=layer_name)
        # collapse output_visualize into 1 channel
        output_visualize = np.sum(output_visualize, axis=(1))
        
        loss_value = loss.forward(output, target)
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (np.mean(loss_list), np.mean(acc_list))
    LOG_INFO(msg)
    
    # save weights and biases
    model.save_weights(loss.name, epoch)
    
    return np.mean(loss_list), np.mean(acc_list), output_visualize # output_visualize: batch_size x height x width

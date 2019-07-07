from network import Network
from utils import LOG_INFO, make_plot
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import time


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture

model1 = Network(name='model1')
model1.add(Linear('m1_fc1', 784, 256, 0.01))
model1.add(Sigmoid('m1_fc1'))
model1.add(Linear('m1_fc3', 256, 10, 0.01))

model2 = Network(name='model2')
model2.add(Linear('m2_fc1', 784, 256, 0.01))
model2.add(Relu('m2_fc2'))
model2.add(Linear('m2_fc3', 256, 10, 0.01))


model3 = Network(name='model3')
model3.add(Linear('m3_fc1', 784, 512, 0.01))
model3.add(Sigmoid('m3_fc2'))
model3.add(Linear('m3_fc3', 512, 128, 0.01))
model3.add(Sigmoid('m3_fc4'))
model3.add(Linear('m3_fc5', 128, 10, 0.01))

model4 = Network(name='model4')
model4.add(Linear('m4_fc1', 784, 512, 0.01))
model4.add(Relu('m4_fc2'))
model4.add(Linear('m4_fc3', 512, 128, 0.01))
model4.add(Relu('m4_fc4'))
model4.add(Linear('m4_fc5', 128, 10, 0.01))


model5 = Network(name='model5')
model5.add(Linear('m5_fc1', 784, 392, 0.01))
model5.add(Relu('m5_fc2'))
model5.add(Linear('m5_fc3', 392, 196, 0.01))
model5.add(Relu('m5_fc4'))
model5.add(Linear('m5_fc5', 196, 10, 0.01))


loss1 = EuclideanLoss(name='Euclidean')
loss2 = SoftmaxCrossEntropyLoss(name='XEntropy')

#models = [model1, model2, model3, model4, model5]
#losses = [loss1, loss2]
model = model4
loss = loss2

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.


config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 30,
    'disp_freq': 300,
    'convergence': 0.001,
}


#for loss in losses:
#    for model in models:
#        
#        train_loss_list = []
#        train_acc_list = []
#        test_loss_list = []
#        test_acc_list = []
#        
#        for epoch in range(config['max_epoch']):
#            LOG_INFO('Training @ %d epoch...' % (epoch))
#            train_loss, train_acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
#            
#            # save loss & accuracy data for training epoch
#            train_loss_list.append(train_loss)
#            train_acc_list.append(train_acc)
#            
#            LOG_INFO('Testing @ %d epoch...' % (epoch))
#            test_loss, test_acc = test_net(model, loss, test_data, test_label, config['batch_size'])
#            
#            # save loss & accuracy data for test epoch
#            test_loss_list.append(test_loss)
#            test_acc_list.append(test_acc)
#        
#        make_plot(config['max_epoch'], train_loss_list, test_loss_list, model.name, loss.name, loss=True)
#        make_plot(config['max_epoch'], train_acc_list, test_acc_list, model.name, loss.name, loss=False)
        
              
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

start_time = time.time()

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_loss, train_acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    
    # save loss & accuracy data for training epoch
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    
    LOG_INFO('Testing @ %d epoch...' % (epoch))
    test_loss, test_acc = test_net(model, loss, test_data, test_label, config['batch_size'])
    
    # save loss & accuracy data for test epoch
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    
end_time = time.time()

make_plot(config['max_epoch'], train_loss_list, test_loss_list, model.name, loss.name, loss=True)
make_plot(config['max_epoch'], train_acc_list, test_acc_list, model.name, loss.name, loss=False)
print('Time per epoch = ' + str((end_time-start_time)/config['max_epoch']))
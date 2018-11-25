import numpy as np
import theano
from theano import tensor as T
from theano.tensor.signal import pool
import pickle
import time
theano.config.gcc.cxxflags = "-D_hypot=hypot"

########################################################################
# practical variables
grid_search = True
print_intermediate = False
plot_anything = False

# parameters
if grid_search:
    version = 1
    iterations = 3
    train_epochs = 2
if print_intermediate:
    version = 1
    learning_rate = 1e-4
    depth_per_filter = [12]
    size_of_minibatch = 16
    train_epochs = 3
    
########################################################################

# read data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict 
batch1 = unpickle('./cifar-10-batches-py/data_batch_1')
batch2 = unpickle('./cifar-10-batches-py/data_batch_2')
batch3 = unpickle('./cifar-10-batches-py/data_batch_3')
batch4 = unpickle('./cifar-10-batches-py/data_batch_4')
batch5 = unpickle('./cifar-10-batches-py/data_batch_5')
test_batch = unpickle('./cifar-10-batches-py/test_batch')
batches_meta = unpickle('./cifar-10-batches-py/batches.meta')


# create train and test arrays
train_data = np.asarray(batch1[b'data'], dtype = theano.config.floatX) # uint8 -> float64
train_labels = np.asarray(batch1[b'labels'], dtype=np.int32) # int32
train_data = np.concatenate((train_data, 
    np.asarray(batch2[b'data'], dtype = theano.config.floatX)), axis = 0)
train_labels = np.concatenate((train_labels, 
    np.asarray(batch2[b'labels'], dtype=np.int32)), axis = 0)
train_data = np.concatenate((train_data, 
    np.asarray(batch3[b'data'], dtype = theano.config.floatX)), axis = 0)
train_labels = np.concatenate((train_labels, 
    np.asarray(batch3[b'labels'], dtype=np.int32)), axis = 0)
train_data = np.concatenate((train_data, 
    np.asarray(batch4[b'data'], dtype = theano.config.floatX)), axis = 0)
train_labels = np.concatenate((train_labels, 
    np.asarray(batch4[b'labels'], dtype=np.int32)), axis = 0)
train_data = np.concatenate((train_data, 
    np.asarray(batch5[b'data'], dtype = theano.config.floatX)), axis = 0)
train_labels = np.concatenate((train_labels, 
    np.asarray(batch5[b'labels'], dtype=np.int32)), axis = 0)
test_data = np.asarray(test_batch[b'data'], dtype = theano.config.floatX) # uint8 -> float64
test_labels = np.asarray(test_batch[b'labels'], dtype=np.int32) # int32


# standardize data and store in shared variables
train_data_shared = theano.shared((train_data - np.mean(train_data,axis=0)) 
        / np.std(train_data, axis=0), borrow=True)
test_data_shared = theano.shared((test_data - np.mean(test_data,axis=0)) 
        / np.std(test_data, axis=0), borrow = True)
train_labels_shared = theano.shared(train_labels, borrow = True)
test_labels_shared = theano.shared(test_labels, borrow = True)


# check distribution of train_labels
y_freq = np.array([ np.sum(i == train_labels) for i in np.arange(10) ])
y_freq_plt = np.insert(y_freq,0,y_freq[0])

if(plot_anything):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.step(np.linspace(-0.5,9.5,11),y_freq_plt / len(train_labels))
    # plt.show()





def ClassificationCNN(Xtrain,
                        Ytrain,
                        Xtest,
                        Ytest,
                        num_epochs = 10,
                        mini_batch_size = 32,
                        alpha = 0.1,
                        num_filters = [9]):
    n_train_batches = Xtrain.get_value(borrow=True).shape[0] // mini_batch_size
    n_test_batches = Xtest.get_value(borrow=True).shape[0] // mini_batch_size
    if(print_intermediate):
        print('Number of train batches: %d' %n_train_batches)
        print('Number of test batches: %d' %n_test_batches)
        
    def convLayer(data_input, filter_spec, image_spec, pool_size, 
                                activation, border_mode = 'half',
                                rng = np.random.RandomState(23455)):
        # Function that creates a convolution layer
        
        W = theano.shared(
            np.asarray(rng.normal(loc=0, scale=0.1, size=filter_spec)),
            name = 'W_conv', borrow=True)
        
        b = theano.shared(np.zeros((filter_spec[0],)) + 0.01, 
            name='b_conv', borrow=True)
        
        conv_op_out = T.nnet.conv2d(
            input=data_input,
            filters=W,
            filter_shape=filter_spec,
            border_mode = border_mode,
            input_shape=image_spec)
        
        layer_activation = activation(conv_op_out + b.dimshuffle('x', 0, 'x', 'x'))
        
        if(pool_size == None):
            output = layer_activation
        else:
            output = pool.pool_2d(input=layer_activation, ws=pool_size, ignore_border = True)
        
        params = [W, b]
        return output, params
        
        
    def fullyConnectedLayer(data_input, num_in, num_out, 
                                        rng = np.random.RandomState(23455)):
        # Function to create the fully-connected layer using softmax function
        
        W = theano.shared(
            value=np.asarray(
                rng.normal(loc=0, scale=0.1, size=(num_in, num_out))),
            name='W',
            borrow=True)
                       
        b = theano.shared(
            value=np.zeros((num_out,)) + 0.01,
            name='b',
            borrow=True)
        
        p_y_given_x = T.nnet.softmax(T.dot(data_input, W) + b)
        
        y_pred = T.argmax(p_y_given_x, axis=1)

        params = [W, b]
        return p_y_given_x, y_pred, params
    
    xdim = Xtrain.shape[1]
    
    mb_index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    # Five different neural nets created
    if(v == 1):
        # 1 convolution layer followed by fully connected layer
        layer0_input = x.reshape((mini_batch_size, 3, 32, 32))
        [layer0_output, layer0_params] = convLayer(
            data_input=layer0_input,
            image_spec=(mini_batch_size, 3, 32, 32),
            filter_spec=(num_filters[0], 3, 5, 5),
            border_mode = 'half',
            pool_size=(2, 2),
            activation=T.tanh) # 9x28x28 -> 9x14x14
        fc_layer_input = layer0_output.flatten(2)
        [p_y_given_x, y_pred, fc_layer_params] = fullyConnectedLayer(
            data_input=fc_layer_input,
            num_in=num_filters[0]*16*16,
            num_out=10)  
        params = layer0_params + fc_layer_params
        
    if(v == 2):
        # 2 convolution layers followed by fully connected layer
        layer0_input = x.reshape((mini_batch_size, 3, 32, 32))
        [layer0_output, layer0_params] = convLayer(
            data_input=layer0_input,
            image_spec=(mini_batch_size, 3, 32, 32),
            filter_spec=(num_filters[0], 3, 5, 5),
            border_mode = 'half',
            pool_size=(2, 2),
            activation=T.nnet.relu) # 9x32x32 -> 9x16x16
        [layer1_output, layer1_params] = convLayer(
            data_input=layer0_output,
            image_spec=(mini_batch_size, num_filters[0], 16, 16),
            filter_spec=(num_filters[1], num_filters[0], 5, 5),
            border_mode = 'half',
            pool_size=(2, 2),
            activation=T.nnet.relu) # 9x16x16 -> 9x8x8
        fc_layer_input = layer1_output.flatten(2)
        [p_y_given_x, y_pred, fc_layer_params] = fullyConnectedLayer(
            data_input=fc_layer_input,
            num_in=num_filters[1]*8*8,
            num_out=10)
        params = layer0_params + layer1_params + fc_layer_params
        
    if(v == 3):
        # 3 convolution layers followed by fully connected layer
        layer0_input = x.reshape((mini_batch_size, 3, 32, 32))
        [layer0_output, layer0_params] = convLayer(
            data_input=layer0_input,
            image_spec=(mini_batch_size, 3, 32, 32),
            filter_spec=(num_filters[0], 3, 3, 3),
            border_mode = 'half',
            pool_size=None,
            activation=T.nnet.relu) # 9x32x32 -> 9x32x32
        [layer1_output, layer1_params] = convLayer(
            data_input=layer0_output,
            image_spec=(mini_batch_size, num_filters[0], 32, 32),
            filter_spec=(num_filters[1], num_filters[0], 3, 3),
            border_mode = 'half',
            pool_size=(2, 2),
            activation=T.nnet.relu) # 9x32x32 -> 9x16x16
        [layer2_output, layer2_params] = convLayer(
            data_input=layer1_output,
            image_spec=(mini_batch_size, num_filters[1], 16, 16),
            filter_spec=(num_filters[2], num_filters[1], 5, 5),
            border_mode = 'half',
            pool_size=(2, 2),
            activation=T.nnet.relu) # 9x16x16 -> 9x8x8    
        fc_layer_input = layer2_output.flatten(2)
        [p_y_given_x, y_pred, fc_layer_params] = fullyConnectedLayer(
            data_input=fc_layer_input,
            num_in=num_filters[2]*8*8,
            num_out=10)
        params = layer0_params + layer1_params + layer2_params + fc_layer_params
        
    if(v == 4):
        # 4 convolution layers followed by fully connected layer
        layer0_input = x.reshape((mini_batch_size, 3, 32, 32))
        [layer0_output, layer0_params] = convLayer(
            data_input=layer0_input,
            image_spec=(mini_batch_size, 3, 32, 32),
            filter_spec=(num_filters[0], 3, 3, 3),
            border_mode = 'half',
            pool_size=None,
            activation=T.nnet.relu) # 9x32x32 -> 9x32x32
        [layer1_output, layer1_params] = convLayer(
            data_input=layer0_output,
            image_spec=(mini_batch_size, num_filters[0], 32, 32),
            filter_spec=(num_filters[1], num_filters[0], 3, 3),
            border_mode = 'half',
            pool_size=(2, 2),
            activation=T.nnet.relu) # 9x32x32 -> 9x16x16
        [layer2_output, layer2_params] = convLayer(
            data_input=layer1_output,
            image_spec=(mini_batch_size, num_filters[1], 16, 16),
            filter_spec=(num_filters[2], num_filters[1], 3, 3),
            border_mode = 'half',
            pool_size=None,
            activation=T.nnet.relu) # 9x16x16 -> 9x16x16   
        [layer3_output, layer3_params] = convLayer(
            data_input=layer2_output,
            image_spec=(mini_batch_size, num_filters[2], 16, 16),
            filter_spec=(num_filters[3], num_filters[2], 3, 3),
            border_mode = 'half',
            pool_size=(2,2),
            activation=T.nnet.relu) # 9x16x16 -> 9x8x8            
        fc_layer_input = layer3_output.flatten(2)
        [p_y_given_x, y_pred, fc_layer_params] = fullyConnectedLayer(
            data_input=fc_layer_input,
            num_in=num_filters[3]*8*8,
            num_out=10)
        params = layer0_params + layer1_params + layer2_params + \
                layer3_params + fc_layer_params
                
    if(v == 5):
        # 6 convolution layers followed by fully connected layer
        layer0_input = x.reshape((mini_batch_size, 3, 32, 32))
        [layer0_output, layer0_params] = convLayer(
            data_input=layer0_input,
            image_spec=(mini_batch_size, 3, 32, 32),
            filter_spec=(num_filters[0], 3, 3, 3),
            border_mode = 'half',
            pool_size=None,
            activation=T.nnet.relu) # 9x32x32 -> 9x32x32
        [layer1_output, layer1_params] = convLayer(
            data_input=layer0_output,
            image_spec=(mini_batch_size, num_filters[0], 32, 32),
            filter_spec=(num_filters[1], num_filters[0], 3, 3),
            border_mode = 'half',
            pool_size=(2, 2),
            activation=T.nnet.relu) # 9x32x32 -> 9x16x16
        [layer2_output, layer2_params] = convLayer(
            data_input=layer1_output,
            image_spec=(mini_batch_size, num_filters[1], 16, 16),
            filter_spec=(num_filters[2], num_filters[1], 3, 3),
            border_mode = 'half',
            pool_size=None,
            activation=T.nnet.relu) # 9x16x16 -> 9x16x16   
        [layer3_output, layer3_params] = convLayer(
            data_input=layer2_output,
            image_spec=(mini_batch_size, num_filters[2], 16, 16),
            filter_spec=(num_filters[3], num_filters[2], 3, 3),
            border_mode = 'half',
            pool_size=(2,2),
            activation=T.nnet.relu) # 9x16x16 -> 9x8x8  
        [layer4_output, layer4_params] = convLayer(
            data_input=layer3_output,
            image_spec=(mini_batch_size, num_filters[3], 8, 8),
            filter_spec=(num_filters[4], num_filters[3], 3, 3),
            border_mode = 'half',
            pool_size=None,
            activation=T.nnet.relu) # 9x8x8 -> 9x8x8
        [layer5_output, layer5_params] = convLayer(
            data_input=layer4_output,
            image_spec=(mini_batch_size, num_filters[4], 8, 8),
            filter_spec=(num_filters[5], num_filters[4], 3, 3),
            border_mode = 'half',
            pool_size=(2,2),
            activation=T.nnet.relu) # 9x8x8 -> 9x4x4
        fc_layer_input = layer5_output.flatten(2)
        [p_y_given_x, y_pred, fc_layer_params] = fullyConnectedLayer(
            data_input=fc_layer_input,
            num_in=num_filters[5]*4*4,
            num_out=10)
        params = layer0_params + layer1_params + layer2_params + \
                layer3_params  + \
                layer4_params + layer5_params + \
                fc_layer_params
    
    cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, y))
    
    
    def gradient_updates_Adam(cost, params, learning_rate = alpha):
        updates=[]
        eps=1e-4# small constant used for numerical stabilization.
        beta1=0.9
        beta2=0.999
        for param in params:       
            t=theano.shared(1)
            s=theano.shared(param.get_value(borrow=True)*0.)
            r=theano.shared(param.get_value(borrow=True)*0.)
            s_new=beta1*s+(1.0-beta1)*T.grad(cost, param)
            r_new=beta2*r+(1.0-beta2)*(T.grad(cost, param)**2)
            updates.append((s, s_new))
            updates.append((r, r_new))
            s_hat=s_new/(1-beta1**t)
            r_hat=r_new/(1-beta2**t)
            updates.append((param, param-learning_rate*s_hat/(np.sqrt(r_hat)+eps) ))
        updates.append((t, t+1))
        return updates

    updates = gradient_updates_Adam(cost, params)
    
    
    def accuracy(y, y_pred):
        acc = T.mean(T.eq(y_pred, y))
        return acc
    
    # train model
    train_model_mb = theano.function(
            inputs   = [mb_index],
            outputs = [cost, accuracy(y, y_pred), y_pred],
            updates = updates,
            givens = {
                x: Xtrain[mb_index * mini_batch_size:
                    (mb_index + 1) * mini_batch_size
                    ],
                y: Ytrain[mb_index * mini_batch_size:
                    (mb_index + 1) * mini_batch_size
                    ]}
            )
    
    # test model performance
    asses_model_test = theano.function(
            inputs = [mb_index],
            outputs = [cost, accuracy(y, y_pred), y_pred],
            givens={
                x: Xtest[
                    mb_index * mini_batch_size:
                    (mb_index + 1) * mini_batch_size
                    ],
                y: Ytest[
                    mb_index * mini_batch_size:
                    (mb_index + 1) * mini_batch_size
                    ]})
    
    ##########################################################
    # TRAINING
    cost_train_vec = np.array([])
    acc_train_vec = np.array([])
    cost_test_vec = np.array([])
    acc_test_vec = np.array([])
    cost_test_epoch = np.array([])
    acc_test_epoch = np.array([])
    
    iter = 0
    epoch = 0
    best = [10, 0, 0]
    print("Training")
    while epoch < num_epochs:
        epoch = epoch + 1
        
        y_pred_train = np.array([])
        for mb_idx in range(n_train_batches):
            iter += 1
            cost_train, acc_train, y_pred = train_model_mb(mb_idx)
            cost_train_vec = np.append(cost_train_vec, cost_train)
            acc_train_vec = np.append(acc_train_vec, acc_train)
            y_pred_train = np.append(y_pred_train, y_pred)
        y_pred_test = np.array([])
        for mb_idx in range(n_test_batches):
            cost_test, acc_test, y_pred = asses_model_test(mb_idx)
            cost_test_vec = np.append(cost_test_vec, cost_test)
            acc_test_vec = np.append(acc_test_vec, acc_test)
            y_pred_test = np.append(y_pred_test, y_pred)
        cost_test_epoch = np.append(cost_test_epoch, 
            np.mean(cost_test_vec[-n_test_batches:]))
        acc_test_epoch = np.append(acc_test_epoch, 
            np.mean(acc_test_vec[-n_test_batches:]))
        
        if(cost_test_epoch[-1] < best[0]):
            best = [ cost_test_epoch[-1], acc_test_epoch[-1], epoch ]
                
        # printing
        if(print_intermediate):
            print("Epoch %6s --- "%epoch)
            print('Train (means) -- cost: %4.4f, accuracy: %%.3f' 
                %np.mean(cost_train_vec[-n_train_batches:]) 
                %np.mean(acc_train_vec[-n_train_batches:]))
            print("Test (means)  -- cost: %4.4f, accuracy: %%.3f\n" 
                %np.mean(cost_test_vec[-n_test_batches:]) 
                %np.mean(acc_test_vec[-n_test_batches:]))
    
    ############################################################
    # print Confidence Matrix
    if(print_intermediate):
        print("\n\n")
        print(" %%%%%% TRAIN %%%%%%%%")
        confmat_train = np.zeros((10,10), dtype=np.int)
        for i in range(n_train_batches * mini_batch_size) :
            confmat_train[Ytrain.eval()[i], int(y_pred_train[i])] += 1
        print((confmat_train / np.sum(confmat_train, axis = 1).reshape(-1,1)).round(2))
        print("\n\n %%%%%%%%%%% TEST %%%%%%%%%")
        confmat_test = np.zeros((10,10), dtype=np.int)
        for i in range(n_test_batches * mini_batch_size) :
            confmat_test[Ytest.eval()[i], int(y_pred_test[i])] += 1
        print((confmat_test / np.sum(confmat_test, axis = 1).reshape(-1,1)).round(2))
        
    
    # PLOT FIGURES
    if(plot_anything):
        # cost
        plt.figure()
        plt.plot(range(iter), cost_train_vec)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')

        plt.figure()
        # plt.plot(range(len(cost_test_vec)), cost_test_vec)
        plt.plot(range(len(cost_test_epoch)), cost_test_epoch)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        
        weights_channelwise = np.array(layer0_params[0].get_value())
        fig_wt, ax_wt = plt.subplots(num_filters[0] // 3, 3, sharex='col', sharey='row',
                                gridspec_kw=dict(wspace=0.02,
                                                 hspace=0.02, 
                                                 top=0.95,
                                                 bottom=0.05,
                                                 left=0.17,
                                                 right=0.845))
        
        for i in range(num_filters[0] // 3):
            for j in range(3):
                idx = 3 * i + j;
                channel = weights_channelwise[idx].mean(axis = 0)
                ax_wt[i, j].set_xticklabels([])
                ax_wt[i, j].set_yticklabels([])
                ax_wt[i, j].set_xticks([])
                ax_wt[i, j].set_yticks([])
                im = ax_wt[i, j].imshow(channel,
                                        cmap='gray',
                                        interpolation='None')
        plt.show()
    
    
    if(grid_search):
        print(loop_i)
        print("Epoch: %d, mb: %d, alpha: %f" %(best[2], mini_batch_size, alpha))
        print(num_filters)
        print("Version %d" %v)
        print("Cost: %f" %best[0])
        print("Acc: %f" %best[1])
        
    return [ best[0], best[1], mini_batch_size, alpha, num_filters, best[2] ]
    
if(print_intermediate):
    np.random.seed(12346)
    print("start")
    v = version
    print("Version %d" %v)
    start = time.time()
    ClassificationCNN(train_data_shared, train_labels_shared, 
            test_data_shared, test_labels_shared, 
            num_epochs = train_epochs, mini_batch_size = size_of_minibatch,
            alpha = learning_rate, num_filters = depth_per_filter)
    end = time.time()
    print("Elapsed time %f min" %((end-start)/60))
        
if(grid_search):
    best_epoch = [3]
    np.random.seed()
    print("start")
    v = version
    start = time.time()
    for loop_i in range(iterations):
        alpha = np.max((1e-4,np.random.normal(1e-3, 4e-3)))
        # num_filters = list(np.random.randint(3,30, size=3))
        num_filters = [ np.max( (3,int(i)) ) for i in np.random.normal(10,5, size=6) ]
        if(np.min(num_filters) < 1):
            print("assertion error")
            continue
        mini_batch_size = 2**(np.max((2, int(np.random.normal(5,2)))))
        result_list = ClassificationCNN(train_data_shared, train_labels_shared, 
                test_data_shared, test_labels_shared, 
                num_epochs = train_epochs, mini_batch_size = mini_batch_size,
                alpha = alpha, num_filters = num_filters)
        if(best_epoch[0] > result_list[0]):
            best_epoch = result_list
    print("\n\nCost, Acc, mb_size, alpha, num_filters, epoch")        
    print(best_epoch)
    end = time.time()
    print("Elapsed time %f min" %((end-start)/60))

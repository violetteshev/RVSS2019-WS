import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_model(model, data_loaders, optimizer, num_epochs=25, device='cpu'):
    criterion = nn.CrossEntropyLoss() # common loss function for classification
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000.0
    best_acc = 0.0
    val_loss_history = []
    val_acc_history = []

    since = time.time()

    # Run training for specified number of epoches.
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # set model to training mode
            else:
                model.eval() # set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for image, turn in data_loaders[phase]:
                batch_size = image.size(0)
                image = image.to(device) # move image to device (cpu or gpu)
                turn = turn.to(device) # move labels
                optimizer.zero_grad() # zero gradients

                with torch.set_grad_enabled(phase == 'train'):
                    pred_turn = model(image) # run model and get outputs
                    loss = criterion(pred_turn, turn) # calculate loss
                    _, preds = torch.max(pred_turn, 1) # get predictions

                    if phase == 'train':
                        loss.backward() # run backpropagation
                        optimizer.step() # one gradient descent step

                # Statistics of loss value and number of correct predictions.
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == turn.data)

            # Statistics for the whole epoch. 
            dataset_size = len(data_loaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
                        
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                        epoch_acc))
            # Save model weights.
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            # Save statistics.
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    # Load best model weights.
    model.load_state_dict(best_model_wts)
    return model, val_loss_history, val_acc_history

import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from steer_net.steerNet import SteerNet
from steer_net.steerDS import SteerDataSet
from steer_net.train_model import train_model
from matplotlib import pyplot as plt


if __name__ == '__main__':
    batch_size = 32
    num_epochs = 35
    lr = 0.002 # learning rate
    seed = 1234 
    classes = ['left2', 'left1', 'straight', 'right1', 'right2']
    resize_size = 84
    out_dir = os.path.join('trained_models', str(random.randint(0, 100000)))
    use_cuda = torch.cuda.is_available()

    # Initialize random seed.
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    # Create output directory.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        exit('Folder already exists')

    device = torch.device('cuda' if use_cuda else 'cpu')

    # Data transformation.
    data_transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Specify training and validation data and prepare it for loading.
    data = {x: SteerDataSet('dev_data/{0}_data'.format(x),
                            img_ext='.jpg',
                            classes=classes,
                            transform=data_transf,
                            resize=resize_size)
            for x in ['train', 'val']}
    data_loaders = {x: DataLoader(data[x], batch_size=batch_size, num_workers=4)
                    for x in ['train', 'val']}

    print('output dir = ' + out_dir)
    print('use cuda = ' + str(use_cuda))

    # Load model and move it to device (cpu or gpu)
    model = SteerNet(len(classes))
    model.to(device)

    # Specify optimiser.
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=1e-4)

    # Train model, get best model weights and validation statistics. 
    best_model, val_loss_history, val_acc_history = train_model(model,
                                                        data_loaders,
                                                        optimizer,
                                                        num_epochs,
                                                        device)

    # Move model to cpu, because robot doesn't have gpu, and save it.
    best_model.to('cpu')
    model_path = os.path.join(out_dir, 'best_model.pt')
    torch.save(best_model.state_dict(), model_path)
    
    # Save validation loss values to text file.
    val_loss_path = os.path.join(out_dir, 'val_loss.txt')
    with open(val_loss_path, 'w') as f:
        f.writelines(['{}\n'.format(v) for v in val_loss_history])

    # Save validation accuracy values to text file.
    val_acc_path = os.path.join(out_dir, 'val_acc.txt')
    with open(val_acc_path, 'w') as f:
        f.writelines(['{}\n'.format(v) for v in val_acc_history])

    # Save all parameters to text file.
    params_path = os.path.join(out_dir, 'params.txt')
    with open(params_path, 'w') as f:
        params = []
        params.append('seed: {}\n'.format(seed))
        params.append('batch_size: {}\n'.format(batch_size))
        params.append('num_epochs: {}\n'.format(num_epochs))
        params.append('lr: {}\n'.format(lr))
        params.append('classes: {}\n'.format(classes))
        params.append('resize_size: {}\n'.format(resize_size))
        params.append('optimizer: {}\n'.format(optimizer.__str__()))
        f.writelines(params)

    # Plot validation statistics and save plot to file.
    plt.plot(val_loss_history, 'r', label='Loss')
    plt.plot(val_acc_history, 'b', label='Accuracy')
    acc_max = np.argmax(val_acc_history)
    plt.plot(acc_max, val_acc_history[acc_max].item(), 'k.',
             label=str(val_acc_history[acc_max].item()))
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'plot.png'))
    plt.show()

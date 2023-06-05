import torch
import pandas as pd

from tqdm.auto import tqdm
from timeit import default_timer as timer
from torchmetrics import PeakSignalNoiseRatio


# training step function
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = torch.device('cpu')):
    """
    Function used to perform the train step of a model built on PyTorch

    Args:
    `model`         - Model built with PyTorch `nn.Module`
    `dataloader`    - DataLoader to load the data
    `loss_fn`       - Loss function used for the model
    `optimizer`     - Optimizer used to optimize the model
    `device`        - Device to run this function on
    `print_every`   - Print results after every `print_every`th batch

    Returns:
    train loss, train psnr
    """
    # Training mode
    model.train()
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)
    train_loss, train_psnr, total = 0, 0, 0
    for batch, (X, y) in tqdm(enumerate(dataloader), desc='Training'):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_psnr += PSNR(y_pred, y)
    
    train_loss /= len(dataloader)
    train_psnr /= len(dataloader)
    
    return train_loss, train_psnr


# validation step function
def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device = torch.device('cpu')):
    """
    Function used to perform the validation step of a model built on PyTorch

    Args:
    `model`         - Model built with PyTorch `nn.Module`
    `dataloader`    - DataLoader to load the data
    `loss_fn`       - Loss function used to validate the model
    `device`        - Device to run this function on

    Returns:
    validation loss, validation PSNR
    """
    # Evaluation mode
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)
    model.eval()
    with torch.inference_mode():
        test_loss, test_psnr = 0, 0
        for X, y in tqdm(dataloader, desc='Validation'):
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            test_psnr += PSNR(test_pred, y)
        
        test_loss /= len(dataloader)
        test_psnr /= len(dataloader)
        
    return test_loss, test_psnr


# trainer function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          epochs: int,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler=None,
          device: torch.device=torch.device('cpu')):
    """
    Function used to train model built on PyTorch library
    Args:
    `model`             - model built on PyTorch `nn.Module`
    `train_dataloader`  - dataloader for training
    `test_dataloader`   - dataloader for testing
    `epochs`            - number of epochs
    `loss_fn`           - loss function 
    `optimizer`         - Optimizer for the model
    `scheduler`         - Learning rate scheduler
    `device`            - Device to train the model on
    """
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    time_taken = []
    
    train_start = timer()
    for epoch in tqdm(range(epochs)):
        epoch_start = timer()
        print(f"----------------\n| Epoch {epoch + 1}/{epochs} |\n----------------")
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = val_step(model=model,
                                       dataloader=test_dataloader,
                                       loss_fn=loss_fn,
                                       device=device)
        if scheduler:
            scheduler.step()
        epoch_end = timer()
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc.item())
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc.item())
        time_taken.append(epoch_end-epoch_start)
        print("---------------------------------------------------------")
        print(f"| Train loss: {train_loss:.4f} | Train PSNR: {train_acc:.4f} |")
        print(f"| Validation loss: {test_loss:.4f} | Validation PSNR: {test_acc:.4f} |")
        print(f"Epoch {epoch + 1} took {epoch_end - epoch_start:.2f} seconds to train and validate")
        print("---------------------------------------------------------")
        print()
    train_end = timer()
    
    print("---------------------------------------------------------")
    print(f"Training {model.__class__.__name__} model took {train_end-train_start:.2f} seconds to train for {epochs} epochs")
    print("---------------------------------------------------------")
    
    model_results = pd.DataFrame()
    model_results['Epoch'] = list(range(epochs))
    model_results['Train Loss'] = train_loss_list
    model_results['Train PSNR'] = train_acc_list
    model_results['Validation Loss'] = test_loss_list
    model_results['Validation PSNR'] = test_acc_list
    model_results['Epoch time'] = time_taken
    
    return model_results, train_end-train_start

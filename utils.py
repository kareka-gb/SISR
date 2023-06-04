import pandas as pd
import matplotlib.pyplot as plt

def plot_evaluation_curves(results: pd.DataFrame):
    """
    Plots training curves using a results dataframe

    Args:
    `results`   - A pandas dataframe containing columns with titles `Epoch`, `Train Loss`, `Validation Loss`, `Train Accuracy`, `Validation Accuracy`
    """
    plt.figure(figsize=(10, 7))
    
    epochs = results['Epoch']
    loss = results['Train Loss']
    test_loss = results['Validation Loss']
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    acc = results['Train accuracy']
    test_acc = results['Validation accuracy']
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label='Train accuracy')
    plt.plot(epochs, test_acc, label='Validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

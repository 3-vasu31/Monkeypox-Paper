import numpy as np
import matplotlib.pyplot as plt
from callbacks import LR_ASK

def train_model(model, train_gen, valid_gen, epochs, ask_epoch):
    ask_callback = LR_ASK(model, epochs, ask_epoch)
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=valid_gen,
        callbacks=[ask_callback],
        verbose=1
    )
    return history

def plot_training_history(tr_data, start_epoch=0):
    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']

    epochs_range = range(start_epoch + 1, start_epoch + len(tacc) + 1)

    # Finding best epochs
    index_loss = np.argmin(vloss)  # epoch with lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)  # epoch with highest validation accuracy
    acc_highest = vacc[index_acc]

    # Plotting
    plt.style.use('fivethirtyeight')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Loss Plot
    axes[0].plot(epochs_range, tloss, 'r', label='Training Loss')
    axes[0].plot(epochs_range, vloss, 'g', label='Validation Loss')
    axes[0].scatter(epochs_range[index_loss], val_lowest, s=150, c='blue', label=f'Best epoch: {epochs_range[index_loss]}')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Accuracy Plot
    axes[1].plot(epochs_range, tacc, 'r', label='Training Accuracy')
    axes[1].plot(epochs_range, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(epochs_range[index_acc], acc_highest, s=150, c='blue', label=f'Best epoch: {epochs_range[index_acc]}')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
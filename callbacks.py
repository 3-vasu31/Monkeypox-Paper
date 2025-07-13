import time
import numpy as np
from tensorflow import keras

class LR_ASK(keras.callbacks.Callback):
    def __init__(self, model, epochs, ask_epoch):
        super(LR_ASK, self).__init__()
        self.model = model
        self.epochs = epochs
        self.ask_epoch = max(1, ask_epoch)  # Ensure ask_epoch is at least 1
        self.ask = self.ask_epoch < epochs
        self.lowest_vloss = np.inf
        self.best_weights = self.model.get_weights()
        self.best_epoch = 1
        self.start_time = None

    def on_train_begin(self, logs=None):
        if self.ask:
            print(f'Training will proceed until epoch {self.ask_epoch}, then you will be asked for further instructions.')
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        print(f'Loading model with best weights from epoch {self.best_epoch}')
        self.model.set_weights(self.best_weights)
        duration = time.time() - self.start_time
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'Training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s')

    def on_epoch_end(self, epoch, logs=None):
        v_loss = logs.get('val_loss', np.inf)
        
        if v_loss < self.lowest_vloss:
            self.lowest_vloss = v_loss
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            print(f'Validation loss improved to {v_loss:.4f}. Saving best weights from epoch {self.best_epoch}.')
        else:
            print(f'Validation loss {v_loss:.4f} did not improve. Keeping weights from epoch {self.best_epoch}.')

        if self.ask and (epoch + 1 == self.ask_epoch):
            ans = input('\nEnter H to halt training or an integer for additional epochs: ')
            
            if ans.lower() == 'h' or ans == '0':
                print(f'Training halted by user at epoch {epoch + 1}.')
                self.model.stop_training = True
            else:
                try:
                    additional_epochs = int(ans)
                    self.ask_epoch += additional_epochs
                    print(f'Training will continue until epoch {self.ask_epoch}.')
                except ValueError:
                    print('Invalid input. Continuing with existing schedule.')
                
                lr = float(keras.backend.get_value(self.model.optimizer.lr))
                new_lr = input(f'Current LR is {lr:.5f}. Press Enter to keep or enter new LR: ')
                if new_lr:
                    try:
                        keras.backend.set_value(self.model.optimizer.lr, float(new_lr))
                        print(f'Learning rate changed to {new_lr}.')
                    except ValueError:
                        print('Invalid LR. Keeping existing value.')

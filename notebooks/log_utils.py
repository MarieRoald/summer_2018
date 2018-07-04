import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras

def plot_mean_std_loss(loss, steps, color=None, legend="", ax=None):
    """Plot mean and standard deviation of loss.
    
    Parameters
    ----------
    loss: ndarray or list
        2D array containing the loss values (num runs x num steps)
    steps: ndarray or list
        1D array containing the steps corresponding to the loss
    color: matplotlib color (optional)
        Color used to plot the mean loss (std is plotted with the same color, but lower opacity)
    legend: string (optional)
        Prefix for the legends.
    ax: Axes (optional):
        Axes to draw the plots in. If `None`, a new figure and axis is created.

    Returns
    -------

    fig: Figure
        Figure containing the plot
    """

    loss = np.asarray(loss)
    steps = np.asarray(steps)

    mean_loss = loss.mean(0)
    std_loss = loss.std(0)

    if color is None:
        color = sns.color_palette()[0]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111) 
    else:
        fig = ax.get_figure()

    std_plot = ax.fill_between(steps, 
                               mean_loss + std_loss, 
                               np.maximum(mean_loss - std_loss, 0), 
                               facecolor=color, 
                               alpha=0.2, 
                               color=color, 
                               label=legend+' std')

    ax.plot(steps, mean_loss, color=color, linestyle="--", label=legend+" mean")
    ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    return fig

class GroupedCometLogger(keras.callbacks.Callback):
    def __init__(self, experiment, log_name):
        self.experiment = experiment
        self.step_count = 1
        self.epoch_count = 1
        self.log_name = log_name
        self.train_loss = []
        self.train_steps = []
        self.val_loss = []
        self.val_steps = []
        
    def on_batch_end(self, batch, logs=None):
        logs = {} if logs is None else logs
        self.experiment.log_metric(f"{self.log_name}_loss", 
                                   logs.get('loss'), 
                                   step=self.step_count)
        self.train_steps.append(self.step_count)
        self.train_loss.append(logs.get("loss"))
        self.step_count = self.step_count + 1 

    def on_epoch_end(self, epoch, logs=None):
        logs = {} if logs is None else logs
        if "val_loss" in logs:
            self.experiment.log_metric(f"{self.log_name}_val_loss", 
                                       logs.get('val_loss'), 
                                       step=self.step_count)
            self.val_steps.append(self.step_count)
            self.val_loss.append(logs.get("val_loss"))

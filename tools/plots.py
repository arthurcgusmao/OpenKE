import matplotlib.pyplot as plt

def plot_learning_curve(learning_curve, model_info, export_path=None):
    """Plots the learning curve following `model_info`. Allows the option to export an SVG figure
    to some path.
    """
    data = learning_curve
    if model_info['log_type'] == 'epoch':
        n_epochs = data.epoch.max()
        y_min = data.epoch_loss.min()
        y_max = data.epoch_loss.iloc[model_info['n_epochs'] // 10]

        fig, ax1 = plt.subplots()

        ax1.set(title='Training Loss and Validation Accuracy',
                xlabel='Epoch',
                ylabel='Loss',
                xlim=[0, model_info['n_epochs']],
                ylim=[y_min, y_max],
               )
        ax1.plot(data.epoch, data.epoch_loss, label='Loss')
        ax1.legend(loc=2)

        ax2 = ax1.twinx()
        ax2.plot(data.epoch, data.valid_acc, label='Valid Acc', color='orange')
        ax2.set(ylabel='Accuracy')
        ax2.legend(loc=1)

        fig.set_size_inches(18, 12)
        if export_path != None: fig.savefig("{}/learning_curve.svg".format(export_path))

    elif model_info['log_type'] == 'batch':
        fig, ax1 = plt.subplots()

        ax1.set(title='Training Loss and Validation Accuracy',
                xlabel='Batch (absolute)',
                ylabel='Loss',
               )
        ax1.plot(data.batch_loss, label='Batch Loss')
        ax1.legend(loc=2)

        ax2 = ax1.twinx()
        ax2.plot(data.valid_acc, label='Valid Acc', color='orange')
        ax2.set(ylabel='Accuracy')
        ax2.legend(loc=1)

        fig.set_size_inches(18, 12)
        if export_path != None: fig.savefig("{}/learning_curve.svg".format(export_path))
    else:
        return ValueError('log_type should be defined in model_info.')
    return fig

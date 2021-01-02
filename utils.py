from matplotlib import pyplot as plt
import numpy as np

def plot_training_hist(df):
    
    fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10,5))
    for ax, nam in zip(axes, [df.columns[0], df.columns[1]]):
        
        ax.plot(range(df.shape[0]), df[nam], c='orange', label=nam)
        ax.plot(range(df.shape[0]), df['val_' + nam], c='c', label='val_' + nam)
        
        if nam != 'loss':
            v = max(df['val_' + nam])
            ax.scatter(np.argmax(df['val_' + nam]), v, c='r', marker='*',s=100, label='max val_' + nam + ' : ' + str(round(v, 4)))
        else:
            v = min(df['val_' + nam])
            ax.scatter(np.argmin(df['val_' + nam]), v, c='r', marker='*',s=100, label='min val_' + nam + ' : ' + str(round(v, 4)))
                       
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('score')
        ax.set_title(nam)
                       
    plt.show()
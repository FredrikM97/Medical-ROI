import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import ceil,floor

def display_all_pd_cols(input_df):
    with pd.option_context('display.max_columns', None):
        display(input_df.head())

def display_dict_to_yaml(input_dict:dict):
    print(yaml.dump(input_dict, allow_unicode=True, default_flow_style=False))
    
def display_advanced_plot(slices):
    import matplotlib.gridspec as gridspec
    
    w = 20
    h = 10
    columns = 4 if len(slices) >= 4 else len(slices)
    rows = floor(len(slices)/columns)
    
    fig = plt.figure(figsize=(w, h))

    
    gs1 = gridspec.GridSpec(rows,columns)
    gs1.update(wspace=0.025, hspace=0.05) 

    ax = []
    
    for i in range(columns*rows):
        #fig.set_size_inches(w, h)
        # create subplot and append to ax
        ax1 = fig.add_subplot(gs1[i])
        #ax1.set_aspect('equal')

        ax.append(ax1)
        #ax[-1].set_title("ax:"+str(i))  # set title
        plt.imshow(slices[i], cmap="gray", origin="lower")
    
    for ax in fig.get_axes():
        ax.label_outer()

    plt.show()  # finally, render the plot
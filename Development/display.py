import pandas as pd
import yaml
import matplotlib.pyplot as plt
from math import ceil,floor

def display_all_pd_cols(input_df):
    with pd.option_context('display.max_columns', None):
        display(input_df.head())

def display_dict_to_yaml(input_dict:dict):
    print(yaml.dump(input_dict, allow_unicode=True, default_flow_style=False))
    
def display_advanced_plot(slices):
    my_dpi=10
    import matplotlib.gridspec as gridspec
    w = 100
    h = 50
    fig = plt.figure(figsize=(20, 10)) #figsize=(9, 13)
    fig.subplots_adjust(hspace=0.1)
    columns = 4 if len(slices) >= 4 else len(slices)
    rows = floor(len(slices)/columns)
    gs1 = gridspec.GridSpec(rows,columns)
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

    # ax enables access to manipulate each of subplots
    ax = []
    
    for i in range(columns*rows):
        fig.set_size_inches(w, h)
        # create subplot and append to ax
        ax1 = fig.add_subplot(gs1[i])
        ax1.set_aspect('equal')
        ax.append(ax1)
        ax[-1].set_title("ax:"+str(i))  # set title
        plt.imshow(slices[i])


    plt.show()  # finally, render the plot
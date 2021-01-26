import pandas as pd
import sys
from ruamel.yaml import YAML
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import ceil,floor
import seaborn as sns

def display_all_pd_cols(input_df):
    with pd.option_context('display.max_columns', None):
        display(input_df.head())

def display_dict_to_yaml(input_dict:dict):
    yaml = YAML()
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.dump(input_dict, sys.stdout)
    
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

def move_legend(ax, new_loc="upper left",title:str=None,**kws):
    """
    If the legend disapear this function should be used. 
    Recommended from https://github.com/mwaskom/seaborn/issues/2280 where matplotlib have some weird implementation which makes the legend disappear..
    """
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    if not title: title = old_legend.get_title().get_text() #, loc=new_loc
    ax.legend(handles, labels, title=title, borderaxespad=0., bbox_to_anchor=(1.02, 1), loc=new_loc,**kws)

def set_plot_settings(ax,rotation:int=0, title:str=None, xlabel:str=None, ylabel:str=None, **kws):
    if title: ax.set_title(title)
    if xlabel and ylabel: ax.set(xlabel=xlabel, ylabel=ylabel)
    
    ax.tick_params(axis='x',which='major', rotation=rotation)
    
    
def do_countplot(
                data, 
                x:str=None, 
                y:str=None, 
                hue:str=None, 
                order:list=None, 
                ax=None,
                plot_kws:dict={},
                setting_kws:dict={},
                legend_kws:dict={},
                **kws
            ):
    """
    Example:
    dplay.do_countplot(
        df, 
        x="subject.visit.assessment.component.assessmentScore_FAQTOTAL", 
        hue="subject.researchGroup", 
        order=np.arange(1,31), 
        rotation=90, 
        xlabel="FAQTOTAL", 
        ylabel="Frequency",
        title="Functional Activities Questionnaires (FAQTOTAL)",
        ax=axes[0,0]
    )
    """
    sns.set(font_scale=1.1)

    ax1 = sns.countplot(x=x,y=y,
                hue = hue, 
                data = data,
                order=order,
                ax=ax,
                **plot_kws
            )

    set_plot_settings(ax1,**setting_kws)
    move_legend(ax1,**legend_kws)
    plt.tight_layout()
    return ax1

def do_histplot(
            df, 
            x=None,
            y=None,
            hue:str=None, 
            title:str=None,
            multiple='stack',
            bins:int='auto',
            discrete=True,
            ax=None,
            plot_kws:dict={},
            setting_kws:dict={},
            legend_kws:dict={},
            **kws
        ):
    sns.set(font_scale=1.1)
    print(bins)
    ax1 = sns.histplot(
        df,
        x=x,
        y=y, 
        hue=hue,
        bins=bins,
        discrete=discrete,
        multiple=multiple,
        ax=ax,
        **plot_kws,
        
    )
    set_plot_settings(ax1,**setting_kws)
    move_legend(ax1,**legend_kws)
    plt.tight_layout()
    return ax1
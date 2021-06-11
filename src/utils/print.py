"""
Common print functions used in different modules.
"""



import pandas as pd
import sys
from ruamel.yaml import YAML
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import ceil,floor
import seaborn as sns

def display_all_pd_cols(input_df):
    "Print all pandas columns"
    with pd.option_context('display.max_columns', None):
        display(input_df.head())

def display_dict_to_yaml(input_dict:dict):
    "Convert dict to yaml"
    yaml = YAML()
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.dump(input_dict, sys.stdout)
    
def display_advanced_plot(slices):
    "Advanced plotting function"
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

def plot_meta_settings(rows=1, cols=2, figsize=(16,16)):
    "Plot settings of meta data"
    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    fig.set_tight_layout(True)
    return fig, axes

def move_legend(ax, new_loc="upper left",title:str=None,**kws):
    """
    If the legend disapear this function should be used. 
    Recommended from https://github.com/mwaskom/seaborn/issues/2280 where matplotlib have some weird implementation which makes the legend disappear..
    """
    if (old_legend := ax.legend_) is not None:
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]
        if not title: title = old_legend.get_title().get_text() #, loc=new_loc
            
        ax.legend(handles, labels, title=title, borderaxespad=0., bbox_to_anchor=(1.02, 1), loc=new_loc,**kws)

def set_plot_settings(ax,rotation:int=0, title:str=None, xlabel:str=None, ylabel:str=None, **kws):
    sns.set_theme(style="ticks", color_codes=True)
    if title: ax.set_title(title)
    if xlabel: ax.set(xlabel=xlabel)
    if ylabel: ax.set(ylabel=ylabel)
    
    ax.tick_params(axis='x',which='major', rotation=rotation)
    
    
def do_countplot(
                df, 
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

    ax1 = sns.countplot(
                x=x,
                y=y,
                hue = hue, 
                data = df,
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

def do_catplot(
            df, 
            x=None,
            y=None,
            hue:str=None, 
            kind='bar',
            multiple='stack',
            bins:int='auto',
            discrete=True,
            plot_kws:dict={},
            setting_kws:dict={},
            legend_kws:dict={},
            **kws
        ):
    sns.set(font_scale=1.1)
    ax1 = sns.catplot(
        df,
        x=x,
        y=y, 
        kind=kind,
        hue=hue,
        bins=bins,
        discrete=discrete,
        multiple=multiple,
        **plot_kws,
        
    )
    set_plot_settings(ax1,**setting_kws)
    move_legend(ax1,**legend_kws)
    plt.tight_layout()
    return ax1

    
def do_boxplot(
                df, 
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
 
    sns.set(font_scale=1.1)

    ax1 = sns.boxplot(
                x=x,
                y=y,
                hue = hue, 
                data = df,
                order=order,
                ax=ax,
                **plot_kws
            )

    set_plot_settings(ax1,**setting_kws)
    move_legend(ax1,**legend_kws)
    plt.tight_layout()
    return ax1

def do_lineplot(
            df, 
            x=None,
            y=None,
            hue:str=None, 
            title:str=None,
            ax=None,
            plot_kws:dict={},
            setting_kws:dict={},
            legend_kws:dict={},
            **kws
        ):
    "Evaluate if this works. Not tested"
    sns.set(font_scale=1.1)
    ax1 = sns.lineplot(
        data=df,
        x=x,
        y=y, 
        hue=hue,
        ax=ax,
        **plot_kws,
        
    )
    set_plot_settings(ax1,**setting_kws)
    move_legend(ax1,**legend_kws)
    plt.tight_layout()
    return ax1

def write_to_file(filepath:str, message:str, flag='a'):
    try:
        with open(filepath, flag) as f:
            f.write(message)
    except Exception as e:
        raise Exception("Could not write message to file!") from e
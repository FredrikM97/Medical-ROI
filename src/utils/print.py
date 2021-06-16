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
    """

    Parameters
    ----------
    input_df :
        

    Returns
    -------

    
    """
    "Print all pandas columns"
    with pd.option_context('display.max_columns', None):
        display(input_df.head())

def display_dict_to_yaml(input_dict:dict):
    """

    Parameters
    ----------
    input_dict : dict
        

    Returns
    -------

    
    """
    "Convert dict to yaml"
    yaml = YAML()
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.dump(input_dict, sys.stdout)
    
def display_advanced_plot(slices):
    """

    Parameters
    ----------
    slices :
        

    Returns
    -------

    
    """
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

def plot_meta_settings(rows:int=1, cols:int=2, figsize=(16,16)):
    """

    Parameters
    ----------
    rows : int
        (Default value = 1)
    cols : int
        (Default value = 2)
    figsize :
        (Default value = (16,16))

    Returns
    -------

    
    """
    "Plot settings of meta data"
    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    fig.set_tight_layout(True)
    return fig, axes

def move_legend(ax, new_loc:str="upper left",title:str=None,**kws):
    """If the legend disapear this function should be used.
    Recommended from https://github.com/mwaskom/seaborn/issues/2280 where matplotlib have some weird implementation which makes the legend disappear..

    Parameters
    ----------
    ax :
        
    new_loc : str
        (Default value = "upper left")
    title : str
        (Default value = None)
    **kws :
        

    Returns
    -------

    
    """
    if (old_legend := ax.legend_) is not None:
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]
        if not title: title = old_legend.get_title().get_text() #, loc=new_loc
            
        ax.legend(handles, labels, title=title, borderaxespad=0., bbox_to_anchor=(1.02, 1), loc=new_loc,**kws)

def set_plot_settings(ax,rotation:int=0, title:str=None, xlabel:str=None, ylabel:str=None, **kws):
    """

    Parameters
    ----------
    ax :
        
    rotation : int
        (Default value = 0)
    title : str
        (Default value = None)
    xlabel : str
        (Default value = None)
    ylabel : str
        (Default value = None)
    **kws :
        

    Returns
    -------

    
    """
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
    """Example:
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

    Parameters
    ----------
    df :
        
    x : str
        (Default value = None)
    y : str
        (Default value = None)
    hue : str
        (Default value = None)
    order : list
        (Default value = None)
    ax :
        (Default value = None)
    plot_kws : dict
        (Default value = {})
    setting_kws : dict
        (Default value = {})
    legend_kws : dict
        (Default value = {})
    **kws :
        

    Returns
    -------

    
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
    """

    Parameters
    ----------
    df :
        
    x :
        (Default value = None)
    y :
        (Default value = None)
    hue : str
        (Default value = None)
    multiple :
        (Default value = 'stack')
    bins : int
        (Default value = 'auto')
    discrete :
        (Default value = True)
    ax :
        (Default value = None)
    plot_kws : dict
        (Default value = {})
    setting_kws : dict
        (Default value = {})
    legend_kws : dict
        (Default value = {})
    **kws :
        

    Returns
    -------

    
    """
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
    """

    Parameters
    ----------
    df :
        
    x :
        (Default value = None)
    y :
        (Default value = None)
    hue : str
        (Default value = None)
    kind :
        (Default value = 'bar')
    multiple :
        (Default value = 'stack')
    bins : int
        (Default value = 'auto')
    discrete :
        (Default value = True)
    plot_kws : dict
        (Default value = {})
    setting_kws : dict
        (Default value = {})
    legend_kws : dict
        (Default value = {})
    **kws :
        

    Returns
    -------

    
    """
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
    """

    Parameters
    ----------
    df :
        
    x : str
        (Default value = None)
    y : str
        (Default value = None)
    hue : str
        (Default value = None)
    order : list
        (Default value = None)
    ax :
        (Default value = None)
    plot_kws : dict
        (Default value = {})
    setting_kws : dict
        (Default value = {})
    legend_kws : dict
        (Default value = {})
    **kws :
        

    Returns
    -------

    
    """
 
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
    """

    Parameters
    ----------
    df :
        
    x :
        (Default value = None)
    y :
        (Default value = None)
    hue : str
        (Default value = None)
    title : str
        (Default value = None)
    ax :
        (Default value = None)
    plot_kws : dict
        (Default value = {})
    setting_kws : dict
        (Default value = {})
    legend_kws : dict
        (Default value = {})
    **kws :
        

    Returns
    -------

    
    """
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
    """

    Parameters
    ----------
    filepath : str
        
    message : str
        
    flag :
        (Default value = 'a')

    Returns
    -------

    
    """
    try:
        with open(filepath, flag) as f:
            f.write(message)
    except Exception as e:
        raise Exception("Could not write message to file!") from e
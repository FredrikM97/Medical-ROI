import pandas as pd
import yaml

def display_all_pd_cols(input_df):
    with pd.option_context('display.max_columns', None):
        display(input_df.head())

def display_dict_to_yaml(input_dict:dict):
    print(yaml.dump(input_dict, allow_unicode=True, default_flow_style=False))
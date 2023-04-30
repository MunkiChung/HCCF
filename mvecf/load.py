import os
import re

from mvecf.path import root_path
from mvecf.utils import load_pickle


def get_data(data_type, year):
    data_dir_name = os.path.join(root_path, "data")

    holdings_data = load_pickle(
        os.path.join(data_dir_name, '{}/{}/holdings_data.pkl'.format(data_type, year)))
    factor_params = load_pickle(
        os.path.join(data_dir_name, '{}/{}/factor_model_params.pkl'.format(data_type, year)))
    return holdings_data, factor_params

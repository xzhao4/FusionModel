# -*- coding: utf-8 -*-
"""Merge_Inference_blend.ipynb


"""

import numpy as np
import pandas as pd

deep_csv = 'result_deep.csv'
table_csv = 'result_table.csv'

deep_pred_df = pd.read_csv(deep_csv)
table_pred_df = pd.read_csv(table_csv)

deep_pred = list(deep_pred_df.pred)
table_pred = list(table_pred_df.pred)

a = 0.3
b = 0.7

merge_pred = a * deep_pred + b * table_pred

table_pred['pred'] = merge_pred
merge_df = table_pred['pred'].copy()

merge_df.to_csv('result_merge.csv')

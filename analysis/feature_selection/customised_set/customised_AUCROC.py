import pandas as pd
import os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)

df_results = pd.read_pickle(str(p.parents[3])+'/test/df_results.plk')

#===============================================================================
#                     Customised meta-feature set
#===============================================================================

selected_X = df_results.iloc[:,[0, 7, 9, 11, 18,19]]

with open('customised_X_AUCROC_202.pickle', 'wb') as handle:
    pickle.dump(selected_X, handle, protocol=pickle.HIGHEST_PROTOCOL)
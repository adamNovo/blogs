import pandas as pd
import numpy as np
from sklearn import preprocessing

def encode_data(df, cat_cols):
    df_enc = df.copy(deep=True)
    for i in cat_cols:
        print("Converting column {}".format(i))
        lb = preprocessing.LabelBinarizer()
        vals = lb.fit_transform(df_enc.loc[:, [i]])
        cols = ["{}_{}".format(i, x) for x in lb.classes_]
        if len(cols) == 2:
            cols = [cols[1]]
        df_enc = pd.concat([df_enc, pd.DataFrame(vals, columns=cols)], axis=1, sort=False)
        df_enc = df_enc.drop(columns=i)
    return df_enc
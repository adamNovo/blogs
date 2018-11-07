import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def dataframe_to_image(data: pd.DataFrame, title):
    print("Saving", title)
    # print(data)
    size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([2, 1])
    fig, ax = plt.subplots(figsize=size)
    ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=[0, 0, 1, 1], 
        colLabels=data.columns, rowLabels=data.index,
        fontsize=22.0)
    mpl_table.scale(1, 1)
    fig.suptitle(title)
    fig.savefig("images/{}.png".format(title))
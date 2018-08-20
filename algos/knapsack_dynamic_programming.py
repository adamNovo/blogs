import numpy as np
import pandas as pd

items = pd.DataFrame(columns=["weight", "value"])
items.loc[len(items)] = [6, 30]
items.loc[len(items)] = [3, 14]
items.loc[len(items)] = [4, 16]
items.loc[len(items)] = [2, 9]

def knapsack_with_repetition(items, weight):
    bag = pd.DataFrame(np.zeros(weight+1), columns=["value"])
    for i in range(1, weight+1):
        for j, row in items.iterrows():
            item_w = row["weight"]
            item_val = row["value"]
            if i - item_w >= 0 and bag.loc[i]["value"] < bag.loc[i-item_w]["value"] + item_val:
                bag.loc[i]["value"] = bag.loc[i-item_w]["value"] + item_val
    return bag

bag = knapsack_with_repetition(items, 10)
print(bag)
assert 48 == bag.loc[len(bag)-1]["value"]
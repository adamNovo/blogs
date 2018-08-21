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

def knapsack_without_repetition(items, weight):
    bag = pd.DataFrame(np.zeros((weight+1, len(items))))
    for i in range(len(items)):
        for j in range(1, weight+1):
            item_w = items.loc[i, "weight"]
            item_val = items.loc[i, "value"]
            if j >= item_w:
                val_new_w = bag.loc[j-item_w, i] + item_val
                bag.loc[j, i] = max(bag.loc[j-1, i], val_new_w)
            else:
                bag.loc[j, i] = bag.loc[j-1, i]
    return bag
bag = knapsack_without_repetition(items, 10)
print(bag)
assert 45 == bag.loc[10, len(items)-1]
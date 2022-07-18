'''
Module supporting the generation of universally unique identifiers
'''

# Standard library imports
import random
import uuid

# External library imports
import numpy as np
import pandas as pd

def generate_uuid_column(col_name, num_rows, miss_prob, frequency_distribution, seed):
    '''
    Have to use seed rather than a spec_wide rng generator because for consistent
    uuids we need to use the same 128-bit integer.
    '''
    
    # parse the frequency distribution into a Pandas dataframe:
    freq_df = pd.DataFrame(
        data=[
            map(str.strip, x.split("|")) for x in frequency_distribution[1:]
        ],
        columns=[x.strip() for x in frequency_distribution[0].split("|")],
    )

    # ensure the probabilities sum up to 1
    prob_vector = freq_df["probability_vector"].astype(float).values
    prob_vector /= prob_vector.sum()
    freq_df.loc[:, "probability_vector"] = prob_vector

    # generate uuids
    rng = random.Random(seed)
    uuids = []

    for row in freq_df.itertuples():

        _uuids = []
        # always round up the number of generated rows before casting to int
        _num_rows = int(np.ceil(
            num_rows * float(row.probability_vector) / int(row.frequency)))

        for _ in range(_num_rows):
            _uuids.append(uuid.UUID(int=rng.getrandbits(128), version=4).hex)

        _uuids = _uuids * int(row.frequency)
        uuids.extend(_uuids)

    # make sure the number of uuid rows matches the num_rows:
    # more generated uuids than num_rows - remove extra uuids
    # from the end, meaning the higher frequencies will be more
    # affected.
    if len(uuids) - num_rows > 0:
        uuids = uuids[:num_rows]
    
    # finally, make uuid null based on the miss_probability
    rands = np.array([rng.random() for _ in range(num_rows)])
    data = list(np.where(rands < miss_prob, None, uuids))  

    # create a series and shuffle
    uuid_series = (
        pd.Series(data, name=col_name)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    return uuid_series

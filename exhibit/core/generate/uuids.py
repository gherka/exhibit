'''
Module supporting the generation of universally unique identifiers
'''

# Standard library imports
import random
import uuid

# External library imports
import numpy as np
import pandas as pd

def generate_uuid_column(
    col_name, num_rows, miss_prob, frequency_distribution, seed, uuid_type="uuid"):
    '''
    Have to use seed rather than a spec_wide rng generator because for consistent
    uuids we need to use the same 128-bit integer.
    '''

    # for internal use in tests or scripting
    if isinstance(frequency_distribution, pd.DataFrame):
        freq_df = frequency_distribution
    
    else:
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
    range_max = 0

    # we need to know the total number of pseudo-chis ahead of generation time
    # to ensure consistency with random seed and to avoid duplicates so that the first
    # N of pseudo-chis for a given seed will always be the same.

    if uuid_type == "pseudo_chi":
        pseudo_chi_total = 0
        for row in freq_df.itertuples():
            _num_rows = int(np.ceil(
            num_rows * float(row.probability_vector) / int(row.frequency)))
            pseudo_chi_total = pseudo_chi_total + _num_rows
            pseudo_chis = _generate_pseudo_chis(n=pseudo_chi_total, seed=seed)

    for row in freq_df.itertuples():

        # initialise intermediate uuid list
        _uuids = []

        # always round up the number of generated rows before casting to int
        _num_rows = int(np.ceil(
            num_rows * float(row.probability_vector) / int(row.frequency)))

        # special case for range-type UUID generation
        if uuid_type == "range":
            _uuids = list(range(range_max, range_max + _num_rows)) * int(row.frequency)
            uuids.extend(_uuids)
            range_max = range_max + _num_rows
            continue

        if uuid_type == "pseudo_chi":
            _uuids = pseudo_chis[range_max: range_max + _num_rows] * int(row.frequency)
            uuids.extend(_uuids)
            range_max = range_max + _num_rows
            continue

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

    # if the UUID type is range, shuffle the values so that we don't end up with
    # UUID == 0 always being freq = 1, etc.
    if uuid_type == "range":
        repl_uuids = list(range(range_max))
        rng.shuffle(repl_uuids)
        uuids = [repl_uuids[x] for x in uuids]
    
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

def _generate_pseudo_chis(n, seed=0):
    '''
    Generate pseudo CHI numbers that consist of 10 digits, including
    a possible zero as the first digit.

    The logic of CHIs is not preserved to avoid potential collisions with the real CHIs.
    In addition, the month part of the CHI is fixed at the impossible 13.

    Parameters
    ----------
    n : int
        the number of dummy CHIs to generate.

    Returns
    -------
    A sorted list with unique dummy CHI numbers
    '''

    random.seed = seed
    result = set()

    while len(result) < n:
        pseudo_chi = (
            str(random.randint(0,31)) + # day will be zero padded if total length < 10
            '13' +                      # ensure no accidental collissions
            str(random.randint(20, 99)) +
            str(random.randint(0,9999)).zfill(4) # no specific logic for 9th digit
        ).zfill(10)

        result.add(pseudo_chi)
    
    return sorted(list(result))
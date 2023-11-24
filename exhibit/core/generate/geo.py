'''
Methods to generate / deal with geo-spatial data
'''

# External library imports
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import linemerge, unary_union, polygonize
from shapely.affinity import scale, rotate

# Exhibit imports
from ..sql import query_exhibit_database

def generate_geospatial_column(col_name, h3_ids, h3_probs, number_of_rows, rng):
    '''
    Sample latitude and longitude from a list of H3 hexes given the probability of
    each hex.

    Parameters (from spec_dict)
    ----------
    col_name       : str
        name of the table in exhibit DB which contains h3 hex ids and their weights
    h3_ids         : list
        list of valid h3 ids from which lat/long pairs should be sampled
    h3_probs       : list
        list of probabilities for each h3 id
    number_of_rows : int
        how many rows to generate
    rng:           : numpy random generator
        random generator passed on from spec_dict
    
    Returns
    -------
    pd.DataFrame with two columns: latitude and longitude, prefixed with col_name

    Default CRS is WGS84 which is understood by the h3 library. If different
    from WGS84, re-project the data before returning.
    '''

    coords = []

    for _ in range(number_of_rows):

        this_hex = rng.choice(h3_ids, p=h3_probs)
        hex_poly = Polygon(h3.h3_to_geo_boundary(str(this_hex), geo_json=True))
        po = get_random_point_in_polygon(hex_poly, rng)
        # latitude, longitude
        coords.append((po.y, po.x))

    # for now, only support splitting the coordinates into two prefixed columns
    output = pd.DataFrame(
        coords, columns=[f"{col_name}_latitude", f"{col_name}_longitude"])

    return output

def get_random_point_in_polygon(poly, rng):
    '''
    Rejection sampling to make sure the generated point is within the hexagon.
    '''

    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(p):
            return p

def geo_make_regions(
    df, filter_idx, target_str, partition_cols, spec_dict):
    '''
    Create contiguous regions (one per partition level) and sample from
    H3 hexes that fall into each region. Only works if partition_cols is provided.

    Regions are created using pseudo-recursive bisection whereby the initial
    bounding box is randomply split into two regions, then each is split into
    a further two and so on - splitting the larger regions before moving down
    a level.

    Parameters
    ----------
    df             : pd.DataFrame
        Unmodified dataframe
    filter_idx     : pd.Index
        Index of rows to be modified by the function
    target_str     : str
        For geo functions acting on coordinate columns this means both lat and long
        columns of the named column.
    partition_cols : list
        Columns to group by to create one region per partition to sample from
    spec_dict      : dict
        User specification to get random number generator and h3_table

    Returns
    -------
    pd.DataFrame
    '''

    if not partition_cols: #pragma: no cover
        raise RuntimeError("make_geo_regions action requires at least one partition")

    geo_target_cols = [x.strip() for x in target_str.split(",")]
    partition_cols = [x.strip() for x in partition_cols.split(",") if x]
    rng = spec_dict["_rng"]

    geo_target_cols_table_names = []
    for target_col in geo_target_cols:

        h3_table_name = spec_dict["columns"][target_col]["h3_table"]
        geo_target_cols_table_names.append(h3_table_name)

    if len(set(geo_target_cols_table_names)) != 1: #pragma: no cover
        raise RuntimeError(
            "Columns used for make_geo_regions action rely on different h3 tables")

    # add placeholders for output columns
    target_cols = []
    for target_col in geo_target_cols:
        temp_target_cols = [f"{target_col}_latitude", f"{target_col}_longitude"]
        target_cols.extend(temp_target_cols)
    # in case values for partition columns are null
    df[target_cols] = np.nan
    
    # get the geo df with h3s (any one of the given geo_target_cols is fine)
    h3_ids = (
        query_exhibit_database(table_name=h3_table_name, column="h3", order="h3")
        .values.ravel())

    output_df = df.set_index(partition_cols).sort_index()

    # create the groups object; when grouping on categorical, ensure observed=True
    grouped = df.groupby(partition_cols, observed=True)[target_cols]

    # get the grouped multiindex
    grouped_idx = grouped.size().index

    # create the geo_df with regions
    geo_df = pd.DataFrame(data={"h3":h3_ids})

    # add H3 centroid coordinates
    geo_df["lat"], geo_df["long"] = zip(
        *geo_df["h3"].transform(h3.h3_to_geo))

    # create initial region indices based on the N of values in level=0
    n_regions = grouped_idx.get_level_values(level=0).nunique()
    prev_result = None
    final_result = []

    # if only one level of regions is required:
    if grouped_idx.nlevels == 1:
        final_result = _create_contiguous_regions(
                geo_df, target_n_regions=n_regions, rng=rng)

    # iterate over 2 index levels at a time
    else:
        i = 0
        while i < (grouped_idx.nlevels - 1):
            
            if prev_result is None:
                prev_result = _create_contiguous_regions(
                    geo_df, target_n_regions=n_regions, rng=rng)
            # hm..unsure?
            else:
                prev_result = final_result #pragma: no cover
            
            region_counts = (
                grouped_idx.to_frame().groupby(level=i, observed=True).size().to_list())

            for j, subregion_count in enumerate(region_counts):
                subdf = geo_df.loc[prev_result[j]]
                final_result.extend(
                    _create_contiguous_regions(subdf, subregion_count, rng=rng)
                )

            i = i + 1

    # assign the regions to the data df
    for i, grouped in enumerate(grouped):

        group = grouped[1]
        index = grouped[0]
        region_ids = geo_df.loc[final_result[i], "h3"].values
        num_rows = len(group)

        # COLUMN SPECIFIC CODE (ALL COLUMNS SHARE THE SAME REGIONS)
        for target_col in geo_target_cols:

            dist = spec_dict["columns"][target_col]["distribution"]
            target_cols = [f"{target_col}_latitude", f"{target_col}_longitude"]

            # pick the hex weights from the DB table, if any
            if dist == "uniform":
                h3_probs = None
            else:
                prob_col = query_exhibit_database(
                    table_name=h3_table_name, column=dist, order="h3", distinct=False)
                h3_probs = (prob_col / prob_col.sum()).values.ravel()

            # add the hex probabiltities to the geo_df (uniform or column specific)
            geo_df = geo_df.sort_values(by="h3").assign(h3_probs=h3_probs)

            # get the probabilities of region_ids:
            if dist == "uniform":
                region_probs = None
            else:
                region_probs_col = geo_df.loc[geo_df["h3"].isin(region_ids), "h3_probs"]
                region_probs = (region_probs_col / region_probs_col.sum()).values.ravel()

            region_df = generate_geospatial_column(
                target_col, region_ids, region_probs, num_rows, rng)

            output_df.loc[index, target_cols] = region_df.values

    return output_df.reset_index()

def _cut_polygon_by_line(polygon, line):
    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)

def _create_contiguous_regions(
    original_df, target_n_regions, rng, current_n_regions=0,
    prev_regions_idx=None, final_regions_idx=None):
    '''
    Returns a list of indices of the original DF that correspond to
    each region
    '''

    if target_n_regions == 1: #pragma: no cover
        return [original_df.index]
    
    if prev_regions_idx is None:
        prev_regions_idx = [original_df.index]
    
    if final_regions_idx is None:
        final_regions_idx = [original_df.index]
    
    if current_n_regions == target_n_regions:
        return final_regions_idx
    
    df = original_df.loc[prev_regions_idx.pop()]

    p1_lat_min, p1_lat_max = np.quantile(df["lat"], [0.2, 0.4])
    p1_long_min, p1_long_max = np.quantile(df["long"], [0.2, 0.4])
    p2_long_min, p2_long_max = np.quantile(df["long"], [0.6, 0.8])

    p1 = Point(
        rng.uniform(low=p1_lat_min, high=p1_lat_max),
        rng.uniform(low=p1_long_min, high=p1_long_max),
    )

    p2 = Point(
        p1.x,
        rng.uniform(low=p2_long_min, high=p2_long_max),
    )

    polygon = Polygon([
        (df["lat"].min(), df["long"].min()),
        (df["lat"].max(), df["long"].min()),
        (df["lat"].max(), df["long"].max()),
        (df["lat"].min(), df["long"].max()),
    ])

    x, y = polygon.exterior.coords.xy
    width = Point(x[0], y[0]).distance(Point(x[1], y[1]))
    height = Point(x[0], y[0]).distance(Point(x[3], y[3]))
    aspect_ratio = height/width

    line = LineString([p1, p2])
    scaled_line = scale(line, xfact=15.0, yfact=15.0, zfact=1.0, origin="center")
    
    # to avoid very thin regions, change the rotation angle of the cutting line based
    # on the aspect rato; 2 is a magic number; another option is to use a tighter "crop"
    # of the initial bounding box for the polygon.
    if aspect_ratio > 2:
        rotation_angle_low = 70
        rotation_angle_high = 110
    
    else:
        rotation_angle_low = 160
        rotation_angle_high = 200

    rotated_line = rotate(
        scaled_line,
        angle=int(rng.uniform(low=rotation_angle_low, high=rotation_angle_high))
    )

    result = _cut_polygon_by_line(polygon, rotated_line)

    # must include a buffer to avoid awkward placements! Make sure the buffer
    # is not larger than the smallest region!
    idx_child_1 = (
        df[df.apply(
            lambda x: result[0].buffer(0.01).contains(Point(x["lat"], x["long"])),
            axis=1)].index
    )
    idx_child_2 = pd.Index(np.setdiff1d(df.index, idx_child_1))

    # sometimes random splitting of the bounding box can lead to a situation when
    # all valid centroids are on one side. In that case, retry the split again!
    # in theory, you should always end up with a valid split unless the number of 
    # regions is so large that it splits the hex into two. In such a case, increase
    # hex resolution.
    retries = 0
    while (len(idx_child_1) == 0 or len(idx_child_2) == 0): #pragma: no cover

        if retries == 5:
            print("Regions created: ", len(final_regions_idx))
            raise RuntimeError("Can't create a subregion.")

        rotated_line = rotate(scaled_line, angle=int(rng.uniform(low=0, high=180)))
        result = _cut_polygon_by_line(polygon, rotated_line)

        idx_child_1 = (
            df[df.apply(
                lambda x: result[0].buffer(0.1).contains(Point(x["lat"], x["long"])),
                axis=1)].index
            )
        idx_child_2 = pd.Index(np.setdiff1d(df.index, idx_child_1))

        retries = retries + 1

    final_regions_idx.extend([idx_child_1, idx_child_2])
    
    parent_idx = None
    for i, _ in enumerate(final_regions_idx):
        if final_regions_idx[i].equals(idx_child_1.union(idx_child_2)):
            parent_idx = i
            
    if parent_idx is not None:       
        del final_regions_idx[parent_idx]
    
    prev_regions_idx.insert(0, idx_child_1)
    prev_regions_idx.insert(0, idx_child_2)
    
    if current_n_regions == 0:
        current_n_regions = 2
    else:
        current_n_regions = current_n_regions + 1        
        
    return _create_contiguous_regions(
        original_df, target_n_regions, rng, current_n_regions,
        prev_regions_idx, final_regions_idx)
    
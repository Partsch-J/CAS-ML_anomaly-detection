import math
import sys
import numpy as np
import pandas as pd


def similarity_matrix(data=pd.DataFrame, sim_func=str) -> pd.DataFrame:

    """Computes the similarity for each value to each value in given DataFrame.
    sim_func = 'jaccard' computes 'jaccard-index'
    sim_func = 'euclidian' computes 'euclidian distance'
    Returns: 'similarity-matrix' as DataFrame"""

    # Making sure individually named indices are also accepted:
    original_index = data.index.tolist()
    data_arr = data.to_records(index=False).astype(np.dtype)

    df_x = pd.DataFrame(pd.DataFrame(data=data_arr)).transpose()
    df_x = pd.concat([df_x] * len(data), ignore_index=True)
    df_y = df_x.transpose()
    df_xy = df_x + df_y

    # Computes 'similatrity-matrix' based on 'jaccard distance' as similarity function:
    if sim_func == "jaccard":
        df_union = df_xy.applymap(func=lambda v: len(set(v)))
        df_intersect = df_xy.applymap(func=lambda v: len(v)) - df_union
        sim_matrix = df_intersect / df_union

    # Computes 'similatrity-matrix' based on 'euclidian distance' as similarity function:
    elif sim_func == "euclidian":
        sim_matrix = df_xy.applymap(
            func=lambda v: math.sqrt(len(set(v)) - (len(v) - len(set(v))))
        )

    else:
        print(
            "Similarity-function accepts 'euclidian' or 'jaccard' as input -> Revise spelling!?"
        )
        sys.exit()

    sim_matrix.index = original_index
    sim_matrix.columns = original_index

    return sim_matrix


def links(data: pd.DataFrame, theta: float) -> pd.DataFrame:

    """Computes 'adjacency matrix' and 'link matrix' for given dataset. Returns 'link-matrix' as DataFrame"""

    # Computes 'adjacency matrix' = Converting all values to 0 if < theta, to 1 if > theta:
    adj_matrix = pd.DataFrame(data=data >= theta).astype(int)
    np.fill_diagonal(adj_matrix.values, 0)

    # Computes number of common links between all points:
    link_matrix = adj_matrix.dot(adj_matrix)
    np.fill_diagonal(link_matrix.values, 0)

    return link_matrix


def rock(
    data: pd.DataFrame, k: int, theta: float, outlier_size: int
) -> pd.Series or tuple:

    """Rock-Algorithm for clustering:
    give: similarity-matrix as DataFrame
    define: number of clusters 'k' -> int
    define: threshold 'theta' -> float.
    define: max clustersize counted as outlier in outlier_size: int / returns 'None' if set to 0
    Stops clustering if number of clusters k is reached or number of links in link-matrix = 0
    Returns: Series or tuple
    Series, indicating clustered indices in index [dtype = list] and size of clusters as value [dtype = int]
    List, indicating detected outliers corresponding to set outlier-size"""

    func = 1.0 + 2.0 * ((1.0 - theta) / (1.0 + theta))
    # Create clustertable to keep track of clustersizes
    size_ci = pd.DataFrame(
        data=np.ones(shape=(len(data), len(data))), index=data.index, columns=data.index
    )
    size_cj = size_ci.transpose()

    # Keeping track of clustered indices for list of outliers and Series 'clusters'
    init_clusters = []
    for i in data.index.tolist():
        init_clusters.append([i])

    clusters_dict = dict(zip(size_ci.index, init_clusters))
    outlier_pred = pd.Series(data=False, index=data.index, name="outlier")

    # Computing link-matrix
    link_matrix = links(data=data, theta=theta)

    while len(link_matrix) > k and link_matrix.values.sum() != 0:
        # computes goodness-matrix:
        goodness_matrix = link_matrix / (
            (size_ci + size_cj) ** func - size_ci ** func - size_cj ** func
        )
        # finds all highest goodness-measures and choses corresponding indices of first occurrance:
        goodness_max = goodness_matrix[
            goodness_matrix.isin([goodness_matrix.to_numpy().max()]) == True
        ]
        c_join = goodness_max.idxmax().dropna().convert_dtypes(convert_integer=True)
        ci = c_join.index[0]
        cj = c_join.values[0]

        # summing up links of joined clusters, dopping joined index from link-matrix:
        link_matrix[ci] += link_matrix[cj]
        link_matrix.loc[ci, :] += link_matrix.loc[cj, :]
        link_matrix = link_matrix.drop(cj, axis=1)
        link_matrix = link_matrix.drop(cj, axis=0)
        link_matrix.at[ci, ci] = 0

        # summing up clustersize of joined clusters, dopping joined index from cluster-matrix:
        size_ci[ci] += size_ci[cj]
        size_ci = size_ci.drop(cj, axis=1)
        size_ci = size_ci.drop(cj, axis=0)
        size_cj = size_ci.transpose()

        # updatig cluster-dictionary
        clusters_dict[ci] = clusters_dict[ci] + clusters_dict.pop(cj)

    if link_matrix.values.sum() == 0:
        print(
            f"\nWhile theta = '{theta}': No more joins possible after k = '{len(size_ci)}' clusters got formed!\nClustering stopped\n"
        )
    else:
        print(
            f"While theta = '{theta}': Desired number of clusters k ='{k}' formed. \nClustering stopped!\n"
        )

    clusters = pd.Series(
        data=clusters_dict.values(), index=size_cj.iloc[:, 0], name="clusters"
    )
    # if outlier_size is set > 0, computes a boolean list of True/False, indicating if clustered indices are outliers/regular observations
    if outlier_size > 0:
        outliers = clusters[clusters.index <= outlier_size].values.tolist()
        if len(outliers) > 0:
            outliers = list(np.concatenate(outliers).flat)
            outlier_pred[outliers] = True
        return clusters, outlier_pred
    # if outlier_size is set to 0, only Series, indicating clusters is returned
    else:
        return clusters


def theta_min(data: pd.DataFrame, try_theta: float) -> float or bool:

    """Returns given theta for which is True: Any point in corresponding link-matrix has at least one link!
    Else returns 'False'"""

    link_matrix = links(data=data, theta=try_theta)
    max_links_c = link_matrix.max().tolist()

    if 0 in max_links_c:
        return False

    else:
        return try_theta

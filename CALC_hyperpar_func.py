import multiprocessing as mp
from itertools import repeat
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
)
import numpy as np
from pandas import DataFrame, concat
from CALC_algorithms import rock, theta_min, similarity_matrix


def work_theta_min(data: DataFrame, try_thetas: float) -> float:

    """Trys all thetas in given list 'try_thetas' in given order.
    Returns first theta for which is True: Any point in corresponding link-matrix has at least one link!
    Returns 'theta_min' as float"""

    init_theta = theta_min(data=data, try_theta=try_thetas)

    if init_theta is False:
        return False
    else:
        return init_theta


def work_clustertable(data: DataFrame, theta: float, outlier_size: int) -> DataFrame:

    """Excecutes rock-alogrithm, returns DataFrame, indicating overall number of clusters,
    number of joined clusters, corresponding theta"""

    clusters = rock(data=data, k=theta, theta=0.34, outlier_size=outlier_size)

    cl_sizes = clusters.index
    num_outlier = sum(cl_sizes == 1)
    num_joins = cl_sizes.size - num_outlier
    max_5 = sum((cl_sizes <= 5) & (cl_sizes > 1))
    max_10 = sum((cl_sizes <= 10) & (cl_sizes > 5))
    max_50 = sum((cl_sizes <= 50) & (cl_sizes > 10))
    max_100 = sum((cl_sizes <= 100) & (cl_sizes > 50))
    max_500 = sum((cl_sizes <= 500) & (cl_sizes > 100))
    max_1000 = sum((cl_sizes <= 1000) & (cl_sizes > 500))
    max_2000 = sum((cl_sizes > 1000))

    clustertable = DataFrame(
        {
            "NUM_CLUSTER": cl_sizes.size,
            "NUM_JOINS": num_joins,
            "NUM_OUTLIER": num_outlier,
            "TO_5": max_5,
            "TO_10": max_10,
            "TO_50": max_50,
            "TO_100": max_100,
            "TO_500": max_500,
            "TO_1000": max_1000,
            "GR_1000": max_2000,
        },
        index=[round(theta, 2)],
    )

    return clustertable


def work_perftable(
    data: DataFrame,
    y_true: np.ndarray,
    theta: float,
    outlier_size: int,
    init_k: int,
    k: int,
) -> DataFrame:

    """Excecutes rock-alogrithm, returns DataFrame, indicating overall number of clusters,
    number of joined clusters, corresponding theta"""

    clusters, outlier_pred = rock(
        data=data, theta=theta, k=k, outlier_size=outlier_size
    )
    y_pred = outlier_pred.values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1_bin = f1_score(y_true=y_true, y_pred=y_pred, average="binary")
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    perf_dict = {
        "NUM_CLUSTER": k,
        "K": k - init_k,
        "THETA": theta,
        "True_Negative": tn,
        "False_Positive": fp,
        "False_Negative": fn,
        "True_Positive": tp,
        "MCC": mcc,
        "F1-Binary": f1_bin,
        "F1-Macro": f1_macro,
        "ACC": acc,
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
        "Precision": tp / (tp + fp),
    }

    return perf_dict


def find_theta_min(
    data: DataFrame, start: float, stop: float, step: float, inverse: bool
) -> float:

    """Define range of 'thetas' by start, stop, step. Start and stop are inclusive. Give parameters as float!
    Inverse order with inverse=True/False. Default = ascending order!
    Computes greatest theta for which is True: Any point in corresponding link-matrix has at least one link!
    Returns 'theta_min' as float"""

    cores_max = mp.cpu_count()
    theta_list = np.arange(start, stop + step, step)

    if inverse is True:
        theta_list = theta_list[::-1].tolist()
    else:
        theta_list = theta_list.tolist()

    if cores_max <= len(theta_list):
        num_cores = cores_max
    else:
        num_cores = len(theta_list)

    with mp.Pool(processes=num_cores) as pool:
        min_thetas = pool.starmap(work_theta_min, zip(repeat(data), theta_list))

        pool.close()
        pool.join()

    min_theta = round(max(min_thetas), len(str(step)) - 2)

    print(f"\nAll points have at least one link in case of:\ntheta >= '{min_theta}'\n")

    return min_theta


def find_clustertable(
    data: DataFrame,
    start: float,
    stop: float,
    step: float,
    inverse: bool,
    outlier_size: int,
) -> float:

    """Define range of 'thetas' by start, stop, step. Start and stop are inclusive. Give parameters as float!
    Inverse order with inverse=True/False. Default = ascending order!
    Computes greatest theta for which is True: Any point in corresponding link-matrix has at least one link!
    Returns 'theta_min' as float"""

    cores_max = mp.cpu_count()
    theta_list = np.arange(start, stop + step, step)
    sim_matrix = similarity_matrix(data=data, sim_func="jaccard")

    if inverse is True:
        theta_list = theta_list[::-1].tolist()
    else:
        theta_list = theta_list.tolist()

    if cores_max <= len(theta_list):
        num_cores = cores_max
    else:
        num_cores = len(theta_list)

    with mp.Pool(processes=num_cores) as pool:
        clustertables = pool.starmap(
            work_clustertable, zip(repeat(sim_matrix), theta_list, repeat(outlier_size))
        )

        pool.close()
        pool.join()

    clustertable = concat(clustertables)

    print(f"Clustertable of size '{clustertable.shape}' formed successfully!")

    return clustertable


def find_perftable(
    data: DataFrame,
    red_cltable: DataFrame,
    try_k: int,
    labels: DataFrame,
    outlier_size: int,
) -> DataFrame:

    """Define range of 'thetas' by start, stop, step. Start and stop are inclusive.
    Give parameters as float!
    Inverse order with inverse=True/False. Default = ascending order!
    Computes greatest theta for which is True: Any point in corresponding link-matrix has at least one link!
    Returns 'theta_min' as float"""

    cores_max = mp.cpu_count()
    theta_list = red_cltable.index.tolist()
    y_true = labels["EVAL_Test"].values
    perf_list = []

    for theta in theta_list:
        if red_cltable.at[theta, "NUM_OUTLIER"] < 1:
            continue
        else:
            init_k = red_cltable.at[theta, "NUM_CLUSTER"]
            try_k_list = [*range(init_k, init_k + try_k, 1)]
            print(f"\nFinding optimal k for theta = '{theta}'. Patience ...\n")

            if cores_max <= len(try_k_list):
                num_cores = cores_max
            else:
                num_cores = len(try_k_list)

            with mp.Pool(processes=num_cores) as pool:
                perf_dicts = pool.starmap(
                    work_perftable,
                    zip(
                        repeat(data),
                        repeat(y_true),
                        repeat(theta),
                        repeat(outlier_size),
                        repeat(init_k),
                        try_k_list,
                    ),
                )

                pool.close()
                pool.join()

            perf_list += perf_dicts

            print(
                f"\n{len(perf_dicts)} entries added to performance-table of for:\n- theta: '{theta}'\n- k-sizes: '{try_k_list[0]} to {try_k_list[-1]}'\n"
            )
            print(
                f"Current size of performance-table: '{len(perf_list)}'\n-> Getting new theta! Patience ...\n"
            )

    perf_table = DataFrame(data=perf_list)

    print(f"Performance-table of size '{perf_table.shape}' formed successfully!")

    return perf_table

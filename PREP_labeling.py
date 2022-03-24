import pandas as pd
import datetime as dt
import sys
from CALC_algorithms import rock, similarity_matrix
from CLEAN_main import (
    df_raw,
    df_B,
    df_A,
    df_trial,
    df_C,
    df_C_cod,
    df_A_cod,
    df_B_cod,
    df_labels,
)


def get_clusters(
    data: pd.DataFrame,
    cl_table: pd.DataFrame,
    try_k: int,
    size: int,
    theta: float,
    outlier_size: int,
) -> None:
    """Imports relevants thresholds, returns clusters with indicated cluster-size 'size', returns indices of smallest clusters first.
    Accepts user-input as labels, stores user-input in column, creates additional column 'Evaluation' = True/False"""

    k = cl_table.loc[theta, "NUM_CLUSTER"] + try_k
    clusters = rock(data=data, theta=theta, k=k, outlier_size=outlier_size)
    rel_indices = set(clusters.index)

    for v in range(1, size + 1):
        rel_indices.discard(v)

    clusters = clusters.sort_index(ascending=True).drop(index=rel_indices)

    return clusters


def multi_labeling(indices: list, user_input: str, labels: list):
    """Multi-labeling"""
    user_input = user_input.split(", ")
    multi_i = pd.Series(index=df_LABELS.loc[indices, "EVAL_Label"], data=indices)

    for l in labels:
        if l in multi_i.index:
            multi_i = multi_i.drop([l])

    if user_input[0] == "T":
        df_LABELS.loc[multi_i, "EVAL_Test"] = False
        df_LABELS.loc[multi_i, "EVAL_Label"] = user_input[0]
    elif user_input[0] in ["U", "N", "D", "F", "M"]:
        df_LABELS.loc[multi_i, "EVAL_Test"] = True
        df_LABELS.loc[multi_i, "EVAL_Label"] = user_input[0]

    print(
        f"\nMulti labeling:\n'{len(multi_i)}' elements out of '{len(indices)}' elements in cluster have been labeled '{user_input[0]}'!\n"
    )

    return True


def labeling(
    data: pd.DataFrame, cl_table: pd.DataFrame, try_k: int, size: int, inverse: bool
) -> None:
    """Data-labeling by User"""

    if inverse is True:
        thetas = cl_table.index[::-1]
    else:
        thetas = cl_table.index[::1]

    for theta in thetas:

        clusters = get_clusters(
            data=data,
            cl_table=cl_table,
            try_k=try_k,
            size=size,
            theta=theta,
            outlier_size=0,
        )

        for index_list in clusters:
            labels = ["U", "N", "D", "M", "F", "T"]
            multi_labels = ["U, all", "N, all", "D, all", "M, all", "F, all", "T, all"]

            for i in index_list:
                if df_LABELS.loc[i, "EVAL_Label"] in labels:
                    continue  # Filtering function can be added here!

                user_input = "to be defined by user"
                print(
                    f"\nDoor to be labeled:\nIndex: '{i}'\nTuernummer: '{df_raw.loc[i, 'ALLG_Nummer']}'\nGeschoss: '{df_raw.loc[i, 'ALLG_Geschoss']}'\n\nIn Cluster: '{index_list}'"
                )
                while (
                    user_input not in labels
                    and user_input not in multi_labels
                    and user_input != "exit"
                    and user_input != "skip"
                ):
                    user_input = input(
                        f"Please indicate if element '{i}' is part of group:\n'U' = Unique Element\n'N' = Naming logic\n'D' = Design logic\n'M' = Missing value\n'F' = Reg. element (False)\n'T' = Reg. element (True)\n\nType 'skip' to skip current element\n\nType 'exit' to safe work and stop program\n"
                    )

                if user_input in multi_labels:
                    multi_labels = multi_labeling(
                        indices=index_list, user_input=user_input, labels=labels
                    )

                elif user_input == "T":
                    df_LABELS.loc[i, "EVAL_Test"] = False
                    df_LABELS.loc[i, "EVAL_Label"] = user_input
                    print(f"Element '{i}' labeled as '{user_input}', rated as 'True'!")
                elif user_input in ["U", "N", "D", "F", "M"]:
                    df_LABELS.loc[i, "EVAL_Test"] = True
                    df_LABELS.loc[i, "EVAL_Label"] = user_input
                    print(f"Element '{i}' labeled as '{user_input}', rated as 'False'!")
                elif user_input == "skip":
                    None
                elif user_input == "exit":
                    timestamp = dt.datetime.now().strftime("%y%m%d-%H-%M")
                    df_LABELS.to_csv(
                        f"C:\\Users\\yann-\\Documents\\03_Education\\03.1_CAS-MachineLearning\\XX_Workspaces\\211202_CAS-MS\\DF_LABELS\\134_{timestamp}_DF_Labels.csv"
                    )
                    print(
                        f"Current work saved to '134_{timestamp}_DF_Labels.csv' -> Exits program"
                    )
                    sys.exit()
                if i == index_list[-1] or multi_labels is True:
                    timestamp = dt.datetime.now().strftime("%y%m%d-%H-%M")
                    df_LABELS.to_csv(
                        f"C:\\Users\\yann-\\Documents\\03_Education\\03.1_CAS-MachineLearning\\XX_Workspaces\\211202_CAS-MS\\DF_LABELS\\134_{timestamp}_DF_Labels.csv"
                    )
                    print(
                        f"\nFinished labeling cluster'{index_list}'\nCurrent work saved to '134_{timestamp}_DF_Labels.csv'\n -> Getting new cluster"
                    )

            if index_list == clusters.iloc[-1]:
                print(
                    f"\nLabeling done for current theta = '{theta}' -> Getting new theta: Patience!"
                )
            else:
                None


# df_B_clustertable = pd.read_csv("134_220219-19-57_df_Bb_clustertable.csv", index_col=0)
# df_B_rel_thetas = pd.read_csv("134_220217-18-04_df_trial_rel-thetas.csv", index_col=0)
# df_LABELS = pd.read_csv("DF_LABELS\\134_220220-01-15_DF_Labels.csv")
# df_B_rel_thetas = pd.read_csv("134_220221-13-39_df_B_red-clustertable.csv", index_col=0)

"""
df_LABELS = pd.DataFrame(
    index=df_trial.index, data=df_trial.index.tolist(), columns=["Name"]
)
df_LABELS.insert(1, "EVAL_Label", "X")
df_LABELS.insert(2, "EVAL_Test", False)
sim_matrix_trial = similarity_matrix(data=df_trial, sim_func="jaccard")
df_trial_clustertable = pd.read_csv(
    "134_220228-19-58_df_trial_clustertable.csv", index_col=0
)


labeling(
    data=sim_matrix_trial,
    cl_table=df_trial_clustertable,
    try_k=0,
    size=12,
    inverse=False,
)"""

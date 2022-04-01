if __name__ == "__main__":

    import datetime as dt
    import sys
    from pandas import DataFrame, read_csv
    from CALC_hyperpar_func import find_theta_min, find_clustertable, find_perftable
    from PREP_labeling import get_clusters
    from CALC_algorithms import similarity_matrix
    from CLEAN_main import (
        df_B_cod,
        df_B_rcltable,
        df_C_cod,
        df_C_rcltable,
        df_A_cod,
        df_trial,
        df_labels,
        df_labels_trial,
    )

    def get_clustertable(
        data: DataFrame,
        start: float,
        stop: float,
        step: float,
        inverse: bool,
        outlier_size: int,
    ) -> DataFrame:

        """Returns clustertable for given dataset!"""

        sim_matrix = similarity_matrix(data=data, sim_func="jaccard")
        clustertable = find_clustertable(
            data=sim_matrix,
            start=start,
            stop=stop,
            step=step,
            inverse=inverse,
            outlier_size=outlier_size,
        )
        red_clustertable = clustertable.drop_duplicates(keep="first")

        return clustertable, red_clustertable

    def get_theta_min(
        data: DataFrame, start: float, stop: float, step: float, inverse: bool
    ) -> float:

        """Computes smallest theta for rock-alorithm which returns at least one outlier (clustersize = 1)"""
        sim_matrix = similarity_matrix(data=data, sim_func="jaccard")
        init_theta = find_theta_min(
            data=sim_matrix, start=start, stop=stop, step=step, inverse=inverse
        )

        return init_theta

    def get_perftable(
        data: DataFrame,
        red_cltable: DataFrame,
        df_labels: DataFrame,
        try_k: int,
        outlier_size: int,
    ) -> DataFrame:

        """Gets auc table"""
        sim_matrix = similarity_matrix(data=data, sim_func="jaccard")
        perf_table = find_perftable(
            data=sim_matrix,
            red_cltable=red_cltable,
            labels=df_labels,
            try_k=try_k,
            outlier_size=outlier_size,
        )
        return perf_table

    def init_labeling_perf(
        data: DataFrame, cl_table: DataFrame, try_k: int, size: int, inverse: bool
    ) -> DataFrame:

        """Data-labeling by User"""

        perf_list = []
        tracking_list = []

        if inverse is True:
            thetas = cl_table.index[::-1]
        else:
            thetas = cl_table.index[::1]

        for theta in thetas:
            sim_matrix = similarity_matrix(data=data, sim_func="jaccard")
            clusters = get_clusters(
                data=sim_matrix,
                cl_table=cl_table,
                try_k=try_k,
                size=size,
                theta=theta,
                outlier_size=0,
            )

            for index_list in clusters:
                labels = ["U", "N", "D", "M", "F", "T"]

                for i in index_list:
                    if i in tracking_list:
                        continue
                    else:
                        if df_labels.loc[i, "EVAL_Label"] in labels:
                            tracking_dict = {
                                "Element": i,
                                "Theta": theta,
                                "Label": df_labels.loc[i, "EVAL_Label"],
                                "Pred": True,
                                "Validation": df_labels.loc[i, "EVAL_Test"],
                            }

                            perf_list.append(tracking_dict)
                            tracking_list.append(i)
                        else:
                            print(
                                f"\nError occured - door '{i}' detected as 'unlabeled'!?\n"
                            )
                            sys.exit()

            print(
                f"\nAll labeling-activity for theta = '{theta}' logged:\n-> Getting new theta! Patience ..."
            )
        perf_table_labeling = DataFrame(data=perf_list)
        print(
            f"\nAll labeling trackend. Performance-table of size '{perf_table_labeling.shape}' created successfully!\n"
        )
        return perf_table_labeling

    def safe_tables(input_data: str, stop: bool) -> None:
        """Writes clustertable and relevant-theats to disk"""

        name = input_data
        timestamp = dt.datetime.now().strftime("%y%m%d-%H-%M")

        perf_table.to_csv(f"134_{timestamp}_{name}_perftable.csv")

        # clustertable.to_csv(
        # f"134_{timestamp}_{name}_clustertable.csv"
        # )

        # red_clustertable.to_csv(
        # f"134_{timestamp}_{name}_red-clustertable.csv"
        # )
        if stop is True:
            sys.exit()
        else:
            None

    perf_table = get_perftable(
        data=df_B_cod,
        red_cltable=df_B_rcltable,
        df_labels=df_labels,
        try_k=150,
        outlier_size=7,
    )

    safe_tables(input_data="df_B7", stop=False)

    perf_table = get_perftable(
        data=df_C_cod,
        red_cltable=df_C_rcltable,
        df_labels=df_labels,
        try_k=150,
        outlier_size=7,
    )

    safe_tables(input_data="df_C7", stop=True)

    label_dfB1 = read_csv("134_220303-15-33_df_B-1_label_perftable.csv", index_col=0)
    df_B_outliers = df_B_cod.loc[label_dfB1["Element"], :]

    clustertable_dfB_out, red_clustertable = get_clustertable(
        data=df_B_outliers,
        start=5,
        stop=250,
        step=5,
        inverse=False,
        outlier_size=0,
    )

    safe_tables(input_data="df_B1_labels", stop=True)

    for k_size in [1, 7, 11]:

        perf_table = init_labeling_perf(
            data=df_B_cod, cl_table=df_B_rcltable, try_k=0, size=k_size, inverse=False
        )

        safe_tables(input_data=f"df_B-{k_size}_label", stop=False)

    for k_size in [1, 7]:

        perf_table = init_labeling_perf(
            data=df_C_cod, cl_table=df_C_rcltable, try_k=0, size=k_size, inverse=False
        )

        safe_tables(input_data=f"df_C-{k_size}_label", stop=False)

    perf_table = init_labeling_perf(
        data=df_C_cod, cl_table=df_C_rcltable, try_k=0, size=11, inverse=False
    )

    safe_tables(input_data="df_C-11_label", stop=True)

    print("Done doing all jobs!")

    for i in [2, 3, 4, 5, 6, 7, 8, 11]:

        perf_table = get_perftable(
            data=df_C_cod,
            red_cltable=df_C_rcltable,
            df_labels=df_labels,
            try_k=15,
            outlier_size=i,
        )
        safe_tables(input_data=f"df_C15-{i}")

    print("Done doing all jobs!")

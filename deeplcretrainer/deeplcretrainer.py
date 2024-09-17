"""
Transfer learning with DeepLC
"""

__author__ = ["Robbin Bouwmeester", "Arthur Declercq", "Ralf Gabriels"]
__credits__ = [
    "Robbin Bouwmeester",
    "Ralf Gabriels",
    "Arthur Declercq",
    "Prof. Lennart Martens",
    "Prof. Sven Degroeve",
]
__license__ = "Apache License, Version 2.0"
__maintainer__ = ["Robbin Bouwmeester", "Ralf Gabriels"]
__email__ = ["Robbin.Bouwmeester@ugent.be", "Ralf.Gabriels@ugent.be"]

import hashlib
import itertools
import os
import tempfile

import pandas as pd
from psm_utils.io.peptide_record import peprec_to_proforma
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

try:
    from gooey import Gooey, GooeyParser
except ImportError:

    def Gooey(
        program_name="DeepLC re-tR-ainer",
        default_size=(720, 790),
        monospace_display=True,
    ):
        def wrapper(f):
            return ""

        return wrapper


try:
    from deeplcretrainer import cnn_functions
except ImportError:
    try:
        import cnn_functions
    except:
        pass

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf


def parse_arguments(gui=False):
    """Read arguments from the CLI or GUI."""
    parser = GooeyParser(
        prog="DeepLC_retrain",
        description=(
            "Retention time prediction for (modified) peptides using deep " "learning."
        ),
    )

    input_args = parser.add_argument_group("Input data and models")

    input_args.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        # default=None,
        metavar="Data sets to fit models",
        widget="MultiFileChooser",
        help=("Data sets to fit models"),
    )

    input_args.add_argument(
        "--mods_transfer_learning",
        nargs="+",
        # default=None,
        required=False,
        metavar="Model file(s) to perform transfer learning",
        widget="MultiFileChooser",
        help=("Model file(s) to perform transfer learning"),
    )

    input_args.add_argument(
        "--costum_modification_file",
        nargs="+",
        # default=None,
        required=False,
        metavar="Costum file with modifications",
        widget="FileChooser",
        help=("Costum file with modifications"),
    )

    training_args = parser.add_argument_group("Training parameters")

    training_args.add_argument(
        "--n_epochs",
        type=int,
        dest="n_epochs",
        default=10,
        widget="IntegerField",
        metavar="n_epochs",
        gooey_options={"min": 1, "max": 250, "increment": 10},
        help=("n_epochs"),
    )

    training_args.add_argument(
        "--freeze_layers",
        dest="freeze_layers",
        widget="CheckBox",
        metavar="freeze_layers",
        action="store_true",
        help=("Freeze layers (paths) before the concatenation"),
    )

    training_args.add_argument(
        "--batch_size",
        type=int,
        dest="batch_size",
        default=128,
        widget="IntegerField",
        metavar="batch_size",
        gooey_options={"min": 1, "max": 102400, "increment": 64},
        help=("batch_size"),
    )

    output_args = parser.add_argument_group("Output files")

    output_args.add_argument(
        "--outpath", type=str, required=True, dest="outpath", widget="DirChooser"
    )

    results = parser.parse_args()

    return results


def retrain(
    datasets=[],
    mods_transfer_learning=[],
    n_epochs=75,
    batch_size=128,
    ratio_test=0.9,
    ratio_valid=0.95,
    freeze_layers=False,
    costum_modification_file=None,
    plot_results=False,
    write_csv_results=False,
    freeze_after_concat=0,
    outpath=None,
    a_blocks=[3],
    a_kernel=[2, 4, 8],
    a_max_pool=[2],
    a_filters_start=[256],
    a_stride=[1],
    b_blocks=[2],
    b_kernel=[2],
    b_max_pool=[2],
    b_filters_start=[128],
    b_stride=[1],
    global_neurons=[16],
    global_num_dens=[3],
    regularizer_val=[0.0000025],
    remove_repeats=True,
    verbose=True,
    **kwargs
):
    if not outpath:
        outpath = tempfile.mkdtemp()

    params = list(
        itertools.product(
            *[
                a_blocks,
                a_kernel,
                a_max_pool,
                a_filters_start,
                a_stride,
                b_blocks,
                b_kernel,
                b_max_pool,
                b_filters_start,
                b_stride,
                global_neurons,
                global_num_dens,
                regularizer_val,
            ]
        )
    )

    #lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1, max_value=20.0)

    fit_hc = True
    use_correction_factor = True
    hc_str = "_hc"
    mods_loc_optimized_all = []

    if type(datasets) == str:
        datasets = [datasets]

    # catch above list creation for PSM list
    if type(datasets) == list:
        datasets = dict([(".".join(d.split(".")[:-1]), d) for d in datasets])
        for dataset_name, file_name in datasets.items():
            list_of_psms = []
            datasets[dataset_name] = pd.read_csv(file_name)
            datasets[dataset_name].fillna("", inplace=True)
            datasets[dataset_name].rename(
                columns={"observed_retention_time": "tr", "peptide": "seq"},
                inplace=True,
            )

            if remove_repeats:
                datasets[dataset_name].sort_values("tr", inplace=True)
                datasets[dataset_name].drop_duplicates(
                    ["seq", "modifications"], keep="first", inplace=True
                )

            if "charge" in datasets[dataset_name].columns:
                for seq, mod, id, tr, z in zip(
                    datasets[dataset_name]["seq"],
                    datasets[dataset_name]["modifications"],
                    datasets[dataset_name].index,
                    datasets[dataset_name]["tr"],
                    datasets[dataset_name]["charge"],
                ):
                    list_of_psms.append(
                        PSM(
                            peptidoform=peprec_to_proforma(seq, mod, charge=z),
                            spectrum_id=id,
                            retention_time=tr,
                        )
                    )
            else:
                for seq, mod, id, tr in zip(
                    datasets[dataset_name]["seq"],
                    datasets[dataset_name]["modifications"],
                    datasets[dataset_name].index,
                    datasets[dataset_name]["tr"],
                ):
                    list_of_psms.append(
                        PSM(
                            peptidoform=peprec_to_proforma(seq, mod),
                            spectrum_id=id,
                            retention_time=tr,
                        )
                    )
            psm_list = PSMList(psm_list=list_of_psms)

            datasets[dataset_name] = psm_list

    elif remove_repeats and type(datasets) == dict:
        for dataset_name, psm_list in datasets.items():
            psm_list_with_tr = [
                (psm, psm.retention_time, psm.peptidoform) for psm in psm_list
            ]
            psm_list_with_tr = sorted(psm_list_with_tr, key=lambda l: l[1])

            prev_analyzed = set()
            list_of_psms = []

            for psm, tr, proforma_seq in psm_list_with_tr:
                if proforma_seq not in prev_analyzed:
                    list_of_psms.append(psm)
                    prev_analyzed.add(proforma_seq)
                else:
                    continue
            psm_list = PSMList(psm_list=list_of_psms)
            datasets[dataset_name] = psm_list

    for dataset_name, psm_list in datasets.items():
        df = cnn_functions.get_feat_df(
            psm_list=psm_list,
            costum_modification_file=costum_modification_file,
            **kwargs
        )

        correction_factor = 1.0

        df_train, df_test = cnn_functions.train_test(df, ratio_train=ratio_test)
        df_train, df_valid = cnn_functions.train_test(df_train, ratio_train=ratio_valid)

        (
            X_train,
            X_train_sum,
            X_train_global,
            X_train_hc,
            y_train,
        ) = cnn_functions.get_feat_matrix(df_train)
        (
            X_valid,
            X_valid_sum,
            X_valid_global,
            X_valid_hc,
            y_valid,
        ) = cnn_functions.get_feat_matrix(df_valid)
        (
            X_test,
            X_test_sum,
            X_test_global,
            X_test_hc,
            y_test,
        ) = cnn_functions.get_feat_matrix(df_test)

        y_train = y_train / correction_factor
        y_valid = y_valid / correction_factor
        y_test = y_test / correction_factor

        mods_optimized = []
        mods_loc_optimized = []

        for p in params:
            (
                a_blocks,
                a_kernel,
                a_max_pool,
                a_filters_start,
                a_stride,
                b_blocks,
                b_kernel,
                b_max_pool,
                b_filters_start,
                b_stride,
                global_neurons,
                global_num_dens,
                regularizer_val,
            ) = p
            param_hash = hashlib.md5(",".join(map(str, p)).encode()).hexdigest()
            mod_name = os.path.join(
                outpath,
                "full%s_%s_%s.keras"
                % (
                    hc_str,
                    os.path.basename(dataset_name).replace(".csv", ""),
                    param_hash,
                ),
            )

            matched_mod = ""
            for mod_name_t in mods_transfer_learning:
                if param_hash in mod_name_t:
                    matched_mod = mod_name_t
                    break
            if len(matched_mod) == 0 and len(mods_transfer_learning) > 0:
                continue
            elif len(matched_mod) > 0:
                freeze_after_concat_t = int(freeze_after_concat)
                model = load_model(
                    matched_mod #, custom_objects={"<lambda>": lrelu}
                )

                if freeze_layers:
                    set_train_to = False
                    start_count = False
                    for layer in model.layers:
                        if "concatenate" in layer.name or start_count:
                            start_count = True
                            if freeze_after_concat_t < 1:
                                set_train_to = True
                            freeze_after_concat_t -= 1
                        layer.trainable = set_train_to
                model.compile(
                    loss="mean_absolute_error",
                    optimizer="Adam",
                    metrics=["mean_absolute_error"],
                )
            else:
                model = cnn_functions.init_model(
                    X_train,
                    X_train_sum,
                    X_train_global,
                    X_test_hc,
                    a_blocks=a_blocks,
                    a_kernel=a_kernel,
                    a_max_pool=a_max_pool,
                    a_filters_start=a_filters_start,
                    a_stride=a_stride,
                    b_blocks=b_blocks,
                    b_kernel=b_kernel,
                    b_max_pool=b_max_pool,
                    b_filters_start=b_filters_start,
                    b_stride=b_stride,
                    global_neurons=global_neurons,
                    global_num_dens=global_num_dens,
                    regularizer_val=regularizer_val,
                    fit_hc=fit_hc,
                )

            mcp_save = ModelCheckpoint(
                mod_name,
                save_best_only=True,
                monitor="val_mean_absolute_error",
                mode="min",
            )
            if fit_hc:
                history = model.fit(
                    [X_train, X_train_sum, X_train_global, X_train_hc],
                    y_train,
                    validation_data=(
                        [X_valid, X_valid_sum, X_valid_global, X_valid_hc],
                        y_valid,
                    ),
                    epochs=n_epochs,
                    verbose=int(verbose),
                    batch_size=batch_size,
                    callbacks=[mcp_save],
                    shuffle=True,
                )
            else:
                history = model.fit(
                    [X_train, X_train_sum, X_train_global],
                    y_train,
                    validation_data=([X_valid, X_valid_sum, X_valid_global], y_valid),
                    epochs=n_epochs,
                    verbose=int(verbose),
                    batch_size=batch_size,
                    callbacks=[mcp_save],
                    shuffle=True,
                )

            mods_optimized.append(
                load_model(mod_name) #, custom_objects={"<lambda>": lrelu}
            )

            mods_loc_optimized.append(mod_name)

        if write_csv_results:
            cnn_functions.write_preds(
                df_train,
                X_train,
                X_train_sum,
                X_train_global,
                X_train_hc,
                mods_optimized,
                fit_hc=fit_hc,
                correction_factor=correction_factor,
                outfile_name=os.path.join(
                    outpath,
                    "%s_full%s_train.csv"
                    % (os.path.basename(dataset_name).replace(".csv", ""), hc_str),
                ),
            )

            cnn_functions.write_preds(
                df_valid,
                X_valid,
                X_valid_sum,
                X_valid_global,
                X_valid_hc,
                mods_optimized,
                fit_hc=fit_hc,
                correction_factor=correction_factor,
                outfile_name=os.path.join(
                    outpath,
                    "%s_full%s_valid.csv"
                    % (os.path.basename(dataset_name).replace(".csv", ""), hc_str),
                ),
            )

            perf = cnn_functions.write_preds(
                df_test,
                X_test,
                X_test_sum,
                X_test_global,
                X_test_hc,
                mods_optimized,
                fit_hc=fit_hc,
                correction_factor=correction_factor,
                outfile_name=os.path.join(
                    outpath,
                    "%s_full%s_test.csv"
                    % (os.path.basename(dataset_name).replace(".csv", ""), hc_str),
                ),
            )

        if plot_results:
            cnn_functions.plot_preds(
                X_train,
                X_train_sum,
                X_train_global,
                X_train_hc,
                y_train,
                mods_optimized,
                fit_hc=fit_hc,
                correction_factor=correction_factor,
                file_save=os.path.join(
                    outpath,
                    "%s_full_hc_train.png"
                    % (os.path.basename(dataset_name).replace(".csv", "")),
                ),
                plot_title="%s_full%s_train"
                % (os.path.basename(dataset_name).replace(".csv", ""), hc_str),
            )

            cnn_functions.plot_preds(
                X_valid,
                X_valid_sum,
                X_valid_global,
                X_valid_hc,
                y_valid,
                mods_optimized,
                fit_hc=fit_hc,
                correction_factor=correction_factor,
                file_save=os.path.join(
                    outpath,
                    "%s_full_hc_valid.png"
                    % (os.path.basename(dataset_name).replace(".csv", "")),
                ),
                plot_title="%s_full%s_valid"
                % (os.path.basename(dataset_name).replace(".csv", ""), hc_str),
            )

            cnn_functions.plot_preds(
                X_test,
                X_test_sum,
                X_test_global,
                X_test_hc,
                y_test,
                mods_optimized,
                fit_hc=fit_hc,
                correction_factor=correction_factor,
                file_save=os.path.join(
                    outpath,
                    "%s_full_hc_test.png"
                    % (os.path.basename(dataset_name).replace(".csv", "")),
                ),
                plot_title="%s_full%s_test"
                % (os.path.basename(dataset_name).replace(".csv", ""), hc_str),
            )

        mods_loc_optimized_all.append(mods_loc_optimized)

    if len(mods_loc_optimized_all) == 1:
        return mods_loc_optimized_all[0]
    else:
        return mods_loc_optimized_all


# @Gooey(
#    program_name="DeepLC re-tR-ainer",
#    default_size=(720, 790),
#    monospace_display=True
# )
def main():
    argu = parse_arguments()
    retrain(**vars(argu))


if __name__ == "__main__":
    Gooey(main)()

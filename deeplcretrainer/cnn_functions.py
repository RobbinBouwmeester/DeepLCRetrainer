"""
Main code used for evaluating DeepLC
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

import math
import multiprocessing
import os
import random
import time

import deeplc
import numpy as np
import pandas as pd
import tensorflow as tf
from deeplc.feat_extractor import FeatExtractor
from psm_utils.io.peptide_record import peprec_to_proforma
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Flatten,
    Input,
    MaxPooling1D,
    concatenate,
)
from tensorflow.keras.models import Model

try:
    import matplotlib
    from matplotlib import pyplot as plt

    matplotlib.use("Agg")
except ImportError:
    _has_matplotlib = False
else:
    _has_matplotlib = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_infile(infile_loc):
    df = pd.read_csv(
        infile_loc,
        sep=",",
        low_memory=False,
        dtype={"seq": "str", "modifications": "str", "tr": "float32"},
    )

    df.index = ["Pep_" + str(ide) for ide in df.index]

    min_tr = df["tr"].min()
    if min_tr < 0:
        df["tr"] = df["tr"] + abs(min_tr)
    df["modifications"].fillna("", inplace=True)

    return df


def read_infile_new(infile_loc):
    p = 0.0075
    df = pd.read_csv(
        infile_loc,
        sep=",",
        low_memory=False,
        dtype={
            "seq": "str",
            "modifications": "str",
            "tr": "float32",
            "first": "float32",
            "seq2": "str",
            "modifications2": "str",
        },
        skiprows=lambda i: i > 0 and random.random() > p,
    )

    df.index = ["Pep_" + str(ide) for ide in df.index]

    # min_tr = df["tr"].min()
    # if min_tr < 0: df["tr"] = df["tr"]+abs(min_tr)
    df["modifications"].fillna("", inplace=True)

    return df


def read_aa_lib(infile, reset_to_glycine=""):
    aa_comp_pd_dict = pd.read_csv(infile, index_col=0).T.to_dict()

    aa_comp = {}

    for aa, v1 in aa_comp_pd_dict.items():
        aa_comp[aa] = {}
        for atom, v2 in aa_comp_pd_dict[aa].items():
            if v2 != 0:
                aa_comp[aa][atom] = v2
            if aa == reset_to_glycine:
                aa_comp[aa] = aa_comp_pd_dict["G"]

    aa_comp["X"] = {"C": 0}

    return aa_comp


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def calc_feats_mods(formula):
    """
    Chemical formula to atom addition/subtraction

    Parameters
    ----------
    formula : str
        chemical formula

    Returns
    -------
    list
        atom naming
    list
        number of atom added/subtracted
    """
    if not formula:
        return [], []
    if len(str(formula)) == 0:
        return [], []
    if type(formula) != str:
        if math.isnan(formula):
            return [], []

    new_atoms = []
    new_num_atoms = []
    for atom in formula.split(" "):
        if "(" not in atom:
            atom_symbol = atom
            num_atom = 1
        else:
            atom_symbol = atom.split("(")[0]
            num_atom = atom.split("(")[1].rstrip(")")
        new_atoms.append(atom_symbol)
        new_num_atoms.append(num_atom)
    return new_atoms, map(int, new_num_atoms)


def get_libs_mods(directory):
    """
    Make a dictionary with unimod to chemical formula

    Parameters
    ----------
    directory : str
        directory of the unimod to chemical formula mapping

    Returns
    -------
    dict
        chemical formula of a PTM when it is added
    dict
        chemical formula of a PTM when it is subtracted
    """
    # TODO replace dir with actual file...
    mod_df = pd.read_csv(os.path.join(directory, "unimod_to_formula.csv"), index_col=0)
    mod_dict = mod_df.to_dict()
    return mod_dict["formula_pos"], mod_dict["formula_neg"]


def encode_atoms(
    seqs_df,
    padding_length=60,
    aa_comp={},
    positions=set([0, 1, 2, 3, -1, -2, -3, -4]),
    sum_mods=2,
    ignore_mods=False,
    dict_index_pos={"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5},
    dict_index_all={"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5},
    dict_index={"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5},
):
    ret_list = {}
    ret_list_sum = {}
    ret_list_all = {}
    ret_list_pos = {}

    seqs = seqs_df["seq"]
    indexes = seqs_df.index
    mods_all = seqs_df["modifications"]

    for row_index, seq, mods in zip(indexes, seqs, mods_all):
        seq_len = len(seq)
        if seq_len > padding_length:
            continue

        padding = "".join(["X"] * (padding_length - len(seq)))
        seq = seq + padding

        matrix = np.zeros((len(seq), len(dict_index.keys())), dtype=np.float16)
        matrix_sum = np.zeros(
            (int(len(seq) / sum_mods), len(dict_index.keys())), dtype=np.float32
        )
        matrix_all = np.zeros(len(dict_index_all.keys()) + 1, dtype=np.float32)
        matrix_pos = np.zeros(
            (len(positions), len(dict_index.keys())), dtype=np.float16
        )

        matrix_all[len(dict_index_all.keys())] = len(seq)

        for index, aa in enumerate(seq):
            if aa == "X":
                break
            index_sum = int(index / sum_mods)

            for atom, val in aa_comp[aa].items():
                matrix[index, dict_index[atom]] = val
                matrix_sum[index_sum, dict_index[atom]] += val
                matrix_all[dict_index_all[atom]] += val

                if index in positions:
                    matrix_pos[index, dict_index_pos[atom]] = val
                elif index - seq_len in positions:
                    matrix_pos[index - seq_len, dict_index_pos[atom]] = val

        if len(mods) == 0:
            ret_list[row_index] = {"index_name": row_index, "matrix": matrix}
            ret_list_sum[row_index] = {
                "index_name": row_index,
                "matrix_sum": matrix_sum,
            }
            ret_list_all[row_index] = {
                "index_name": row_index,
                "matrix_all": matrix_all,
            }
            ret_list_pos[row_index] = {
                "index_name": row_index,
                "pos_matrix": matrix_pos.flatten(),
            }
            continue

        # mods = row["modifications"].split("|")

        lib_add, lib_subtract = get_libs_mods(os.path.join(os.getcwd(), "unimod/"))

        mods = mods.split("|")
        for i in range(1, len(mods), 2):
            if ignore_mods:
                continue
            mod = mods[i]
            try:
                fill_mods, fill_num = calc_feats_mods(lib_add[mod])
            except:
                continue
            subtract_mods, subtract_num = calc_feats_mods(lib_subtract[mod])

            loc = int(mods[i - 1]) - 1
            if loc > len(seq):
                loc = len(seq) - 1

            for atom, atom_change in zip(fill_mods, fill_num):
                try:
                    matrix[loc, dict_index[atom]] += atom_change
                    matrix_all[dict_index_all[atom]] += val
                    if loc in positions:
                        matrix_pos[loc, dict_index_pos[atom]] += val
                    elif loc - len(seq) in positions:
                        matrix_pos[loc - len(seq), dict_index_pos[atom]] += val

                except KeyError:
                    pass
                except IndexError:
                    print(
                        "Index does not exist for: ", atom, atom_change, ident, mod, seq
                    )

            for atom, atom_change in zip(subtract_mods, subtract_num):
                try:
                    matrix[loc, dict_index[atom]] -= atom_change
                    matrix_all[dict_index_all[atom]] -= val
                    if loc in positions:
                        matrix_pos[loc, dict_index_pos[atom]] -= val
                    elif loc - len(seq) in positions:
                        matrix_pos[loc - len(seq), dict_index_pos[atom]] -= val

                except KeyError:
                    pass
                except IndexError:
                    print(
                        "Index does not exist for: ", atom, atom_change, ident, mod, seq
                    )

        ret_list[row_index] = {"index_name": row_index, "matrix": matrix}
        ret_list_sum[row_index] = {"index_name": row_index, "matrix_sum": matrix_sum}
        ret_list_all[row_index] = {"index_name": row_index, "matrix_all": matrix_all}
        ret_list_pos[row_index] = {
            "index_name": row_index,
            "pos_matrix": matrix_pos.flatten(),
        }

    return (
        pd.DataFrame.from_dict(ret_list).T,
        pd.DataFrame.from_dict(ret_list_sum).T,
        pd.DataFrame.from_dict(ret_list_pos).T,
        pd.DataFrame.from_dict(ret_list_all).T,
    )


def count_aa(
    df,
    aas=[
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ],
):
    ret_list = {}

    seqs = df["seq"]
    indexes = df.index

    for s, i in zip(seqs, indexes):
        ret_list[i] = {}
        for aa in s:
            ret_list[i][aa] = int(s.count(aa))
    return pd.DataFrame(ret_list).T


def add_count_aa(df):
    df_aa = count_aa(df).fillna(0)


def get_feat_df(
    df=None,
    psm_list=None,
    aa_comp={},
    costum_modification_file=None,
    num_cores=False,
    ignore_mods=False,
    standard_feat=False,
    **kwargs,
):
    if not num_cores:
        num_cores = multiprocessing.cpu_count()
    num_cores = 1
    if costum_modification_file:
        if type(costum_modification_file) == list:
            costum_modification_file = costum_modification_file[0]
        if costum_modification_file.endswith(".csv"):
            costum_modification_file = os.path.dirname(
                os.path.abspath(costum_modification_file)
            )
        f_extractor = FeatExtractor(
            cnn_feats=True,
            verbose=True,
            lib_path_mod=costum_modification_file,
            ignore_mods=ignore_mods,
        )
    else:
        f_extractor = FeatExtractor(
            cnn_feats=True, verbose=True, ignore_mods=ignore_mods
        )

    pepper = deeplc.DeepLC(
        f_extractor=f_extractor,
        cnn_model=True,
        n_jobs=num_cores,
        verbose=True,
        **kwargs,
    )

    if type(df) == pd.core.frame.DataFrame:
        list_of_psms = []
        if "charge" in df.columns:
            for seq, mod, id, tr, z in zip(
                df["seq"], df["modifications"], df.index, df["tr"], df["charge"]
            ):
                list_of_psms.append(
                    PSM(
                        peptidoform=peprec_to_proforma(seq, mod, charge=z),
                        spectrum_id=id,
                        retention_time=tr,
                    )
                )
                print("Added the charges for CCS prediction")
        else:
            for seq, mod, id, tr in zip(
                df["seq"], df["modifications"], df.index, df["tr"]
            ):
                list_of_psms.append(
                    PSM(
                        peptidoform=peprec_to_proforma(seq, mod),
                        spectrum_id=id,
                        retention_time=tr,
                    )
                )

        psm_list = PSMList(psm_list=list_of_psms)

    df_feat = pepper.do_f_extraction_psm_list_parallel(psm_list)
    df_feat["matrix"] = pd.Series(df_feat["matrix"])
    df_feat["matrix_sum"] = pd.Series(df_feat["matrix_sum"])
    df_feat["matrix_all"] = pd.Series(df_feat["matrix_all"])
    df_feat["pos_matrix"] = pd.Series(df_feat["pos_matrix"])
    df_feat["matrix_hc"] = pd.Series(df_feat["matrix_hc"])

    df_feat = pd.concat(
        [
            df_feat["matrix"],
            df_feat["matrix_sum"],
            df_feat["matrix_all"],
            df_feat["pos_matrix"],
            df_feat["matrix_hc"],
        ],
        axis=1,
    )
    df_feat.columns = ["matrix", "matrix_sum", "matrix_all", "pos_matrix", "matrix_hc"]
    df_feat.index = [psm.spectrum_id for psm in psm_list]
    df_feat["tr"] = [psm.retention_time for psm in psm_list]

    # df_feat = pd.DataFrame(df_feat,index=df.index)

    # df = pd.concat([df,df_feat],axis=1)

    return df_feat


def split_seq(a, n):
    """
    Split a list (a) into multiple chunks (n)

    Parameters
    ----------
    a : list
        list to split
    n : list
        number of chunks

    Returns
    -------
    list
        chunked list
    """

    # since chunking is not alway possible do the modulo of residues
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def cal_tr(uncal_preds, calibrate_dict, calibrate_min, calibrate_max, bin_dist=1):
    cal_preds = []

    for uncal_pred in uncal_preds:
        try:
            slope, intercept, x_correction = calibrate_dict[
                str(round(uncal_pred, bin_dist))
            ]
            cal_preds.append(slope * (uncal_pred - x_correction) + intercept)
        except KeyError:
            # outside of the prediction range ... use the last calibration curve
            if uncal_pred <= calibrate_min:
                slope, intercept, x_correction = calibrate_dict[
                    str(round(calibrate_min, bin_dist))
                ]
                cal_preds.append(slope * (uncal_pred - x_correction) + intercept)
            elif uncal_pred >= calibrate_max:
                slope, intercept, x_correction = calibrate_dict[
                    str(round(calibrate_max, bin_dist))
                ]
                cal_preds.append(slope * (uncal_pred - x_correction) + intercept)
            else:
                slope, intercept, x_correction = calibrate_dict[
                    str(round(calibrate_max, bin_dist))
                ]
                cal_preds.append(slope * (uncal_pred - x_correction) + intercept)
    return cal_preds


def calibrate_preds(
    tr_main,
    tr_sub,
    use_median=True,
    bin_dist=1,
    dict_cal_divider=100,
    split_cal=25,
    verbose=True,
):
    """
    Make calibration curve for predictions TODO make similar function for pd.DataFrame

    Parameters
    ----------
    seqs : list
        peptide sequence list; should correspond to mods and identifiers
    mods : list
        naming of the mods; should correspond to seqs and identifiers
    identifiers : list
        identifiers of the peptides; should correspond to seqs and mods
    measured_tr : list
        measured tr of the peptides; should correspond to seqs, identifiers, and mods

    Returns
    -------

    """

    calibrate_min = float("inf")
    calibrate_max = 0
    calibrate_dict = {}

    if verbose:
        t0 = time.time()

    # sort two lists, predicted and observed based on measured tr
    tr_sort = [
        (mtr, ptr)
        for mtr, ptr in sorted(zip(tr_main, tr_sub), key=lambda pair: pair[0])
    ]
    tr_main = [mtr for mtr, ptr in tr_sort]
    tr_sub = [ptr for mtr, ptr in tr_sort]

    mtr_mean = []
    ptr_mean = []

    # smooth between observed and predicted
    for mtr, ptr in zip(split_seq(tr_main, split_cal), split_seq(tr_sub, split_cal)):
        if not use_median:
            mtr_mean.append(sum(mtr) / len(mtr))
            ptr_mean.append(sum(ptr) / len(ptr))
        else:
            mtr_mean.append(np.median(mtr))
            ptr_mean.append(np.median(ptr))

    # calculate calibration curves
    for i in range(0, len(ptr_mean)):
        if i >= len(ptr_mean) - 1:
            continue
        delta_ptr = ptr_mean[i + 1] - ptr_mean[i]
        delta_mtr = mtr_mean[i + 1] - mtr_mean[i]

        slope = delta_mtr / delta_ptr
        intercept = mtr_mean[i]
        x_correction = ptr_mean[i]

        # optimized predictions using a dict to find calibration curve very fast
        for v in np.arange(
            round(ptr_mean[i], bin_dist),
            round(ptr_mean[i + 1], bin_dist),
            1 / ((bin_dist) * dict_cal_divider),
        ):
            if v < calibrate_min:
                calibrate_min = v
            if v > calibrate_max:
                calibrate_max = v
            calibrate_dict[str(round(v, 1))] = [slope, intercept, x_correction]

    if verbose:
        print("Time to calibrate: %s seconds" % (time.time() - t0))

    return calibrate_dict, calibrate_min, calibrate_max


def init_model(
    X_train,
    X_train_sum,
    X_train_global,
    X_bidirect,
    a_blocks=3,
    a_kernel=5,
    a_max_pool=2,
    a_filters_start=256,
    a_stride=1,
    b_blocks=3,
    b_kernel=5,
    b_max_pool=2,
    b_filters_start=256,
    b_stride=1,
    global_neurons=64,
    global_num_dens=4,
    regularizer_val=0.000005,
    num_gpus=1,
    verbose=True,
    fit_hc=False,
):
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    # lrelu = lambda x: tf.keras.activations.relu(x, negative_slope=0.1, max_value=20.0)
    lrelu = "leaky_relu"

    with strategy.scope():
        random.seed(42)

        initi = "RandomNormal"

        input_cnn = Input(shape=(X_train.shape[1], X_train.shape[2]))
        a = input_cnn
        for num_blocks in range(1, a_blocks + 1):
            a = Conv1D(
                filters=int(a_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=a_kernel,
                strides=a_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(a)
            a = Conv1D(
                filters=int(a_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=a_kernel,
                strides=a_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(a)
            if num_blocks < a_blocks:
                a = MaxPooling1D(pool_size=a_max_pool)(a)

        a = Flatten()(a)
        a = Model(inputs=input_cnn, outputs=a)

        input_cnn_sum = Input(shape=(X_train_sum.shape[1], X_train_sum.shape[2]))
        b = input_cnn_sum
        for num_blocks in range(1, b_blocks + 1):
            b = Conv1D(
                filters=int(b_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=b_kernel,
                strides=b_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(b)
            b = Conv1D(
                filters=int(b_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=b_kernel,
                strides=b_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(b)
            if num_blocks < b_blocks:
                b = MaxPooling1D(pool_size=2)(b)

        b = Flatten()(b)
        b = Model(inputs=input_cnn_sum, outputs=b)

        input_global = Input(shape=(X_train_global.shape[1],))
        c = input_global
        for num_dens in range(1, global_num_dens + 1):
            c = Dense(
                global_neurons,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
            )(c)
        c = Model(inputs=input_global, outputs=c)

        if fit_hc:
            input_bidirect = Input(shape=(X_bidirect.shape[1], X_bidirect.shape[2]))
            e = Conv1D(
                filters=2,
                kernel_size=2,
                strides=1,
                activation="tanh",
                kernel_regularizer=None,
                kernel_initializer=initi,
                padding="same",
            )(input_bidirect)
            e = Conv1D(
                filters=2,
                kernel_size=2,
                strides=1,
                activation="tanh",
                kernel_regularizer=None,
                kernel_initializer=initi,
                padding="same",
            )(e)
            e = MaxPooling1D(pool_size=10)(e)

            e = Flatten()(e)
            e = Model(inputs=input_bidirect, outputs=e)

        if fit_hc:
            combined = concatenate([a.output, b.output, c.output, e.output], axis=-1)
        else:
            combined = concatenate([a.output, b.output, c.output], axis=-1)

        d = Dense(
            128, activation=lrelu, kernel_regularizer=None, kernel_initializer=initi
        )(
            combined
        )  # l2(0.001)
        d = Dense(
            128, activation=lrelu, kernel_regularizer=None, kernel_initializer=initi
        )(d)
        d = Dense(
            128, activation=lrelu, kernel_regularizer=None, kernel_initializer=initi
        )(d)
        d = Dense(
            128, activation=lrelu, kernel_regularizer=None, kernel_initializer=initi
        )(d)
        d = Dense(
            128, activation=lrelu, kernel_regularizer=None, kernel_initializer=initi
        )(d)

        d = Dense(1, kernel_regularizer=None, kernel_initializer=initi)(d)

        if fit_hc:
            model = Model(inputs=[a.input, b.input, c.input, e.input], outputs=d)
        else:
            model = Model(inputs=[a.input, b.input, c.input], outputs=d)

        parallel_model = model
        parallel_model.compile(
            loss="mean_absolute_error",
            optimizer="Adam",
            metrics=["mean_absolute_error"],
        )

        if verbose:
            print(parallel_model.summary())

    return parallel_model


def init_model_new(
    X_train,
    X_train_sum,
    X_train_global,
    X_bidirect,
    a_blocks=3,
    a_kernel=5,
    a_max_pool=2,
    a_filters_start=256,
    a_stride=1,
    b_blocks=3,
    b_kernel=5,
    b_max_pool=2,
    b_filters_start=256,
    b_stride=1,
    global_neurons=64,
    global_num_dens=4,
    regularizer_val=0.000005,
    num_gpus=1,
    verbose=True,
    fit_hc=False,
):
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1, max_value=20.0)

    with strategy.scope():
        random.seed(42)

        initi = "RandomNormal"

        input_cnn = Input(shape=(X_train.shape[1], X_train.shape[2]))
        a = input_cnn
        for num_blocks in range(1, a_blocks + 1):
            a = Conv1D(
                filters=int(a_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=a_kernel,
                strides=a_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(a)
            a = Conv1D(
                filters=int(a_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=a_kernel,
                strides=a_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(a)
            if num_blocks < a_blocks:
                a = MaxPooling1D(pool_size=a_max_pool)(a)

        a = Flatten()(a)
        a = Model(inputs=input_cnn, outputs=a)

        input_cnn_sum = Input(shape=(X_train_sum.shape[1], X_train_sum.shape[2]))
        b = input_cnn_sum
        for num_blocks in range(1, b_blocks + 1):
            b = Conv1D(
                filters=int(b_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=b_kernel,
                strides=b_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(b)
            b = Conv1D(
                filters=int(b_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=b_kernel,
                strides=b_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(b)
            if num_blocks < b_blocks:
                b = MaxPooling1D(pool_size=2)(b)

        b = Flatten()(b)
        b = Model(inputs=input_cnn_sum, outputs=b)

        input_global = Input(shape=(X_train_global.shape[1],))
        c = input_global
        for num_dens in range(1, global_num_dens + 1):
            c = Dense(
                global_neurons,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
            )(c)
        c = Model(inputs=input_global, outputs=c)

        if fit_hc:
            input_bidirect = Input(shape=(X_bidirect.shape[1], X_bidirect.shape[2]))
            e = Conv1D(
                filters=2,
                kernel_size=2,
                strides=1,
                activation="tanh",
                kernel_regularizer=None,
                kernel_initializer=initi,
                padding="same",
            )(input_bidirect)
            e = Conv1D(
                filters=2,
                kernel_size=2,
                strides=1,
                activation="tanh",
                kernel_regularizer=None,
                kernel_initializer=initi,
                padding="same",
            )(e)
            e = MaxPooling1D(pool_size=10)(e)

            e = Flatten()(e)
            e = Model(inputs=input_bidirect, outputs=e)

        input_cnn_siam = Input(shape=(X_train.shape[1], X_train.shape[2]))
        a_siam = input_cnn_siam
        for num_blocks in range(1, a_blocks + 1):
            a_siam = Conv1D(
                filters=int(a_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=a_kernel,
                strides=a_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(a_siam)
            a_siam = Conv1D(
                filters=int(a_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=a_kernel,
                strides=a_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(a_siam)
            if num_blocks < a_blocks:
                a_siam = MaxPooling1D(pool_size=a_max_pool)(a_siam)

        a_siam = Flatten()(a_siam)
        a_siam = Model(inputs=input_cnn_siam, outputs=a_siam)

        input_cnn_sum = Input(shape=(X_train_sum.shape[1], X_train_sum.shape[2]))
        b_siam = input_cnn_sum
        for num_blocks in range(1, b_blocks + 1):
            b_siam = Conv1D(
                filters=int(b_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=b_kernel,
                strides=b_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(b_siam)
            b_siam = Conv1D(
                filters=int(b_filters_start / (2 ** (num_blocks - 1))),
                kernel_size=b_kernel,
                strides=b_stride,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
                padding="same",
            )(b_siam)
            if num_blocks < b_blocks:
                b_siam = MaxPooling1D(pool_size=2)(b_siam)

        b_siam = Flatten()(b_siam)
        b_siam = Model(inputs=input_cnn_sum, outputs=b_siam)

        input_global_siam = Input(shape=(X_train_global.shape[1],))
        c_siam = input_global_siam
        for num_dens in range(1, global_num_dens + 1):
            c_siam = Dense(
                global_neurons,
                activation=lrelu,
                kernel_regularizer=regularizers.l1(regularizer_val),
                kernel_initializer=initi,
            )(c_siam)
        c_siam = Model(inputs=input_global_siam, outputs=c_siam)

        if fit_hc:
            input_bidirect_siam = Input(
                shape=(X_bidirect.shape[1], X_bidirect.shape[2])
            )
            e_siam = Conv1D(
                filters=2,
                kernel_size=2,
                strides=1,
                activation="tanh",
                kernel_regularizer=None,
                kernel_initializer=initi,
                padding="same",
            )(input_bidirect_siam)
            e_siam = Conv1D(
                filters=2,
                kernel_size=2,
                strides=1,
                activation="tanh",
                kernel_regularizer=None,
                kernel_initializer=initi,
                padding="same",
            )(e_siam)
            e_siam = MaxPooling1D(pool_size=10)(e_siam)

            e_siam = Flatten()(e_siam)
            e_siam = Model(inputs=input_bidirect_siam, outputs=e_siam)

        if fit_hc:
            combined = concatenate(
                [
                    a.output,
                    b.output,
                    c.output,
                    e.output,
                    a_siam.output,
                    b_siam.output,
                    c_siam.output,
                    e_siam.output,
                ],
                axis=-1,
            )
        else:
            combined = concatenate([a.output, b.output, c.output], axis=-1)

        d = Dense(
            128, activation=lrelu, kernel_regularizer=None, kernel_initializer=initi
        )(
            combined
        )  # l2(0.001)
        d = Dense(
            128, activation=lrelu, kernel_regularizer=None, kernel_initializer=initi
        )(d)
        d = Dense(
            128, activation=lrelu, kernel_regularizer=None, kernel_initializer=initi
        )(d)
        d = Dense(
            128, activation=lrelu, kernel_regularizer=None, kernel_initializer=initi
        )(d)
        d = Dense(
            128, activation=lrelu, kernel_regularizer=None, kernel_initializer=initi
        )(d)

        d = Dense(1, kernel_regularizer=None, kernel_initializer=initi)(
            d
        )  # ,activation="sigmoid"

        if fit_hc:
            model = Model(
                inputs=[
                    a.input,
                    b.input,
                    c.input,
                    e.input,
                    a_siam.input,
                    b_siam.input,
                    c_siam.input,
                    e_siam.input,
                ],
                outputs=d,
            )
        else:
            model = Model(inputs=[a.input, b.input, c.input], outputs=d)

        parallel_model = model
        parallel_model.compile(
            loss="mean_absolute_error",
            optimizer="Adam",
            metrics=["mean_absolute_error"],
        )

        if verbose:
            print(parallel_model.summary())

    return parallel_model


def train_test(df, seed=42, ratio_train=0.9):
    random.seed(seed)

    unique_gene_ids = list(df.index)

    random.shuffle(unique_gene_ids)

    ids_train = list(set(unique_gene_ids[0 : int(len(unique_gene_ids) * ratio_train)]))
    ids_test = list(set(unique_gene_ids[int(len(unique_gene_ids) * ratio_train) + 1 :]))

    df_train = df.loc[ids_train]
    df_test = df.loc[ids_test]

    return df_train, df_test


def get_feat_matrix(df):
    X = np.stack(df["matrix"])
    X_sum = np.stack(df["matrix_sum"])

    X_global = np.concatenate(
        (np.stack(df["matrix_all"]), np.stack(df["pos_matrix"])), axis=1
    )
    X_hc = np.stack(df["matrix_hc"])

    return X, X_sum, X_global, X_hc, np.array(df["tr"])  # , np.array(df["first"])


def get_feat_matrix_new(df):
    X = np.stack(df["matrix"])
    X_sum = np.stack(df["matrix_sum"])
    X_global = np.concatenate(
        (np.stack(df["matrix_all"]), np.stack(df["pos_matrix"])), axis=1
    )
    X_hc = np.stack(df["matrix_hc"])

    try:
        return X, X_sum, X_global, X_hc, np.array(df[["first"]])  # ,"first"tr
    except:
        return X, X_sum, X_global, X_hc


def write_preds(
    df,
    X,
    X_sum,
    X_global,
    X_hc,
    mods,
    fit_hc=True,
    correction_factor=1.0,
    outfile_name="outfile.csv",
):
    df = df.copy()

    preds_test = []
    for mod in mods:
        if fit_hc:
            pred_test = (
                mod.predict([X, X_sum, X_global, X_hc]).flatten() * correction_factor
            )
        else:
            pred_test = mod.predict([X, X_sum, X_global]).flatten() * correction_factor
        preds_test.append(pred_test)
    pred_test = [float(sum(pred)) / len(pred) for pred in list(zip(*preds_test))]
    df["predictions"] = pred_test
    # df["tr"] = df["tr"]*correction_factor
    df.to_csv(outfile_name)

    return sum((df["tr"] - df["predictions"]).abs()) / len(df.index)


def plot_preds(
    X,
    X_sum,
    X_global,
    X_hc,
    y,
    mods,
    fit_hc=True,
    correction_factor=1.0,
    file_save="results.png",
    plot_title="Plot title",
):
    y = y * correction_factor

    try:
        preds_test = []
        for mod in mods:
            if fit_hc:
                pred_test = (
                    mod.predict([X, X_sum, X_global, X_hc]).flatten()
                    * correction_factor
                )
            else:
                pred_test = (
                    mod.predict([X, X_sum, X_global]).flatten() * correction_factor
                )
            preds_test.append(pred_test)
        pred_test = [float(sum(pred)) / len(pred) for pred in list(zip(*preds_test))]
    except:
        if fit_hc:
            pred_test = (
                mods.predict([X, X_sum, X_global, X_hc]).flatten() * correction_factor
            )
        else:
            pred_test = mods.predict([X, X_sum, X_global]).flatten() * correction_factor

    corr = np.corrcoef(y, pred_test)[0][1]
    mae = sum(abs(np.array(y) - pred_test) / len(pred_test))

    plt.figure(figsize=(10, 8))
    plt.scatter(pred_test, y, s=2)
    plt.xlabel("predicted tr")
    plt.ylabel("observed tr")
    plt.title("%s - mae: %s - R: %s" % (plot_title, round(mae, 3), round(corr, 4)))
    plt.plot([0, max(y)], [0, max(y)], c="grey")
    plt.savefig(file_save)
    plt.close()

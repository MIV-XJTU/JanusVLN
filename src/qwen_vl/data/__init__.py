import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

SPAR = {
    "annotation_path": "data/train/spar_7m.jsonl",
    "data_path": "data/media",
    "tag": "3d"
}

LLAVA_HOUND = {
    "annotation_path": "data/train/llava_hound_255k.json",
    "data_path": "data/media",
    "tag": "2d"
}

SCANNET_DET = {
    "annotation_path": "data/train/scannet_det_train_4frames.json",
    "data_path": "data/media",
    "tag": "3d"
}

SCANREFER = {
    "annotation_path": "data/train/scanrefer_train_32frames.json",
    "data_path": "data/media",
    "tag": "3d"
}

SCAN2CAP = {
    "annotation_path": "data/train/scan2cap_train_32frames.json",
    "data_path": "data/media",
    "tag": "3d"
}


# TRAIN_R2R_RxR = {
#     "annotation_path": "path-to-train_r2r_rxr.json",
#     "data_path": "",
#     "tag": "train_r2r_rxr"
# }


# TRAIN_R2R_RxR_EXTRA = {
#     "annotation_path": "path-to-train_r2r_rxr_extra.json",
#     "data_path": "",
#     "tag": "train_r2r_rxr_extra"
# }

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "spar": SPAR,
    "llava_hound": LLAVA_HOUND,
    "scannet_det": SCANNET_DET,
    "scanrefer": SCANREFER,
    "scan2cap": SCAN2CAP,
    # "train_r2r_rxr": TRAIN_R2R_RxR,
    # "train_r2r_rxr_extra": TRAIN_R2R_RxR_EXTRA
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config["dataset_name"] = dataset_name
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)

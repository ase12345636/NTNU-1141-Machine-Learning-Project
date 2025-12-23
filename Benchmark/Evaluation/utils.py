import os
import random
import numpy as np

import torch

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_gcp_env(args):
    # Only set environment variables when values are provided (avoid None)
    if getattr(args, 'GOOGLE_CLOUD_LOCATION', None) is not None:
        os.environ["GOOGLE_CLOUD_LOCATION"] = str(args.GOOGLE_CLOUD_LOCATION)
    if getattr(args, 'GOOGLE_GENAI_USE_VERTEXAI', None) is not None:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = str(args.GOOGLE_GENAI_USE_VERTEXAI)
    if getattr(args, 'TOKENIZERS_PARALLELISM', None) is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = str(args.TOKENIZERS_PARALLELISM)
    if getattr(args, 'GOOGLE_CLOUD_PROJECT', None) is not None:
        os.environ["GOOGLE_CLOUD_PROJECT"] = str(args.GOOGLE_CLOUD_PROJECT)


dx_task_measurement = ['rotation', 'projection', 'cardiomegaly',
                       'mediastinal_widening', 'carina_angle', 'aortic_knob_enlargement',
                       'descending_aorta_enlargement', 'descending_aorta_tortuous']

dx_task_multi_bodyparts = ['inspiration', 'rotation', 'projection', 'cardiomegaly',
                           'trachea_deviation', 'mediastinal_widening', "aortic_knob_enlargement",
                           "ascending_aorta_enlargement", "descending_aorta_enlargement"]
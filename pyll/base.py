import inspect
from abc import abstractmethod
from typing import Union

from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from pyll.config import Config
from pyll.utils.misc import import_object, invoke_functional_with_params


class TorchModel(nn.Module):
    @abstractmethod
    def loss(self, prediction, target):
        pass


class TorchDataset(Dataset):
    @property
    @abstractmethod
    def shape(self): pass

    @property
    @abstractmethod
    def num_classes(self): pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def invoke_dataset_from_config(config: Config, required: Union[str, list, tuple] = None):
    """
    Initializes datasets from config. Imports specified data reader and instantiates it with parameters from config.
    :param config: config
    :param required: string, list or tuple specifying which datasets have to be loaded (e.g. ["train", "val"])
    :return: initialized data readers
    """
    # Initialize Data Reader if specified
    readers = {}
    if config.has_value("dataset"):
        def to_list(value):
            if value is None:
                result = []
            elif isinstance(value, str):
                result = list([value])
            else:
                result = list(value)
            return result

        dataset = config.dataset
        required = to_list(required)

        try:
            reader_class = import_object(dataset["reader"])
            reader_args = inspect.signature(reader_class).parameters.keys()
            datasets = [key for key in dataset.keys() if key not in reader_args and key != "reader"]
            global_args = [key for key in dataset.keys() if key not in datasets and key != "reader"]

            # check for required datasets before loading anything
            if required is not None:
                required = to_list(required)
                missing = [d for d in required if d not in datasets]
                if len(missing) > 0:
                    raise Exception("Missing required dataset(s) {}".format(missing))

            # read "global" parameters
            global_pars = {}
            for key in global_args:
                value = dataset[key]
                global_pars[key] = value
                if isinstance(value, str) and "import::" in value:
                    global_pars[key] = import_object(value[len("import::"):])
                if key == "transforms":
                    global_pars[key] = Compose([invoke_functional_with_params(t) for t in value])

            # read dataset specific parameters
            for dset in datasets:
                # inspect parameters and resolve if necessary
                for key, value in dataset[dset].items():
                    if isinstance(value, str) and "import::" in value:
                        dataset[dset][key] = import_object(value[len("import::"):])
                    if key == "transforms":
                        dataset[dset][key] = Compose([invoke_functional_with_params(t) for t in value])
                print("Loading dataset '{}'...".format(dset))
                readers[dset] = reader_class(**{**global_pars, **dataset[dset]})
        except (AttributeError, TypeError) as e:
            print("Unable to import '{}'".format(e))
            raise e
    return readers


def invoke_model_from_config(config: Config, dataset: TorchDataset = None):
    model = config.import_architecture()
    model_params = inspect.signature(model).parameters.keys()
    model_args = {}
    if "num_classes" in model_params and hasattr(dataset, "num_classes"):
        model_args["num_classes"] = dataset.num_classes
    if "input_shape" in model_params and hasattr(dataset, "shape"):
        model_args["input_shape"] = dataset.shape
    if "model_params" in model_params:
        model_args["model_params"] = config.get_value("model_params", None)
    elif config.get_value("model_params", None) is not None:
        model_args.update(config.get_value("model_params", None))
    return model(**model_args)


def calc_conv2d_out_shape(input, ksize, padding, stride, dilation):
    return [int(((input[i] + 2 * padding[i] - (ksize[i] - 1) * dilation[i] - 1) / stride[i]) + 1) for i in range(0, len(input))]


def calc_out_shape(in_h, in_w, layers):
    relevant = []
    for l in layers:
        k, s, p, d = None, None, None, (1, 1)
        if hasattr(l, "kernel_size"):
            if type(l.kernel_size) is int:
                k = (l.kernel_size, l.kernel_size)
            else:
                k = l.kernel_size
        if hasattr(l, "stride"):
            if type(l.stride) is int:
                s = (l.stride, l.stride)
            else:
                s = l.stride
        if hasattr(l, "padding"):
            if type(l.padding) is int:
                p = (l.padding, l.padding)
            else:
                p = l.padding
        if hasattr(l, "dilation"):
            if type(l.dilation) is int:
                d = (l.dilation, l.dilation)
            else:
                d = l.dilation
        if k is not None:
            relevant.append([k, s, p, d])
    h, w = in_h, in_w
    for l in relevant:
        h, w = calc_conv2d_out_shape([h, w], ksize=l[0], padding=l[2], stride=l[1], dilation=l[3])
    return h, w

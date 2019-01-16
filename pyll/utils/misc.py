# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017
"""
"""dev_layers.py: Unfinished layer classes


Author -- Michael Widrich
Created on -- Wed Oct 19 10:30:51 2016
Contact -- michael.widrich@jku.at

Unfinished layer classes


=======  ==========  =================  ================================
Version  Date        Author             Description
0.1      2016-10-15  Michael Widrich    Added comments and revised
=======  ==========  =================  ================================

"""
import argparse
import errno
import importlib
import os
import tempfile
import zipfile
from pathlib import Path
from shutil import copyfile


# ----------------------------------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------------------------------

class AbortRun(Exception):
    pass


def check_kill_file(workspace):
    """raise an AbortRun error if the kill file exists"""
    if os.path.isfile(workspace.get_kill_file()):
        print("Detected kill file, aborting...")
        raise AbortRun


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="JSON file with the model params")
    args, override_args = parser.parse_known_args()
    if len(override_args) == 0:
        override_args = None
    return args, override_args


def touch(path):
    Path(path).touch()


def chmod(path: str, permissions=0o775, recursive=False):
    if recursive:
        for root, dirs, files in os.walk(path):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), permissions)
                except PermissionError:
                    continue
            for f in files:
                try:
                    os.chmod(os.path.join(root, f), permissions)
                except PermissionError:
                    continue
    else:
        os.chmod(path, permissions)


def copydir(src: str, dst: str, exclude: list = None):
    if os.path.isdir(dst):
        rmdir(dst)
    for root, dirs, files in os.walk(src):
        for file in files:
            f = os.path.realpath(os.path.join(root, file))
            # don't add files / folders in exclude
            exclude_file = False
            if exclude:
                for entry in exclude:
                    if os.path.realpath(entry) in f:
                        exclude_file = True
                        break

            if not exclude_file:
                d = f.replace(os.path.realpath(src), os.path.realpath(dst))
                if not os.path.exists(os.path.dirname(d)):
                    os.makedirs(os.path.dirname(d))
                copyfile(f, d)


def rmdir(path: str):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for d in dirs:
                current = os.path.join(root, d)
                try:
                    rmdir(current)
                except PermissionError:
                    continue
            for f in files:
                try:
                    os.remove(os.path.join(root, f))
                except PermissionError:
                    continue
        os.rmdir(path)


def zipdir(dir, zip, info=None, exclude=None):
    # canonicalize paths
    dir = os.path.realpath(dir)
    zip = os.path.realpath(zip)
    # open zipfile
    zipf = zipfile.ZipFile(zip, 'w', zipfile.ZIP_DEFLATED)

    if info:
        zipf.writestr("00-INFO", info)

    # add dir
    for root, dirs, files in os.walk(dir):
        for file in files:
            f = os.path.realpath(os.path.join(root, file))
            # don't add zip archive to itself
            if zip == os.path.join(root, file) or "__pycache__" in f:
                continue
            # don't add files / folders in exclude
            exclude_file = False
            if exclude:
                for entry in exclude:
                    if os.path.realpath(entry) in f:
                        exclude_file = True
                        break

            if not exclude_file:
                zipf.write(filename=f, arcname=f[len(dir):])
    zipf.close()


def import_object(objname):
    module_str = objname.split('.', maxsplit=1)
    # small hack for more intuitive import of tf modules
    if module_str[0] == "tf":
        module_str[0] = "tensorflow"
    objmodule = importlib.import_module(module_str[0])
    return get_rec_attr(objmodule, module_str[-1])


def invoke_functional_with_params(method_call):
    mname, params = method_call[:method_call.index("(")], method_call[method_call.index("(") + 1:-1]
    method = import_object(mname)
    if len(params) > 0:
        params = eval(params)
        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = (params,)
        return method(*params)
    else:
        return method()


def get_rec_attr(obj, attrstr):
    """Get attributes and do so recursively if needed"""
    if attrstr is None:
        return None
    if "." in attrstr:
        attrs = attrstr.split('.', maxsplit=1)
        if hasattr(obj, attrs[0]):
            obj = get_rec_attr(getattr(obj, attrs[0]), attrs[1])
        else:
            try:
                obj = get_rec_attr(importlib.import_module(obj.__name__ + "." + attrs[0]), attrs[1])
            except ImportError:
                raise
    else:
        if hasattr(obj, attrstr):
            obj = getattr(obj, attrstr)
    return obj


def load_architecture(arch_name: str = None):
    """Import an architecture defined as a class in a file in the current namespace.
    
    :param arch_name
        name of architecture to load; format: <filename>.<classname>
                
    :returns architecture handle or none if not found
    """
    architecture = importlib.import_module(arch_name.split('.', maxsplit=1)[0])
    return get_rec_attr(architecture, arch_name.split('.', maxsplit=1)[-1])


def extract_to_tmp(file: str):
    tempdir = tempfile.mkdtemp("tell")
    ziphandle = zipfile.ZipFile(file, "r")
    ziphandle.extractall(tempdir)
    ziphandle.close()
    return tempdir


def get_tensor_shape_list(tensor):
    """get shape of a tf tensor as list of integers usable as conventional shape"""
    return [d if isinstance(d, int) else -1 for d in tensor.shape.as_list()]


def load_files_in_dir(directory: str, suffix: str = ''):
    """Search for all files in sample_directory (optionally) with suffix sample_suffix an return them as pandas
    dataframe
    
    Parameters
    -------
    directory : str
        Directory to search for files in
    suffix : str
        If a string is provided as sample_suffix, the file names without this suffix will be ignored

    Returns
    -------
    : pd.DataFrame
        Pandas dataframe with base filenames (=filenames without path or suffix) as keys and full filenames as values;
        Dataframe is sorted by full filenames;
    """
    from os import path
    import glob.glob
    import pandas as pd

    sample_pattern = "**/*{}".format(suffix)

    # Collect files in path, sort them by name, and store them into dictionary
    samples = glob.glob(path.join(directory, sample_pattern))
    samples.sort()

    # Extract base filenames without path or suffix and store them as keys for the pandas dataframe
    keys = [path.basename(file)[:-len(suffix)] for file in samples]

    # Store in data frame for easy indexing and fast key->value access
    samples = pd.DataFrame(index=keys, data=samples)

    return samples


def extract_named_args(arglist):
    result = {}
    for i in range(0, len(arglist)):
        a = arglist[i]
        if a.startswith("--"):
            if i + 1 < len(arglist) and not arglist[i + 1].startswith("--"):
                result[a] = arglist[i + 1]
            else:
                result[a] = None
    return result


def extract_unnamed_args(arglist):
    result = []
    for i in range(1, len(arglist)):
        a = arglist[i]
        p = arglist[i - 1]
        if not a.startswith("--") and not p.startswith("--"):
            result.append(a)
    return result


def try_to_number_or_bool(value):
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            if isinstance(value, str) and (value.lower() == "false" or value.lower() == "true"):
                value = (value.lower() == "true")
            pass
    return value


class Tee(object):
    """Created from snippets on stackoverflow"""

    def __init__(self, original_stdout, *files):
        self.files = files
        self.original_stdout = original_stdout

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        try:
            self.original_stdout.flush()
        except AttributeError:
            # if original_stdout does not support flush()
            pass


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    # TODO: move this little helper to an appropriate place
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def get_value(self, name, default=None):
        if name in self:
            return self[name]
        else:
            return default

    def has_value(self, name):
        return name in self

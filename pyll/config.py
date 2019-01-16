# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

Default configuration settings

"""
import json
import os

from pyll.utils.misc import import_object, extract_named_args, try_to_number_or_bool, parse_args, dotdict


class Config(object):
    def __init__(self, filename: str = None):
        """Create config object from json file.
        
        filename : optional;
            If passed read config from specified file, otherwise parse command line for config parameter and optionally
            override arguments.
        """
        if filename is None:
            args, override_args = parse_args()
            config_file = args.config
        else:
            args = None
            override_args = None
            config_file = filename
        
        # Read config and override with args if passed
        if os.path.exists(config_file):
            with open(config_file) as f:
                self.initialize_from_json(json.loads(f.read()).items())
            # override if necessary
            if override_args is not None:
                self.override_from_commandline(override_args)
        else:
            raise Exception("Configuration file does not exist!")
    
    def override(self, name, value):
        if value is not None:
            value = self.__import_value_rec__(value)
            setattr(self, name, value)
            print("CONFIG: {}={}".format(name, getattr(self, name)))
    
    set_value = override
    
    def __import_value_rec__(self, value):
        if isinstance(value, str) and "import::" in value:
            try:
                value = import_object(value[len("import::"):])
            except ModuleNotFoundError:
                print("ERROR WHEN IMPORTING '{}'".format(value))
        elif isinstance(value, dict):
            for k, v in value.items():
                value[k] = self.__import_value_rec__(v)
            value = dotdict(value)
        
        return value
    
    def has_value(self, name):
        return hasattr(self, name)
    
    def get_value(self, name, default=None):
        return getattr(self, name, default)
    
    def import_architecture(self):
        if hasattr(self, "model"):
            return import_object(self.model)
        else:
            return None
    
    def initialize_from_json(self, nv_pairs=None):
        if nv_pairs:
            for i, (name, value) in enumerate(nv_pairs):
                self.override(name, value)
    
    def override_from_commandline(self, override_args=None):
        if override_args is not None:
            override = extract_named_args(override_args)
            for k, v in override.items():
                name = k[2:] if "--" in k else k  # remove leading --
                value = v if v.startswith('"') or v.startswith("'") else try_to_number_or_bool(v)
                if "." in name:
                    names = name.split(".")
                    name = names[0]
                    if len(names) == 2:
                        if hasattr(self, names[0]):
                            curdict = getattr(self, names[0])
                        else:
                            curdict = dict()
                        curdict[names[1]] = value
                        value = curdict
                    elif len(names) == 3:
                        if hasattr(self, names[0]):
                            curdict = getattr(self, names[0])
                        else:
                            curdict = dict()
                        
                        if names[1] in curdict:
                            subdict = curdict[names[1]]
                        else:
                            curdict[names[1]] = dict()
                            subdict = curdict[names[1]]
                        
                        subdict[names[2]] = value
                        value = curdict
                    else:
                        raise Exception("Unsupported command line option (can only override dicts with 1 or 2 levels)")
                self.override(name, value)

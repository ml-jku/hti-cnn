# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

"""
import datetime as dt
import os
import sys
import tempfile
from os import path

import __main__

from pyll.utils.misc import make_sure_path_exists, zipdir, chmod, copydir, rmdir


class Workspace(object):
    def __init__(self, workspace: str = None, name: str = None, comment: str = None, resume: str = None):
        """

        :param workspace: str
            path to general workspace directory
        :param specs: str
            short description of specs for a specific test run;
            will be part of the run specific working directory so don't use spaces or special chars
        """
        self.name = name
        self.comment = comment
        
        if resume is None:
            self.workspace, self.timestamp, self.results, self.statistics, self.checkpoints, self.kill_file = self.__setup_working_dir__(workspace)
        else:
            self.workspace, self.timestamp, self.results, self.statistics, self.checkpoints, self.kill_file = self.__resume_from_dir__(resume)
    
    @property
    def workspace_dir(self):
        return self.workspace
    
    @property
    def results_dir(self):
        return self.results
    
    @property
    def statistics_dir(self):
        return self.statistics
    
    @property
    def has_kill_file(self):
        return path.exists(self.kill_file)
    
    @property
    def get_timestamp(self):
        return self.timestamp
    
    @property
    def checkpoint_dir(self):
        return self.checkpoints
    
    def __setup_working_dir__(self, workspace_root):
        # fix permissions of workspace root
        make_sure_path_exists(workspace_root)
        try:
            chmod(workspace_root, 0o775)
        except PermissionError:
            print("PermissionError when trying to change permissions of workspace to 775")
        
        # setup working directory
        experiment_dir = path.realpath("{}/{}".format(workspace_root, self.name))
        timestamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = path.realpath("{}/{}".format(experiment_dir, timestamp))
        # Set up artifact folders
        results_dir = make_sure_path_exists("{}/results".format(run_dir))
        statistics_dir = make_sure_path_exists("{}/statistics".format(run_dir))
        checkpoints_dir = make_sure_path_exists("{}/checkpoints".format(run_dir))
        
        # set path to kill file (if this file exists abort run)
        kill_file_name = "ABORT_RUN"
        kill_file = path.join(run_dir, kill_file_name)
        
        # fix permissions to grant group write access (to allow kill_file creation and plot control)
        try:
            # chmod(self.workspace, 0o775, recursive=False)
            chmod(experiment_dir, 0o775, recursive=False)
            chmod(run_dir, 0o775, recursive=True)
        except PermissionError:
            print("PermissionError when trying to change permissions of workspace to 775")
        
        # compress and copy current script and dependencies to results dir
        self.__zip_source__(run_dir)
        
        # return paths
        return [run_dir, timestamp, results_dir, statistics_dir, checkpoints_dir, kill_file]
    
    def __resume_from_dir__(self, dir):
        if "checkpoints" not in dir:
            print("invalid resume path")
            exit(1)
        
        # setup working directory
        run_dir = path.dirname(path.dirname(path.realpath(dir)))
        self.name = path.basename(path.dirname(run_dir))
        timestamp = path.basename(run_dir)
        
        # Set up result folder structure
        results_dir = "{}/results".format(run_dir)
        statistics_dir = "{}/statistics".format(run_dir)
        checkpoints_dir = "{}/checkpoints".format(run_dir)
        
        # clear kill file if necessary
        kill_file_name = "ABORT_RUN"
        kill_file = path.join(run_dir, kill_file_name)
        if path.exists(kill_file):
            os.remove(kill_file)
        
        if not path.exists(results_dir) or not path.exists(statistics_dir) or not path.exists(checkpoints_dir):
            raise Exception("can not resume from given directory")
        
        # compress and copy current script and dependencies to results dir
        self.__zip_source__(run_dir)
        
        return [run_dir, timestamp, results_dir, statistics_dir, checkpoints_dir, kill_file]
    
    def __zip_source__(self, run_dir):
        # on resume create new zip file
        zip_file = path.join(run_dir, '00-script.zip')
        if path.exists(zip_file):
            i = 1
            zip_file = path.join(run_dir, '00-script-{0:02d}.zip'.format(i))
            while path.exists(zip_file):
                i += 1
                zip_file = path.join(run_dir, '00-script-{0:02d}.zip'.format(i))
        
        # compress and copy current script and dependencies to results dir
        command = " ".join(sys.argv)
        # copy current code to temp dir
        script_dir = path.dirname(path.realpath(__main__.__file__))
        tempdir = tempfile.mkdtemp("pyll")
        copydir(script_dir, tempdir,
                exclude=[run_dir, path.join(script_dir, ".git"), path.join(script_dir, ".idea"),
                         path.join(script_dir, "__pycache__")])
        # also copy currently used TeLL library so it can be used for resuming runs
        # copydir(TeLL.__path__[0], path.join(tempdir, path.basename(TeLL.__path__[0])))
        # rmdir(path.join(path.join(tempdir, path.basename(TeLL.__path__[0])), "__pycache__"))
        
        zipdir(dir=tempdir, zip=zip_file, info=command, exclude=[run_dir, '.git'])
        rmdir(tempdir)

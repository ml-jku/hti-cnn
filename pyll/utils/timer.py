# -*- coding: utf-8 -*-
"""
© Michael Widrich, Markus Hofmarcher, 2017

"""
import time


class Timer(object):
    def __init__(self, name="", verbose=True, precision='msec'):
        self.verbose = verbose
        self.name = name
        self.precision = precision
        self.restart()

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        if self.verbose:
            self.print()

    def restart(self):
        self.begin = time.time()
        self.end = 0
        self.secs = 0
        self.msecs = 0

    def stop(self):
        self.end = time.time()
        self.secs = self.end - self.begin
        self.msecs = self.secs * 1000  # millisecs
        return self.msecs

    def print(self):
        if self.end == 0:
            self.stop()
        if self.precision == 'msec':
            print('Timer ({0}): {1:.2f} ms'.format(self.name, self.msecs))
        else:
            print('Timer ({0}): {1:.3f} s'.format(self.name, self.secs))

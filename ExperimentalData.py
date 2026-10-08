import pandas as pd
import numpy as np

class ExperimentalData:

    def __init__(self, nameFile):

        self.data = self.read(nameFile)


    def read(self, nameFile):
        pass

    def save(self, nameFile=None):
        pass

    def plot(self):
        pass

    def approximations(self):
        pass

    def regressions(self):
        pass

    def smooth(self):
        pass

    
from ..node import Pipe
import numpy as np

class SimpleEnsembler(Pipe):
    def __init__(self, label: str) -> None:
        super().__init__(label)

    def pipe(self, inputs):
        # return "Coming Soon!"

        return np.array([inputs["Mod1"], inputs["Mod2"], inputs["Mod3"], inputs["Mod4"], inputs["Mod5"]])
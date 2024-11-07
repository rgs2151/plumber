from ..node import Pipe

class SimpleEnsembler(Pipe):
    def __init__(self, label: str) -> None:
        super().__init__(label)

    def pipe(self, inputs):
        return "Coming Soon!"
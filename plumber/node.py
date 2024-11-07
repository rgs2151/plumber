from collections import defaultdict

class ABCNamed():
    
    def __init__(self, label: str) -> None:
        self.label = str(label)

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ABCNamed) and self.label == other.label

    def __hash__(self) -> int:
        return hash(self.label)
    
class Node(ABCNamed):
    
    def __init__(self, label: str) -> None:
        super().__init__(label)

class Stream(ABCNamed):
    data = defaultdict(None)
    def __init__(self, label: str) -> None: super().__init__(label)

class Pipe(Node):

    def __init__(self, label: str) -> None:
        super().__init__(label)

    def pipe(self):
        raise NotImplementedError('pipe method not implemented')



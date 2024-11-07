from .graph import Graph
from .node import Node, Pipe, Stream

class PipeLine(Graph):

    def __init__(self):
        super().__init__()

    def flow(self, input_stream: Stream):
        load_order = self.topological_sort()
        for pipe in load_order:
            dependencies = self.reverse_graph[pipe]
            input_data = {str(dep): input_stream.data[dep.label] for dep in dependencies}
            if isinstance(pipe, Pipe):
                input_stream.data[pipe.label] = pipe.pipe(input_data)
            # print(f"{pipe.label}: {input_stream.data[pipe.label]}")

    def register(self, pipe1: Node, pipe2: Node):
        self.add_edge(pipe1, pipe2)
    
    def sequential_connect(self, pipes: list) -> Pipe:
        previous = pipes[0]
        for pipe in pipes[1:]:
            self.register(previous, pipe)
            previous = pipe
        return previous
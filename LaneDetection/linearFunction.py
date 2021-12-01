
class LinearFunction(object):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def getValue(self, x: float):
        return self.a * x + self.b

    def getArgument(self, y: float):
        return (y - self.b) / self.a

    def __repr__(self):
        return f'y = {self.a}x + {self.b}'
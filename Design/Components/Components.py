

class Component(object):
    def __init__(self):
        self.weight = 0.0
        self.cost = 0.0
        # Type is inhertited to upper level components and handled there
        self.type = ''

    def log(self,logger):
        logger.info('Component: %s\r\tWeight: %s\r\tCost: %s\r\n' % (self.type, self.weight, self.cost))


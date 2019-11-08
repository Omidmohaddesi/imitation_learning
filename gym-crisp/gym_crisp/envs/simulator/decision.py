""" decision provide definitions for various types of decisions """

class Decision(object):
    pass


class ProduceDecision(Decision):
    def __init__(self):
        self.amount = int(0)


class AllocateDecision(Decision):
    def __init__(self):
        self.item = None
        self.downstream_node = None
        self.order = None
        self.amount = int(0)


class OrderDecision(Decision):
    def __init__(self):
        self.upstream = None
        self.amount = int(0)


class TreatDecision(Decision):
    def __init__(self):
        self.urgent = int(0)
        self.non_urgent = int(0)

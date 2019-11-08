import numpy as np

class Network(object):
    def __init__(self, numAgent):
        self.num_agent = numAgent
        self.connectivity = np.full((numAgent, numAgent), -1, dtype=np.int)
        self.payloads = []

    def to_json(self):
        s = '{'
        s += '\n\t"num_agents":' + str(self.num_agent) + ','

        s += '\n\t"connectivity_matrix": ['
        first = True
        for x in np.nditer(self.connectivity):
            if not first:
                s += ','
            first = False
            s += str(x)
        s += '],'

        s += '\n\t"payload":['
        first = True
        for p in self.payloads:
            if not first:
                s += ','
            first = False
            s += '\n\t\t' + p.to_json()
        s += ']\n'

        s += '}'
        return s


class NetworkPayload(object):
    def __init__(self):
        self.src = None
        self.dst = None
        self.sendTime = 0
        self.leadTime = 0


class InTransit(NetworkPayload):
    def __init__(self, item):
        super(InTransit, self).__init__()
        self.item = item

    def to_json(self):
        s = '{'
        s += '"src":' + str(self.src.id) + ','
        s += '"dst":' + str(self.dst.id) + ','
        s += '"leadTime":' + str(self.leadTime) + ','
        s += '"item":' + str(self.item.to_json())
        s += '}'
        return s


class OrderMessage(NetworkPayload):
    def __init__(self, order):
        super(OrderMessage, self).__init__()
        self.order = order

    def to_json(self):
        s = '{'
        s += '"src":' + str(self.src.id) + ','
        s += '"dst":' + str(self.dst.id) + ','
        s += '"leadTime":' + str(self.leadTime)
        s += '}'
        return s


import itertools


class Order(object):
    """ Order is a message sent from an downstream node to upstream node to
        buy some drug
    """

    new_id = itertools.count().next

    def __init__(self):
        self.id = Order.new_id()

        self.src = None
        self.dst = None
        self.amount = int(0)
        self.place_time = 0

        self.delivery = []

        self.recv_time = 0
        self.exp_recv_time = 0
        self.expire_time = 0

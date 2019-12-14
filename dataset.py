import pandas as pd
import os


class CrispDataset:
    def __init__(self, path):
        self.order = pd.DataFrame()
        self.inventory = pd.DataFrame()
        self.demand = pd.DataFrame()
        self.backlog = pd.DataFrame()
        self.shipment = pd.DataFrame()
        self.cost = pd.DataFrame()

        self.read_from_csv(path)

    def read_from_csv(self, expert_data_path):
        self.order = pd.read_csv(os.path.join(expert_data_path, 'order_data.csv'), index_col=0)
        self.inventory = pd.read_csv(os.path.join(expert_data_path, 'inventory_data.csv'), index_col=0)
        self.demand = pd.read_csv(os.path.join(expert_data_path, 'demand_data.csv'), index_col=0)
        self.backlog = pd.read_csv(os.path.join(expert_data_path, 'backlog_data.csv'), index_col=0)
        self.shipment = pd.read_csv(os.path.join(expert_data_path, 'shipments_data.csv'), index_col=0)
        self.cost = pd.read_csv(os.path.join(expert_data_path, 'cost_data.csv'), index_col=0)

        self.order = self.order.reset_index(drop=True)
        self.inventory = self.inventory.reset_index(drop=True)
        self.demand = self.demand.reset_index(drop=True)
        self.backlog = self.backlog.reset_index(drop=True)
        self.shipment = self.shipment.reset_index(drop=True)
        self.cost = self.cost.reset_index(drop=True)

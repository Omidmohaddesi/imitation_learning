""" Disruption provides the basic types of disruption to happen in the simulation """
from simulation import Simulation


class Disruption(object):
    """ Disruption abstract any type of disruptions to the simulation """

    def happen(self, now):
        """ what happens with the disruption"""
        pass


class LineShutDownDisruption(Disruption):
    def __init__(self, simulation, num_active_lines):
        """
        :param Simulation simulation: the simulation where this disruption is going to happen.
        :param int num_active_lines: the number of active lines in the manufacturer without a disruption.
        """

        self.simulation = simulation
        self.num_active_lines = num_active_lines

        self.happen_day_1 = 20
        self.end_day_1 = 30
        self.decrease_factor_1 = 0.625

        self.happen_day_2 = -1  # 45
        self.end_day_2 = -1  # 50
        self.decrease_factor_2 = 0

        self.manufacturer_id = 1

    def happen(self, now):
        if now < self.happen_day_1:
            return
        elif now <= self.end_day_1:
            self.simulation.manufacturers[self.manufacturer_id].num_active_lines = \
                int(self.num_active_lines * (1 - self.decrease_factor_1))
        elif now < self.happen_day_2:
            self.simulation.manufacturers[self.manufacturer_id].num_active_lines = self.num_active_lines
        elif now <= self.end_day_2:
            self.simulation.manufacturers[self.manufacturer_id].num_active_lines = \
                int(self.num_active_lines * (1 - self.decrease_factor_1))
        else:
            self.simulation.manufacturers[self.manufacturer_id].num_active_lines = self.num_active_lines


class RecallDisruption(Disruption):

    def __init__(self, simulation):
        self.simulation = simulation
        self.happen_day = 30
        self.defect_day = [25, 26, 27, 28, 29]
        self.defect_line = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # check the capacity of each line
        self.manufacturer_id = 1

    def happen(self, now):
        if now < self.happen_day:
            return

        # print "now: ", now
        hc1 = self.simulation.health_centers[0]
        # print "before: ", hc1.inventory_level()

        batch_numbers = []
        for day in self.defect_day:
            for line in self.defect_line:
                batch_number = str(self.manufacturer_id) + \
                               '_' + str(day) + '_' + str(line)
                batch_numbers.append(batch_number)

        for batch in batch_numbers:
            for hc in self.simulation.health_centers:
                hc.inventory[:] = [
                    i for i in hc.inventory if i.batch_no != batch]

            for ds in self.simulation.distributors:
                ds.inventory[:] = [
                    i for i in ds.inventory if i.batch_no != batch]

            for mn in self.simulation.manufacturers:
                mn.inventory[:] = [
                    i for i in mn.inventory if i.batch_no != batch]
                # for item in mn.inventory:
                # print item.batch_no, batch

        # print "after: ", hc1.inventory_level()
        return


class DemandChangeDisruption(Disruption):

    def __init__(self, patient_models):
        self.patient_models = patient_models
        self.start_time = 30
        self.end_time = 40
        self.change = 10

    def happen(self, now):
        if now == self.start_time:
            for pm in self.patient_models:
                pm.urgent += self.change
                # pm.non_urgent += self.change
        elif now == self.end_time:
            for pm in self.patient_models:
                pm.urgent -= self.change
                # pm.non_urgent -= self.change

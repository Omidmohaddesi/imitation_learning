''' PatientModel provides algorithms to generate patient '''

import numpy as np
import time

class PatientModel(object):
    ''' PatientModel defines the interface of all Patient Models. '''

    def generate_patient(self, now):
        ''' Update the number of patients of each health center. '''
        pass


class NormalDistPatientModel(PatientModel):

    def __init__(self, health_centers):
        np.random.seed(0)
        self.health_centers = health_centers
        self.urgent_mean = 20
        self.urgent_stdev = 10
        self.non_urgent_mean = 100
        self.non_urgent_stdev = 10

    def generate_patient(self, now):
        for hc in self.health_centers:
            urgent = np.random.normal(self.urgent_mean, self.urgent_stdev, 1)[0]
            if urgent < 0:
                urgent = 0
            urgent = round(urgent)

            non_urgent = np.random.normal(self.non_urgent_mean,
                                          self.non_urgent_stdev, 1)[0]
            if non_urgent < 0:
                non_urgent = 0
            non_urgent = round(non_urgent)

            hc.receive_patient(urgent, non_urgent, now)


class UniformDistPatientModel(PatientModel):

    def generate_patient(self, now):
        pass


class ConstantPatientModel(PatientModel):

    def __init__(self, health_centers):
        self.urgent = 20
        self.non_urgent = 100
        self.health_centers = health_centers

    def generate_patient(self, now):
        for hc in self.health_centers:
            hc.receive_patient(self.urgent, self.non_urgent, now)


class AccumulatingConstantPatientModel(PatientModel):

    def __init__(self, health_centers):
        self.urgent = 20
        self.non_urgent = 100
        self.health_centers = health_centers

    def generate_patient(self, now):
        for hc in self.health_centers:
            hc.receive_patient(
                hc.urgent + self.urgent,
                hc.non_urgent + self.non_urgent,
                now)


### Added this for BeerGame demand. The non-urgent should be 40 for 4 weeks and then 80 and the urgent should be zero
class end_customer_demand(PatientModel):

    def __init__(self, health_centers):
        self.urgent = 0

        self.health_centers = health_centers

    def generate_patient(self, now):
        if now < 5:
            self.non_urgent = 40
        else:
            self.non_urgent = 80

        for hc in self.health_centers:
            hc.receive_patient(self.urgent, self.non_urgent, now)


class PredefinedPatientModel(PatientModel):

    def __init__(self, health_centers):
        self.health_centers = health_centers

    def generate_patient(self, now):
        for hc in self.health_centers:
            if now < 140:
                hc.receive_patient(10, 50, now)
            else:
                hc.receive_patient(20, 100, now)

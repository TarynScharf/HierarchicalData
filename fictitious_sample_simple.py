import math
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import scipy.stats
from testing_framework import AbstractSampler

@dataclass
class Observation:
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@dataclass
class InitialConditions:
    '''
    Objects of this class will hold the 1 hypothetical initial condition that defines a sample.
    '''
    variable1: float


class FictitiousSamplerSimple(AbstractSampler[InitialConditions, Observation]):
    intraclass_variability: float
    interclass_variability: float

    def __init__(self, intraclass_variability,interclass_variability):
        self.intraclass_variability = intraclass_variability
        self.interclass_variability = interclass_variability

    def _sample_the_initial_conditions(self, number_of_samples: int) -> Iterable[InitialConditions]:
        '''
        This function simulates the "inital conditions" that lead to a geological entity. E.g. The initial magmatic conditions such as temperature, pressure, oxygen fugacity that result in a pluton.
        (NB This does not attempt to directly model these magmatic parameters but to simulate a system - in the general sense - that is dependent on multiple parameters!)
        Three variable distributions represent the spread of values for these conditions, in the same way that e.g. temp, pressure, oxygen fugacity exist on a spectrum in the natural world.
        This function generates a collection of initial conditions that result in a geological entity (e.g. magmatic conditions) by randomly sampling from these distributions (i.e. randomly combining T, P, Of)
        It thus returns a sequence of randomly selected initial conditions, which correspond to "entities" in the geological sense.
        '''
        variable1_skewness = 5
        variable1_scale = self.interclass_variability #*3
        variable1_centre = 10
        #sample from distribution
        variable1_samples = scipy.stats.skewnorm.rvs(variable1_skewness,variable1_centre,variable1_scale,number_of_samples)

        samples = [InitialConditions(value) for value in
                   variable1_samples]
        return samples

    def _sample_the_observations(self, conditions: InitialConditions, number_of_analyses: int, coefficient:int) -> Iterable[Observation]:
        '''
        We define an entity as a geological phenomenon that can be directly measured, e.g. a rock sample.
        This function generates observations associated with the entity, e.g. mineral measurements associated with a rock
        The observations (e.g. a mineral) is described by several features (e.g. mineral trace element chemistry)
        The features of an observation are either entirely random, influenced by initial conditions, or influenced by other features (autocorrelation).
        The mathematical relationships between initial conditions and features has no geological interpretation and have been design to exhibit various types of interdependencies.
        '''

        #y=mx+c
        m=0.8
        c=1.5
        feature1 = m*conditions.variable1 +c
        feature1_skewness = 0
        feature1_noise = scipy.stats.skewnorm.rvs(loc=0, a=feature1_skewness, scale=self.intraclass_variability,
                                                  size=number_of_analyses)
        feature1_observations = feature1 + feature1_noise

        # A feature completely independent of initial conditions
        feature2_observations = scipy.stats.uniform.rvs(loc=8, scale=4, size=number_of_analyses)

        #a feature non-linearly related to initial conditions
        a = 0.5
        b= 1.5
        d = 2.2
        feature3 = a*conditions.variable1**2 +b*conditions.variable1 +d
        feature3_observations = feature3+feature1_noise

        feature4_observations = feature1_observations + feature2_observations + feature3_observations+feature1_noise

        '''feature1 = conditions.variable1
        feature1_skewness = 0
        feature1_noise = scipy.stats.skewnorm.rvs(loc=0, a=feature1_skewness,scale=self.intraclass_variability,size=number_of_analyses)
        feature1_observations =feature1 + feature1_noise'''

        observations = [Observation(*values) for values in
                        zip(feature1_observations, feature2_observations,feature3_observations,feature4_observations)]

        return observations


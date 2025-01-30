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
    Objects of this class will hold the 3 hypothetical initial conditions that define a sample.
    '''
    variable1: float
    variable2: float
    variable3: float

class FictitiousSamplerPredictive(AbstractSampler[InitialConditions, Observation]):
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
        variable1_scale = self.interclass_variability #*3
        variable1_centre = 10
        #sample from distribution
        variable1_samples = scipy.stats.norm.rvs(variable1_centre,variable1_scale,number_of_samples)

        variable2_skewness = 0.5
        variable2_centre = 13
        variable2_scale = self.interclass_variability # *4
        # sample from distribution
        variable2_samples = scipy.stats.skewnorm.rvs(variable2_skewness,variable2_centre,variable2_scale,number_of_samples)

        variable3_skewness = 0.3
        variable3_centre = 7
        variable3_scale = self.interclass_variability #*3
        # sample from distribution
        variable3_samples = scipy.stats.skewnorm.rvs(variable3_skewness,variable3_centre,variable3_scale,number_of_samples)

        samples = [InitialConditions(*values) for values in
                   zip(variable1_samples, variable2_samples, variable3_samples)]
        return samples

    def _sample_the_observations(self, conditions: InitialConditions, number_of_analyses: int) -> Iterable[Observation]:
        '''
        We define an entity as a geological phenomenon that can be directly measured, e.g. a rock sample.
        This function generates observations associated with the entity, e.g. mineral measurements associated with a rock
        The observations (e.g. a mineral) is described by several features (e.g. mineral trace element chemistry)
        The features of an observation are either entirely random, influenced by initial conditions, or influenced by other features (autocorrelation).
        The mathematical relationships between initial conditions and features has no geological interpretation and have been design to exhibit various types of interdependencies.
        '''
        # A feature completely independent of initial conditions
        feature1_observations = scipy.stats.uniform.rvs(loc=1, scale=4, size=number_of_analyses)

        # A feature influenced by all 3 initial conditions
        feature2 = (0.4 * (conditions.variable1+conditions.variable2) + 0.3 * conditions.variable1 + 48) + 0.1*(conditions.variable2 + conditions.variable3)
        feature2_skewness = 12
        feature2_noise = scipy.stats.skewnorm.rvs(loc=0, a=feature2_skewness,scale=self.intraclass_variability,size=number_of_analyses)
        feature2_observations =feature2 + feature2_noise

        # A feature derived from features 1 and 2
        feature3_noise = scipy.stats.norm.rvs(loc=1.23,scale=self.intraclass_variability,size=number_of_analyses)
        feature3_observations = 0.7 * conditions.variable1 + 0.3 * conditions.variable2 + 0.13 * conditions.variable3 + feature3_noise

        # A feature influenced by initial conditions (variables 1 and 3)
        feature4_skewness = 7
        feature4_noise = scipy.stats.skewnorm.rvs(loc=2.45,a = feature4_skewness,scale = self.intraclass_variability,size = number_of_analyses)
        feature4_observations = -feature1_observations - np.sqrt(np.abs(feature2_observations)) - feature3_observations + feature4_noise

        observations = [Observation(*values) for values in
                   zip(feature1_observations, feature2_observations, feature3_observations, feature4_observations)]

        return observations


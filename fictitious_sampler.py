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
class SampleConditions:
    variable1: float
    variable2: float
    variable3: float

class FictitiousSampler(AbstractSampler[SampleConditions, Observation]):
    intraclass_variability: float
    interclass_variability: float

    def __init__(self, intraclass_variability,interclass_variability):
        self.intraclass_variability = intraclass_variability
        self.interclass_variability = interclass_variability

    def _generate_samples(self, number_of_samples: int) -> Iterable[SampleConditions]:
        variable1_scale = self.interclass_variability
        variable1_centre = 0
        variable1_distribution = scipy.stats.norm.rvs(variable1_centre,variable1_scale,number_of_samples)

        variable2_skewness = 0.5
        variable2_centre = 13
        variable2_scale = 5*self.interclass_variability
        variable2_distribution = scipy.stats.skewnorm.rvs(variable2_skewness,variable2_centre,variable2_scale,number_of_samples)

        variable3_skewness = 0.3
        variable3_centre = 7
        variable3_scale = 3*self.interclass_variability
        variable3_distribution = scipy.stats.skewnorm.rvs(variable3_skewness,variable3_centre,variable3_scale,number_of_samples)

        samples = [SampleConditions(*values) for values in
                         zip(variable1_distribution, variable2_distribution, variable3_distribution)]
        return samples

    def _generate_observations(self, sample: SampleConditions, number_of_analyses: int) -> Iterable[Observation]:

        feature1_distribution = scipy.stats.uniform.rvs(loc=1, scale=5, size=number_of_analyses)

        feature2 = (-0.4 * sample.variable1 ** 2 + 0.0017 * sample.variable1 + 48) *(sample.variable2/sample.variable3)
        feature2_skewness = 12
        feature2_noise = scipy.stats.skewnorm.rvs(loc=0, a=feature2_skewness,scale=self.intraclass_variability,size=number_of_analyses)
        feature2_distribution =  feature2 + feature2_noise

        feature3_noise = scipy.stats.norm.rvs(loc=1.23,scale=self.intraclass_variability,size=number_of_analyses)
        feature3_distribution = 0.3*feature1_distribution+0.23*feature2_distribution + feature3_noise

        feature4_skewness = 7
        feature4_noise = scipy.stats.skewnorm.rvs(loc=2.45,a = feature4_skewness,scale = 3*self.intraclass_variability,size = number_of_analyses)
        feature4_distribution = 0.7*sample.variable1 + 0.3*sample.variable3**2 + 0.13*sample.variable3 + feature4_noise

        observations = [Observation(*values) for values in
                   zip(feature1_distribution, feature2_distribution, feature3_distribution, feature4_distribution)]

        return observations


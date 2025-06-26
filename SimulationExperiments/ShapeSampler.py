from abc import abstractmethod
from typing import List, Iterable

import numpy as np
from dataclasses import dataclass

import scipy.stats
from scipy.stats import truncnorm

from testing_framework import AbstractSampler

def estimate_ellipse_perimeter(minor_axis, major_axis):
    perimeter = np.pi * (3 * (minor_axis + major_axis) -
                         np.sqrt((3 * minor_axis + major_axis) * (minor_axis + 3 * major_axis)))
    return perimeter

@dataclass
class ZirconGrain:
    area:float
    max_length: float
    min_length: float
    perimeter: float

@dataclass
class CrystallisationParameters:
    silica: float
    cooling_rate:float

class ShapeSampler(AbstractSampler[CrystallisationParameters, ZirconGrain]):
    min_silica=40
    max_silica=80
    low_silica_peak = 45
    low_silica_std_dev = 6
    high_silica_peak = 65
    high_silica_std_dev = 9

    min_cooling_rate = 0
    max_cooling_rate = 10
    mean_cooling_rate = 5
    cooling_rate_std_dev = 4.5

    min_minor_axis = 40
    max_minor_axis = 200
    #TODO get a better idea of the skewness from my paper 2 samples
    major_axis_skewness = 0.5

    def _sample_the_initial_conditions(self, number_of_samples: int) -> Iterable[CrystallisationParameters]:
        number_of_low_silica_samples = int(number_of_samples/3)
        number_of_high_silica_samples = number_of_samples-number_of_low_silica_samples
        low_silica_samples = truncnorm.rvs(self.min_silica,self.max_silica,self.low_silica_peak, self.low_silica_std_dev,number_of_low_silica_samples )
        high_silica_samples = truncnorm.rvs(self.min_silica,self.max_silica,self.high_silica_peak, self.high_silica_std_dev,number_of_high_silica_samples )
        silica_samples = np.concat(low_silica_samples,high_silica_samples)

        cooling_rates = truncnorm.rvs(self.min_cooling_rate,self.max_cooling_rate,self.mean_cooling_rate,self.cooling_rate_std_dev,number_of_samples)

        crystallisation_parameters = [CrystallisationParameters(silica,cooling_rate) for silica,cooling_rate in zip(silica_samples,cooling_rates)]

        return crystallisation_parameters

    def _sample_the_observations(self, conditions: CrystallisationParameters, number_of_analyses: int) -> Iterable[ZirconGrain]:
        scale = self.max_minor_axis-self.min_minor_axis
        minor_axis_measurements = scipy.stats.uniform.rvs(self.min_minor_axis,scale,number_of_analyses)

        cooling_factor = (self.max_cooling_rate - conditions.cooling_rate) / (self.max_cooling_rate - self.min_cooling_rate)
        silica_factor = -4e-8 * conditions.silica ** 2 + 0.0017 * conditions.silica + 48
        hypothetical_major_axis_of_sample = (0.5*cooling_factor +0.5)*silica_factor
        major_axis_measurements = scipy.stats.skewnorm.rvs(loc=hypothetical_major_axis_of_sample, a=self.major_axis_skewness,scale=0.05)

        area_measurements = np.pi*minor_axis_measurements*major_axis_measurements
        perimeter_measurements = estimate_ellipse_perimeter(minor_axis_measurements,major_axis_measurements)

        sample_zircon = [ZirconGrain(*values) for values in zip(area_measurements,major_axis_measurements, minor_axis_measurements, perimeter_measurements)]

        return sample_zircon

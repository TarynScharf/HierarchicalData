from matplotlib import pyplot as plt

from fictitious_sampler import FictitiousSampler
from testing_framework import visualise_data, predict,SplittingStrategy,plot_prediction_results

number_of_tests = 2
fig, axs = plt.subplots(nrows=number_of_tests, ncols=2,sharey=True)
ax= axs.flatten()

for i in range(number_of_tests):
    number_of_samples = 10

    sampler = FictitiousSampler(
        intraclass_variability=10,
        interclass_variability=10
    )

    sample_observation_pairs = sampler.generate_sample_observation_pairs(
        number_of_samples=number_of_samples,
        average_number_of_observations_per_sample=30
    )

    #visualise_data(sample_observation_pairs)
    _sl_x,sl_actuals, sl_predict = predict(sample_observation_pairs,SplittingStrategy.SAMPLE_LEVEL,'variable1')
    _ol_x,ol_actuals, ol_predict = predict(sample_observation_pairs,SplittingStrategy.OBSERVATION_LEVEL,'variable1')
    plot_prediction_results(ax, [sl_actuals,sl_predict], [ol_actuals,ol_predict], i, number_of_samples)

    number_of_samples += 100
plt.tight_layout()
plt.show()
print('complete')
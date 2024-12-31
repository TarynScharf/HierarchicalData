from matplotlib import pyplot as plt

from fictitious_sampler import FictitiousSampler
from testing_framework import visualise_data, predict,SplittingStrategy,plot_prediction_results

number_of_tests = 1
fig, axs = plt.subplots(nrows=number_of_tests, ncols=2,sharey=True,figsize=(10,5))
ax= axs.flatten()

for i in range(number_of_tests):
    number_of_samples = 10

    sampler = FictitiousSampler(
        intraclass_variability=1,
        interclass_variability=1
    )

    sample_observation_pairs = sampler.generate_sample_observation_pairs(
        number_of_samples=number_of_samples,
        average_number_of_observations_per_sample=5
    )

    #visualise_data(sample_observation_pairs)
    sl_predictions = predict(sample_observation_pairs,SplittingStrategy.SAMPLE_LEVEL,'variable1')
    ol_predictions = predict(sample_observation_pairs,SplittingStrategy.OBSERVATION_LEVEL,'variable1')

    plot_prediction_results(ax, i, number_of_samples, sl_predictions, ol_predictions)

    number_of_samples += 100
plt.tight_layout()
plt.show()
print('complete')
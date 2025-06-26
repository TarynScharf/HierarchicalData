import pandas as pd
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from seaborn import objects as so
from matplotlib.ticker import MaxNLocator


def sample_variable_distributions(interclass_variability,number_of_samples):
    variable1_scale = interclass_variability #*3
    variable1_centre = 10
    variable1_samples = scipy.stats.norm.rvs(variable1_centre,variable1_scale,number_of_samples)

    variable2_skewness = 0.5
    variable2_centre = 13
    variable2_scale = interclass_variability # *4
    variable2_samples = scipy.stats.skewnorm.rvs(variable2_skewness,variable2_centre,variable2_scale,number_of_samples)

    variable3_skewness = 0.3
    variable3_centre = 7
    variable3_scale = interclass_variability #*3
    variable3_samples = scipy.stats.skewnorm.rvs(variable3_skewness,variable3_centre,variable3_scale,number_of_samples)

    data_dict = {
        'variable1':variable1_samples,
        'variable2':variable2_samples,
        'variable3':variable3_samples
    }

    df = pd.DataFrame(data_dict)

    return df

def plot_distributions(data):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    fig, axes = plt.subplots(1,3, figsize=(18/2.54,6/2.54), sharey=True)


    p1 = so.Plot(data, x='variable1').add(so.Line(), so.KDE())
    p1.on(axes[0]).plot()

    p2 = so.Plot(data, x='variable2').add(so.Line(), so.KDE())
    p2.on(axes[1]).plot()

    p3 = so.Plot(data, x='variable3').add(so.Line(), so.KDE())
    p3.on(axes[2]).plot()

    for i in range(3):
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i].tick_params(axis='both', width=0.5)

        if i >0:
            axes[i].set_yticklabels([])
            #axes[i].set_yticks([])

        for spine in axes[i].spines.values():
            spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.show()

sampled_data = sample_variable_distributions(1,1000)
plot_distributions(sampled_data)
print('done')


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from scipy.stats import linregress


def load_files(path):
    df = pd.read_csv(path)
    df.rename({df.columns.values[0]: 'count'}, axis = 1, inplace=True)
    df['xvalid'] = 0
    return df

def label_xvalid_iterations(df):
    xvalid = 1
    for i, row in df.iterrows():
        index = row['count']
        if i >0:
            previous_index = df.loc[i-1,'count']
        else:
            previous_index = index

        if index < previous_index:
            xvalid = xvalid + 1

        df.loc[i,'xvalid']=xvalid

    return df

def main():
    obs_path ='C:/Users/283284G/Downloads/Silica-Estimation/Silica-Estimation-main/Outputs/observation_results.csv'
    ent_path ='C:/Users/283284G/Downloads/Silica-Estimation/Silica-Estimation-main/Outputs/entity_results_df.csv'
    df_obs = load_files(obs_path)
    df_ent = load_files(ent_path)

    df_obs = label_xvalid_iterations(df_obs)
    medians_obs = df_obs.groupby(['xvalid', 'actual'])['predicted', 'actual', 'xvalid'].median()
    medians_obs.reset_index(inplace = True, drop = True)
    slope_obs, intercept_obs, r_value_obs, p_value_obs, std_err_obs = linregress(medians_obs['actual'],
                                                                                 medians_obs['predicted'])

    df_ent = label_xvalid_iterations(df_ent)
    df_ent.groupby(['xvalid', 'actual'])
    medians_ent = df_ent.groupby(['xvalid', 'actual'])['predicted', 'actual', 'xvalid'].median()
    medians_ent.reset_index(inplace=True, drop=True)
    slope_ent, intercept_ent, r_value_ent, p_value_ent, std_err_ent = linregress(medians_ent['actual'], medians_ent['predicted'])

    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10

    red_line = (206 / 255, 61 / 255, 48 / 255, 0.9)  # rgba
    red_envelope = (242 / 255, 71 / 255, 56 / 255, 0.6)

    blue_line = (29 / 255, 66 / 255, 115 / 255, 0.9)
    blue_envelope = (4 / 255, 196 / 255, 217 / 255, 0.6)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13 / 2.54, 6.5 / 2.54))
    axes[0].scatter(medians_ent['actual'], medians_ent['predicted'],facecolor=red_envelope, edgecolor=red_line, alpha=0.4 )
    axes[0].plot(medians_ent['actual'],slope_obs*medians_ent['actual']+intercept_ent, color=red_line, label=f'Entity-split R2:{r_value_ent:.1f} ')
    axes[0].set_xlabel('Actual whole-rock silica (%)')
    axes[0].set_ylabel('Median predicted silica (%)')
    axes[0].set_xlim(45, 70)  # Example x-axis limits
    axes[0].set_ylim(45, 70)
    axes[0].legend()

    axes[1].scatter(medians_obs['actual'], medians_obs['predicted'],facecolor=blue_envelope, edgecolor=blue_line, alpha=0.4)
    axes[1].plot(medians_obs['actual'], slope_ent * medians_obs['actual'] + intercept_obs,color=blue_line,label=f'Observation-split R2:{r_value_obs:.1f}')
    axes[1].set_xlabel('Actual whole-rock silica (%)')
    axes[1].set_ylabel('Median predicted silica (%)')
    axes[1].set_xlim(45, 70)  # Example x-axis limits
    axes[1].set_ylim(45, 70)
    axes[1].legend()

    plt.tight_layout()
    #fig.savefig(f'Median points.svg')
    plt.show()
    plt.close(fig)


    print('hello')


main()
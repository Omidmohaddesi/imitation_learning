import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':

    data = {'df1': pd.read_csv('performance_1.csv', index_col=0), 'df2': pd.read_csv('performance_2.csv', index_col=0),
            'df3': pd.read_csv('performance_3.csv', index_col=0)}

    num_traj_dict = {'num_traj_1': [1, 6, 12, 18], 'num_traj_2': [1, 5, 10, 16], 'num_traj_3': [1, 7, 13, 20]}

    data_scaled = data.copy()

    for df, num_traj in zip(data_scaled.values(), num_traj_dict.values()):
        tmp_df = df.copy()
        for i in num_traj:
            min_n = tmp_df[(tmp_df['num_traj'] == i) & (tmp_df['data_type'] == 'Random')]['performance'].values[0]
            max_n = tmp_df[(tmp_df['num_traj'] == i) & (tmp_df['data_type'] == 'Expert')]['performance'].values[0]
            df.performance[df['num_traj'] == i] = df.performance[df['num_traj'] == i].apply(
                lambda x: (x - min_n) / (max_n - min_n))

    titles = ['Human-Like', 'OUL w/o Suggestion', 'OUL w/ Suggestion']
    with sns.axes_style("darkgrid"):

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.25, hspace=0.3, left=.07, wspace=0.2)

        new_labels = ['', 'Expert', 'Random', 'Regression', 'GAIL', 'Behavioral Cloning']
        i = 0
        for dfs, num_traj, title in zip(data_scaled.values(), num_traj_dict.values(), titles):
            sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 1, "lines.markeredgewidth": 1})
            sns.lineplot(x='num_traj', y='performance', hue='data_type', data=dfs, style='data_type',
                         style_order=['GAIL', 'Expert', 'Random', 'Behavioral Cloning', 'Regression'], ax=axes[i])
            axes[i].get_legend().set_title('')
            axes[i].title.set_text(title)
            axes[i].set(xticks=num_traj, ylabel='', xlabel='')
            axes[i].get_legend().remove()
            handles, labels = axes[i].get_legend_handles_labels()
            fig.legend(handles[1:], labels[1:], loc='upper center',
                       bbox_to_anchor=(0.52, 0.11),
                       fancybox=True, shadow=True, ncol=5)
            i += 1
            fig.text(0.52, 0.13, 'Number of trajectories in dataset', va='center', ha='center',
                     fontsize=10, fontname='sans-serif')
            fig.text(0.02, 0.6, 'Performance (scaled)', va='center', ha='center', rotation='vertical',
                     fontsize=10, fontname='sans-serif')
        sns.despine(offset=5, trim=True)
    fig.savefig('performance_all_2.png', format='png', dpi=600)
    # plt.show()

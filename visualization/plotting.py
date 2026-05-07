import matplotlib.pyplot as plt
import seaborn as sns

def setup_project_plots():
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    sns.set_style('darkgrid')

def setup_notebook_plots():
    setup_project_plots()

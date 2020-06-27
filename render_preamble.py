import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import math
from parse import *
from glob import glob
from os.path import join

plt.clf()
sn.set(context='paper', style='darkgrid', palette='muted', font_scale=1.5)
plt.tight_layout()
plt.margins(0,0)

def create_figure(nx, ny, title):
    fig, axes = plt.subplots(nx, ny, figsize=(10,8), sharex=True)
    fig.tight_layout()
    # fig.margins(0, 0)
    fig.subplots_adjust(top=0.90)
    fig.suptitle(title, y=0.97, fontsize=20)
    return fig, axes

def rotate_axis(ax):
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)


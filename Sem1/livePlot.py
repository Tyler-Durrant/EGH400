import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import pandas as pd


df = pd.read_csv('positions.csv')
timestamps = df['timestamp'].to_numpy()
positions = df[['x_pixel', 'y_pixel']].to_numpy()



plt.ion()
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-300, 300)
ax.set_ylim(0, 2000)
ax.axis('off')


for idx, (x,y) in enumerate(positions):

    ax.cla()
    ax.set_aspect('equal')
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-200, 6000)
    ax.axis('off')
    person = patches.Circle((x,y), radius=150, color='blue')
    ax.add_patch(person)
    camera = patches.Polygon([[0, 0], [-100, -200], [100, -200]], closed=True, color='black')
    ax.add_patch(camera)
    ax.plot([0, -4189], [0, 5460], 'gray', linestyle='--')
    ax.plot([0, 4189], [0, 5460], 'gray', linestyle='--')
    if idx < len(timestamps) - 1:
        diff = positions[idx + 1] - positions[idx]
        angle = np.arctan2(diff[1], diff[0])
        arrow_len = 300  # adjust as needed
        x1 = x + arrow_len * np.cos(angle)
        y1 = y + arrow_len * np.sin(angle)
        ax.arrow(x, y, x1 - x, y1 - y, head_width=200, head_length=200, fc='blue', ec='blue')
    plt.draw()
    if idx < len(timestamps) - 1:
        ts = timestamps[idx + 1] - timestamps[idx]
        plt.pause(ts)
plt.ioff()
plt.show()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
import math
import sys

import config as config

class Visualizer(object):
    projection_view_list = []
    t = 0.0
    pause = False

    @classmethod
    def show(cls, project_manager):
        cls.project_manager = project_manager
        # Set up figure and axis
        n_row, n_col = (1, len(project_manager.projection_list))

        cls.fig = plt.figure(1, figsize=(n_col * 4, 4))
        cls.gs = gridspec.GridSpec(n_row, n_col) #, height_ratios=[2, 1, 1, 1])
        cls.gs.update(left=.035, right=.98, top=.95, bottom=.05)

        cls.fig.canvas.mpl_connect('key_press_event', cls.key_press)
        # Init projection views
        if n_col == 1:
            p_axs = plt.subplot(cls.gs[0])
            cls.projection_view_list.append(ProjectionView(cls.project_manager.projection_list[0], p_axs))
        else:
            for p in cls.project_manager.projection_list:
                p_axs = plt.subplot(cls.gs[0, p.position])
                cls.projection_view_list.append(ProjectionView(p, p_axs))

        plt.show()

    @classmethod
    def generate_t(cls):
        t_max = cls.project_manager.n_timesteps - 1
        dt = config.speed
        while cls.t < t_max - dt:
            if not cls.pause:
                cls.t = cls.t + dt
                if cls.t % 1 < dt and config.autopause:  # Pause when we reach a round number
                    cls.pause = True
            yield cls.t
        yield t_max

    @classmethod
    def key_press(cls, event):
        print('press', event.key)
        sys.stdout.flush()
        if event.key == ' ':  # Continue animation with spacebar
            cls.pause ^= True


class ProjectionView:
    def __init__(self, projection, ax):
        plt.figure(1)

        self.projection = projection
        self.ax_p = ax
        self.ax_p.set_title(projection.name, fontsize=9)
        self.scatter = self.ax_p.scatter([], [], s=10, edgecolors='#000000', animated=True)
        self.colormap = matplotlib.cm.Set3
        self.colors = [self.colormap(cl) for cl in self.projection.get_class_list()]
        self.ax_p.set_xlim(self.projection.x_limits)
        self.ax_p.set_ylim(self.projection.y_limits)
        # self.labels = [self.ax_p.annotate(txt, (0, 0), fontsize=8) for i, txt in enumerate(self.projection.coord_df['id'])]

        plt.figure(1)

        self.time_text = plt.text(0, 0, '0', fontsize=12, transform=self.ax_p.transAxes)
        if self.projection.position != 0:
            self.time_text.set_visible(False)

        self.ani = FuncAnimation(Visualizer.fig, self.update_anim, Visualizer.generate_t,
                                 blit=True, interval=10, repeat=False)

    def update_anim(self, t):
        plt.figure(1)
        df = self.projection.coord_df
        # Interpolate x and y in time
        dt = t - math.floor(t)
        x_series = df['t{}d0'.format(math.floor(t))] * (1 - dt) + df['t{}d0'.format(math.ceil(t))] * dt
        y_series = df['t{}d1'.format(math.floor(t))] * (1 - dt) + df['t{}d1'.format(math.ceil(t))] * dt
        self.scatter.set_offsets(np.vstack((x_series, y_series)).T)
        self.scatter.set_color(self.colors)
        self.scatter.set_edgecolor('k')
        self.scatter.set_linewidth(.3)
        self.time_text.set_text(str(t)[:4])
        # for i, txt in enumerate(self.labels):
        #     txt.set_position((x_series[i], y_series[i]))

        if self.projection.position == 0:
            legend_elements = [Line2D([0], [0], color=self.colormap(cl[1]), marker='.', label=cl[0]) for cl in self.projection.class_dict.items()]
            legend = self.ax_p.legend(handles=legend_elements, loc=2, prop={'size': 6})
            return self.scatter, self.time_text, legend

        # return (self.scatter, self.time_text, *self.labels)
        return self.scatter, self.time_text

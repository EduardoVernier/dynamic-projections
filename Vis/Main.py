import pandas as pd
import sys
import re

from ProjectionManager import ProjectionManager, Projection
from Visualizer import Visualizer
import config as config


if __name__ == '__main__':
    # Read inputs and instantiate Projection objects
    projection_manager = ProjectionManager()
    for i in range(1, len(sys.argv)):
        name = re.match(r'.*/(.*).csv', sys.argv[i]).group(1)
        df = pd.read_csv(sys.argv[i])
        position = i - 1
        projection_manager.append(Projection(name, df, position))

    if config.visualize_projections:
        Visualizer.show(projection_manager)

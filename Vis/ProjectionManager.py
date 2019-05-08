import re

class ProjectionManager:
    def __init__(self):
        self.projection_list = []
        self.max_row = 0
        self.max_col = 0
        self.n_timesteps = 0

    def append(self, projection):
        self.projection_list.append(projection)
        if self.n_timesteps == 0: self.n_timesteps = int(re.match(r't(\d*).*', projection.coord_df.columns[-1]).group(1)) + 1

    def get_projection_names(self):
        return [p.name for p in self.projection_list]


class Projection:
    def __init__(self, name, coord_df, position):
        self.name = name
        self.coord_df = coord_df.copy()
        self.n_timesteps = int(re.match(r't(\d*).*', self.coord_df.columns[-1]).group(1)) + 1
        classes = coord_df['id'].str.split('-').str[0].unique()
        self.class_dict = {c:n for n,c in enumerate(classes)}
        # Plotting
        self.position = position
        self.labeled_data = True  # id in coord df gives says the class of each item TODO Put this is argv
        self.x_limits = (coord_df[[col_name for col_name in coord_df.columns if 'd0' in col_name]].values.max(),
                         coord_df[[col_name for col_name in coord_df.columns if 'd0' in col_name]].values.min())
        self.y_limits = (coord_df[[col_name for col_name in coord_df.columns if 'd1' in col_name]].values.max(),
                         coord_df[[col_name for col_name in coord_df.columns if 'd1' in col_name]].values.min())
        # Metrics
        self.movement_df = None
        self.distances_df = None

    def get_class_list(self):
        return (self.class_dict[cl] for cl in self.coord_df['id'].str.split('-').str[0])

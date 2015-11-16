from . import plot_utils


class MetaModel():
    """MetaModel class"""
    def __init__(self, models=[]):
        self.models = models

    def add_model(self, model):
        """Add a model to the meta model"""
        self.models.append(model)

    def plot_meta_model(self):
        """Plot the meta model"""
        ax = plot_utils.init_3d_figure()

        for model in self.models:
            # Add each model to the plot
            model.plot_model(ax=ax, show=False)

        # Display the plot
        plot_utils.show_figure()

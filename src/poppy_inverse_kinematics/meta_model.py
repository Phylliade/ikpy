# coding= utf8


class MetaModel():
    """MetaModel class"""
    def __init__(self, models=[], pypot_object=None):
        self.models = models
        self.pypot_object = pypot_object

    def add_model(self, model):
        """Add a model to the meta model"""
        model.pypot_object = self.pypot_object
        self.models.append(model)

    def plot_meta_model(self):
        """Plot the meta model"""
        from . import plot_utils
        ax = plot_utils.init_3d_figure()

        for model in self.models:
            # Add each model to the plot
            model.plot_model(ax=ax, show=False)

        # Display the plot
        plot_utils.show_figure()

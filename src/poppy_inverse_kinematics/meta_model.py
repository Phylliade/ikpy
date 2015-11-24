# coding= utf8
from . import model_interface


class MetaModel(model_interface.ModelInterface):
    """MetaModel class"""
    def __init__(self, models=[], pypot_object=None, interface_type=None, move_duration=None):
        self.models = models
        model_interface.ModelInterface.__init__(self, pypot_object=pypot_object, interface_type=interface_type, move_duration=move_duration)

    def add_model(self, model):
        """Add a model to the meta model"""
        model.pypot_object = self.pypot_object
        model.interface_type = self.interface_type
        model.move_duration = self.move_duration
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

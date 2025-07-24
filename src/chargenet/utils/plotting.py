import joblib
import numpy as np

from vedo import Volume, Plotter


button_kwargs = dict(
    c=["w", "w"],
    bc=["dg", "dv"],
    font="courier",
    size=25,
    bold=True,
    italic=False,
)


def load_electrostatics(path, variant_id):
    data = joblib.load(path)[variant_id].numpy()
    charge = data[0, :, :, :]
    charge_density = data[1, :, :, :]
    potential = np.clip(data[2, :, :, :], -100, 100)
    return charge, charge_density, potential


def create_volume(array, mid_range, color_min, color_max, mode, name, bar_position):
    volume = Volume(
        inputobj=array,
        alpha=[
            (array.min(), 0.8),
            (array.min() / 2, 0.8),
            (0, 0),
            (array.max() / 2, 0.8),
            (array.max(), 0.8),
        ],
        mode=mode,
        mapper="smart",
    )
    colors = [
        (array.min(), color_min),
        (mid_range, [255, 255, 255]),
        (array.max(), color_max),
    ]
    volume.cmap(colors)
    volume.add_scalarbar(name, use_alpha=True, pos=bar_position)
    return volume


def create_volume_hide_function(volume, button):
    def hide():
        if volume._alpha == 0:
            array = volume.tonumpy()
            volume.alpha(
                [
                    (array.min(), 0.8),
                    (array.min() / 2, 0.8),
                    (0, 0),
                    (array.max() / 2, 0.8),
                    (array.max(), 0.8),
                ],
            )
        else:
            volume.alpha(0)
        button.switch()

    return hide


def create_mesh_hide_function(mesh, button):
    def hide():
        if mesh.alpha != 0:
            mesh.force_translucent()
        else:
            mesh.force_opaque()
        button.switch()

    return hide


def create_mode_function(volumes, button):
    def set_mode():
        for volume in volumes:
            if volume._mode == 0:
                volume.mode(1)
            elif volume._mode == 1:
                volume.mode(0)
        button.switch()

    return set_mode


def create_alpha_function(volumes):
    def set_alpha(widget, event):
        for volume in volumes:
            if volume._alpha != 0:
                volume.alpha(widget.value)

    return set_alpha


def plot_electrostatics(path, variant_id):
    charge, charge_density, potential = load_electrostatics(path, variant_id)
    linear_space = np.linspace(charge_density.min(), charge_density.max(), 20)
    filtered_linear_space = linear_space[(linear_space < -0.05) | (linear_space > 0.05)]
    charge_density_volume = create_volume(
        charge_density,
        mid_range=0,
        color_min=[0.0, 0.0, 255],
        color_max=[255, 0.0, 0.0],
        mode=0,
        name="Charge density",
        bar_position=(0.8, 0.05),
    )
    charge_density_mesh = (
        charge_density_volume.isosurface(filtered_linear_space)
        .alpha(0.5)
        .cmap(["red", "blue"])
    )

    potential_volume = create_volume(
        potential,
        mid_range=0,
        color_min=[0.0, 255, 255],
        color_max=[255, 255, 0.0],
        mode=0,
        name="Potential",
        bar_position=(0.75, 0.05),
    )
    charge_volume = create_volume(
        charge,
        mid_range=0,
        color_min=[0, 100, 0],
        color_max=[255, 0.0, 255],
        mode=0,
        name="Charge",
        bar_position=(0.7, 0.05),
    )

    plt = Plotter(bg="blackboard", axes=7)

    hide_potential_button = plt.add_button(
        fnc=None,
        pos=(0.2, 0.15),  # x,y fraction from bottom left corner
        states=["Hide potential", "Show potential"],
        **button_kwargs,
    )
    hide_potential_button.function = create_volume_hide_function(
        potential_volume, hide_potential_button
    )

    hide_charge_button = plt.add_button(
        fnc=None,
        pos=(0.2, 0.1),  # x,y fraction from bottom left corner
        states=["Hide charge", "Show charge"],
        **button_kwargs,
    )
    hide_charge_button.function = create_volume_hide_function(
        charge_volume, hide_charge_button
    )

    mode_button = plt.add_button(
        fnc=None,
        pos=(0.8, 0.8),
        states=["Composite", "Max projection"],
        **button_kwargs,
    )
    mode_button.function = create_mode_function(
        [potential_volume, charge_volume], mode_button
    )
    plt.show(charge_density_mesh, potential_volume, charge_volume)

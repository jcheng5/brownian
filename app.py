from pathlib import Path

from brownian_motion import brownian_data, brownian_widget
from mediapipe import hand_to_camera_eye, info_smoother
from shiny import reactive, req
from shiny.express import input, ui, render
from shinywidgets import render_plotly
import pandas as pd

from shinymediapipe import input_hand
from smoother import reactive_smooth

# Check that JS prerequisites are installed
if not (Path(__file__).parent / "shinymediapipe" / "node_modules").is_dir():
    raise RuntimeError(
        "Mediapipe dependencies are not installed. "
        "Please run `npm install` in the 'shinymediapipe' subdirectory."
    )

# Set to True to see underlying XYZ values and canvas
debug = True

ui.page_opts(title="Brownian motion", fillable=True)

with ui.sidebar(open="desktop"):
    ui.input_action_button("data_btn", "New Data", class_="btn-primary")
    ui.p(ui.input_switch("use_smoothing", "Smooth tracking", True))
    if debug:

        @render.data_frame
        def table():
            return pd.DataFrame(
                {
                    "dim": ["x", "y", "z"],
                    "value": [
                        camera_info()["eye"]["x"],
                        camera_info()["eye"]["y"],
                        camera_info()["eye"]["z"],
                    ],
                }
            )


@render_plotly(fill=True)
def plot():
    return brownian_widget(600, 600)


input_hand("hand", debug=debug, throttle_delay_secs=0.05)

# BROWNIAN MOTION ====


@reactive.calc
@reactive.event(input.data_btn, ignore_none=False)
def random_walk():
    """Generates brownian data whenever 'New Data' is clicked"""
    return brownian_data(n=200)


@reactive.effect
def resize_widget():
    """Manually size the plotly widget to fill its container"""
    width = input[".clientdata_output_plot_width"]()
    height = input[".clientdata_output_plot_height"]()
    if width != 0 and height != 0:
        plot.widget.update_layout(width=int(width), height=int(height))


@reactive.effect
def update_plotly_data():
    walk = random_walk()
    layer = plot.widget.data[0]
    layer.x = walk["x"]
    layer.y = walk["y"]
    layer.z = walk["z"]
    layer.marker.color = walk["z"]


# HAND TRACKING ====


@reactive.calc
def camera_info():
    """The eye position, as reflected by the hand input"""
    hand_val = input.hand()
    req(hand_val)

    res = hand_to_camera_eye(hand_val, detect_ok=True)
    req(res)
    return res


# The raw data is a little jittery. Smooth it out by averaging a few samples
@reactive_smooth(n_samples=5, smoother=info_smoother)
@reactive.calc
def smooth_camera_info():
    return camera_info()


@reactive.effect
def update_plotly_camera():
    """Update Plotly camera using the hand tracking"""
    info = smooth_camera_info() if input.use_smoothing() else camera_info()
    plot.widget.layout.scene.camera.eye = info["eye"] if info is not None else None
    plot.widget.layout.scene.camera.up = info["up"] if info is not None else None

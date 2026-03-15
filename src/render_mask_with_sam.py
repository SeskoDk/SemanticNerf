import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton


from src.utils import load_config, arg_parse


def main():
    args = arg_parse()
    cfg = load_config(args.config_path)

    image_path = Path(cfg.paths.root) / "axis_renders" / "z_rgb.png"
    output_mask = Path(cfg.paths.root) / "axis_renders" / "topdown_filled_mask.png"

    print()
    print(image_path)
    print()

    image = cv2.imread(str(image_path))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    points = []

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.set_title("Left click: add point | Enter: close polygon | R: reset")
    ax.axis("off")

    (line,) = ax.plot([], [], "-g", lw=2)
    scatter = ax.scatter([], [], c="g", s=30)

    def on_click(event):
        if event.inaxes != ax:
            return

        if event.button is MouseButton.LEFT:
            if event.xdata is None or event.ydata is None:
                return
            points.append((event.xdata, event.ydata))
            update_plot()

    def on_key(event):
        nonlocal_poly = False

        if event.key == "enter" and len(points) >= 3:
            create_mask()
            plt.close(fig)

        elif event.key == "r":
            points.clear()
            update_plot()

    def update_plot():
        if len(points) == 0:
            line.set_data([], [])
            scatter.set_offsets(np.empty((0, 2)))
        else:
            xs, ys = zip(*points)
            line.set_data(xs, ys)
            scatter.set_offsets(points)

        fig.canvas.draw_idle()

    def create_mask():
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
        cv2.imwrite(str(output_mask), mask)
        print(f"Polygon mask saved to {output_mask}")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    main()

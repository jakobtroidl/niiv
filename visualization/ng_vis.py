from __future__ import print_function

import argparse
import numpy as np

import neuroglancer
import neuroglancer.cli
import webbrowser


def add_example_layers(state):
    a = np.load(
        "/home/jakobtroidl/Desktop/neural-volumes/logs/hemibrain-final-fixed/results_iteration_1/15750_18505_17484.npy/bilinear.npy"
    )

    print(a.shape)

    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
    )

    state.dimensions = dimensions
    state.layers.append(
        name="a",
        layer=neuroglancer.LocalVolume(
            data=a,
            dimensions=dimensions,
            voxel_offset=(1, 1, 1),
            volume_type="image",
        )
    )
    return a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        a = add_example_layers(s)

    webbrowser.open_new(viewer.get_viewer_url())

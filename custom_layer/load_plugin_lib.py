import os
import ctypes

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
PLUGIN_LIBRARY_NAME = "libpreproc_layer.so"
PLUGIN_LIBRARY = [os.path.join(WORKING_DIR, "plugin_src", "build", PLUGIN_LIBRARY_NAME)]


def load_plugin_lib():
    for plugin_lib in PLUGIN_LIBRARY:
        if os.path.isfile(plugin_lib):
            try:
                ctypes.CDLL(plugin_lib, winmode=0)
            except TypeError:
                # winmode only introduced in python 3.8
                ctypes.CDLL(plugin_lib)
            return

    raise IOError(
        "\n{}\n{}\n{}\n".format(
            "Failed to load library ({}).".format(PLUGIN_LIBRARY_NAME),
            "Please build the Hardmax sample plugin.",
            "For more information, see the included README.md",
        )
    )
import sys

import geo_toolbox.geochem_gui
import geo_toolbox.gui

blocked = {"tkinter", "matplotlib", "harmonica", "verde", "ppigrf", "pyrolite",
           "pykrige", "gstools"} & set(sys.modules)
print("loaded:", sorted(blocked) if blocked else "none")

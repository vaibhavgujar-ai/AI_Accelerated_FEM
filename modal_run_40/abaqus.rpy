# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2019 replay file
# Internal Version: 2018_09_25-00.11.51 157541
# Run by User on Sun May 31 15:11:50 2026
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=283.536956787109, 
    height=189.933334350586)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
o2 = session.openOdb(name='modal_run_40.odb')
#: Model: C:/Paper3/50_run_automation/modal_run_40/modal_run_40.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       5
#: Number of Node Sets:          5
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].view.setValues(nearPlane=115.499, 
    farPlane=209.912, width=103.758, height=50.1637, viewOffsetX=-0.899144, 
    viewOffsetY=1.86404)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))

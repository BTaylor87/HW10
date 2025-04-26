#region imports
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import math
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg

#these imports are necessary for drawing a matplot lib graph on my GUI
#no simple widget for this exists in QT Designer, so I have to add the widget in code.
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
#endregion

#region class definitions
#region specialized graphic items
class MassBlock(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, width=30, height=10, parent=None, pen=None, brush=None, name='CarBody', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = brush
        self.width = width
        self.height = height
        self.top = self.y - self.height/2
        self.left = self.x - self.width/2
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)
        self.name = name
        self.mass = mass
        self.transformation = qtg.QTransform()
        stTT = self.name +"\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)

    def boundingRect(self):
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def paint(self, painter, option, widget=None):
        self.transformation.reset()
        if self.pen is not None:
            painter.setPen(self.pen)  # Red color pen
        if self.brush is not None:
            painter.setBrush(self.brush)
        self.top = -self.height/2
        self.left = -self.width/2
        self.rect=qtc.QRectF( self.left, self.top, self.width, self.height)
        painter.drawRect(self.rect)
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
        self.transformation.reset()
        # brPen=qtg.QPen()
        # brPen.setWidth(0)
        # painter.setPen(brPen)
        # painter.setBrush(qtc.Qt.NoBrush)
        # painter.drawRect(self.boundingRect())

class Wheel(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, radius=10, parent=None, pen=None, wheelBrush=None, massBrush=None, name='Wheel', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = wheelBrush
        self.radius = radius
        self.rect = qtc.QRectF(self.x - self.radius, self.y - self.radius, self.radius*2, self.radius*2)
        self.name = name
        self.mass = mass
        self.transformation = qtg.QTransform()
        stTT = self.name +"\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)
        self.massBlock = MassBlock(CenterX, CenterY, width=2*radius*0.85, height=radius/3, pen=pen, brush=massBrush, name="Wheel Mass", mass=mass)

    def boundingRect(self):
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect
    def addToScene(self, scene):
        scene.addItem(self)
        scene.addItem(self.massBlock)

    def paint(self, painter, option, widget=None):
        self.transformation.reset()
        if self.pen is not None:
            painter.setPen(self.pen)  # Red color pen
        if self.brush is not None:
            painter.setBrush(self.brush)
        self.rect=qtc.QRectF(-self.radius, -self.radius, self.radius*2, self.radius*2)
        painter.drawEllipse(self.rect)
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
        self.transformation.reset()
        # brPen=qtg.QPen()
        # brPen.setWidth(0)
        # painter.setPen(brPen)
        # painter.setBrush(qtc.Qt.NoBrush)
        # painter.drawRect(self.boundingRect())

#endregion

#region MVC for quarter car model
class CarModel():
    """
    I re-wrote the quarter car model as an object oriented program
    and used the MVC pattern.  This is the quarter car model.  It just
    stores information about the car and results of the ode calculation.
    """
    def __init__(self):
        """
        self.results to hold results of odeint solution
        self.t time vector for odeint and for plotting
        self.tramp is time required to climb the ramp
        self.angrad is the ramp angle in radians
        self.ymag is the ramp height in m
        """
        self.results = []
        self.tmax = 3.0  # limit of timespan for simulation in seconds
        self.t = np.linspace(0, self.tmax, 200)
        self.tramp = 1.0  # time to traverse the ramp in seconds
        self.angrad = 0.1
        self.ymag = 6.0 / (12 * 3.3)  # ramp height in meters.  default is 0.1515 m
        self.yangdeg = 45.0  # ramp angle in degrees.  default is 45
        self.results = None

        # Set default values using physics calculations and GUI defaults
        self.m1 = 450.0  # kg (car body mass from GUI default)
        self.m2 = 20.0  # kg (wheel mass from GUI default)
        self.c1 = 4500.0  # N*s/m (damping from GUI default)
        self.k1 = 15000.0  # N/m (suspension spring from GUI)
        self.k2 = 90000.0  # N/m (tire spring from GUI default)
        self.v = 120.0  # kph (speed from GUI default)

        # Calculate min/max spring constants using static deflection
        g = 9.81  # m/s² (gravity)

        # Suspension static deflection range: 3" (0.0762 m) to 6" (0.1524 m)
        self.mink1 = (self.m1 * g) / 0.1524  # min k1 at max deflection (6")
        self.maxk1 = (self.m1 * g) / 0.0762  # max k1 at min deflection (3")

        # Tire static deflection range: 0.75" (0.01905 m) to 1.5" (0.0381 m)
        total_mass = self.m1 + self.m2  # mass on tire
        self.mink2 = (total_mass * g) / 0.0381  # min k2 at max deflection (1.5")
        self.maxk2 = (total_mass * g) / 0.01905  # max k2 at min deflection (0.75")

        # Acceleration limits (2.0g in m/s²)
        self.accelLim = 2.0 * 9.81  # 19.62 m/s²
        self.accelMax = 0.0  # initialize acceleration maximum

class CarView():
    def __init__(self, args):
        self.input_widgets, self.display_widgets = args
        # unpack widgets with same names as they have on the GUI
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
         self.le_tmax, self.chk_IncludeAccel = self.input_widgets

        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
        self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        # creating a canvas to draw a figure for the car model
        self.figure = Figure(tight_layout=True, frameon=True, facecolor='none')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout_horizontal_main.addWidget(self.canvas)

        # axes for the plotting using view
        self.ax = self.figure.add_subplot()
        if self.ax is not None:
            self.ax1 = self.ax.twinx()

        self.buildScene()

    def updateView(self, model=None):
        self.le_m1.setText("{:0.2f}".format(model.m1))
        self.le_k1.setText("{:0.2f}".format(model.k1))
        self.le_c1.setText("{:0.2f}".format(model.c1))
        self.le_m2.setText("{:0.2f}".format(model.m2))
        self.le_k2.setText("{:0.2f}".format(model.k2))
        self.le_ang.setText("{:0.2f}".format(model.yangdeg))
        self.le_tmax.setText("{:0.2f}".format(model.tmax))
        stTmp="k1_min = {:0.2f}, k1_max = {:0.2f}\nk2_min = {:0.2f}, k2_max = {:0.2f}\n".format(model.mink1, model.maxk1, model.mink2, model.maxk2)
        stTmp+="SSE = {:0.2f}".format(model.SSE)
        self.lbl_MaxMinInfo.setText(stTmp)
        self.doPlot(model)

    def buildScene(self):
        # create a scene object
        self.scene = qtw.QGraphicsScene()
        self.scene.setObjectName("MyScene")
        self.scene.setSceneRect(-200, -200, 400, 400)  # xLeft, yTop, Width, Height

        # set the scene for the graphics view object
        self.gv_Schematic.setScene(self.scene)
        # make some pens and brushes for my drawing
        self.setupPensAndBrushes()

        # Wheel object: draw wheel with tire mass
        self.Wheel = Wheel(0, 50, 50, pen=self.penWheel, wheelBrush=self.brushWheel, massBrush=self.brushMass,
                           name="Wheel")
        self.Wheel.addToScene(self.scene)

        # Car body object: draw mass block representing the car body
        self.CarBody = MassBlock(0, -70, 100, 30, pen=self.penWheel, brush=self.brushMass, name="Car Body", mass=150)
        self.scene.addItem(self.CarBody)

        # JES MISSING CODE - ADDITIONAL GRAPHICS

        # --- Suspension Spring (between Car Body and Wheel) ---
        # Create a simple spring using a line
        spring_pen = qtg.QPen(qtg.QColor("blue"))
        spring_pen.setWidth(2)
        self.scene.addLine(0, -55, 0, 25, spring_pen)  # vertical line from bottom of car to top of wheel

        # --- Damper (Shock Absorber) next to Spring ---
        damper_pen = qtg.QPen(qtg.QColor("red"))
        damper_pen.setWidth(2)
        self.scene.addLine(20, -55, 20, 25, damper_pen)  # vertical line for damper, offset sideways for clarity

        # --- Tire Spring (between Wheel and Ground) ---
        tire_pen = qtg.QPen(qtg.QColor("green"))
        tire_pen.setWidth(2)
        self.scene.addLine(0, 75, 0, 100, tire_pen)  # line from wheel bottom downward to ground

        # --- Ground Line (Road Surface) ---
        ground_pen = qtg.QPen(qtg.QColor("black"))
        ground_pen.setWidth(2)
        self.scene.addLine(-100, 100, 100, 100, ground_pen)  # long horizontal line representing the ground

        # --- Labels (optional but helpful) ---
        # Car label
        car_text = qtw.QGraphicsTextItem("Car Body")
        car_text.setPos(-50, -100)
        self.scene.addItem(car_text)

        # Wheel label
        wheel_text = qtw.QGraphicsTextItem("Wheel + Tire")
        wheel_text.setPos(-50, 80)
        self.scene.addItem(wheel_text)

        # Ground label
        ground_text = qtw.QGraphicsTextItem("Ground")
        ground_text.setPos(-50, 105)
        self.scene.addItem(ground_text)

        # Spring label
        spring_text = qtw.QGraphicsTextItem("Suspension Spring")
        spring_text.setPos(25, -15)
        self.scene.addItem(spring_text)

        # Damper label
        damper_text = qtw.QGraphicsTextItem("Damper")
        damper_text.setPos(45, -15)
        self.scene.addItem(damper_text)

        # Tire spring label
        tire_text = qtw.QGraphicsTextItem("Tire Spring")
        tire_text.setPos(25, 60)
        self.scene.addItem(tire_text)

    #End of Region

    def setupPensAndBrushes(self):
        self.penWheel = qtg.QPen(qtg.QColor("orange"))
        self.penWheel.setWidth(1)
        self.brushWheel = qtg.QBrush(qtg.QColor.fromHsv(35,255,255, 64))
        self.brushMass = qtg.QBrush(qtg.QColor(200,200,200, 128))

    def doPlot(self, model=None):
        if model.results is None:
            return
        ax = self.ax
        ax1 = self.ax1
        # plot result of odeint solver
        QTPlotting = True  # assumes we are plotting onto a QT GUI form
        if ax == None:
            ax = plt.subplot()
            ax1 = ax.twinx()
            QTPlotting = False  # actually, we are just using CLI and showing the plot

        ax.clear()
        ax1.clear()

        t = model.t

        ycar = model.results[:, 0]
        ywheel = model.results[:, 2]
        accel = model.accel



        if self.chk_LogX.isChecked():
            ax.set_xlim(0.001,model.tmax)
            ax.set_xscale('log')
        else:
            ax.set_xlim(0.0, model.tmax)
            ax.set_xscale('linear')

        if self.chk_LogY.isChecked():
            ax.set_ylim(0.0001,max(ycar.max(), ywheel.max()*1.05))
            ax.set_yscale('log')
        else:
            ax.set_ylim(0.0, max(ycar.max(), ywheel.max()*1.05))
            ax.set_yscale('linear')

        ax.plot(t, ycar, 'b-', label='Body Position')
        ax.plot(t, ywheel, 'r-', label='Wheel Position')
        if self.chk_ShowAccel.isChecked():
            ax1.plot(t, accel, 'g-', label='Body Accel')
            ax1.axhline(y=accel.max(), color='orange')  # horizontal line at accel.max()
            ax1.set_yscale('log' if self.chk_LogAccel.isChecked() else 'linear')

        # add axis labels
        ax.set_ylabel("Vertical Position (m)", fontsize='large' if QTPlotting else 'medium')
        ax.set_xlabel("time (s)", fontsize='large' if QTPlotting else 'medium')
        ax1.set_ylabel("Y'' (g)", fontsize = 'large' if QTPlotting else 'medium')
        ax.legend()

        ax.axvline(x=model.tramp)  # vertical line at tramp
        ax.axhline(y=model.ymag)  # horizontal line at ymag
        # modify the tick marks
        ax.tick_params(axis='both', which='both', direction='in', top=True,
                       labelsize='large' if QTPlotting else 'medium')  # format tick marks
        ax1.tick_params(axis='both', which='both', direction='in', right=True,
                       labelsize='large' if QTPlotting else 'medium')  # format tick marks
        # show the plot
        if QTPlotting == False:
            plt.show()
        else:
            self.canvas.draw()

class CarController():
    def __init__(self, args):
        """
        This is the controller I am using for the quarter car model.
        """
        self.input_widgets, self.display_widgets = args
        #unpack widgets with same names as they have on the GUI
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
         self.le_tmax, self.chk_IncludeAccel = self.input_widgets

        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
        self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        self.model = CarModel()
        self.view = CarView(args)

        self.chk_IncludeAccel=qtw.QCheckBox()

    def ode_system(self, X, t):
        # Unpack state variables
        x1, x1dot, x2, x2dot = X

        # Road profile calculation
        if t < self.model.tramp:
            y = self.model.ymag * (t / self.model.tramp)
        else:
            y = self.model.ymag

        # Calculate forces using Hooke's Law and damping
        F_suspension = self.model.k1 * (x2 - x1) + self.model.c1 * (x2dot - x1dot)
        F_tire = self.model.k2 * (y - x2)

        # Newton's Second Law for both masses
        x1ddot = (F_suspension) / self.model.m1
        x2ddot = (-F_suspension + F_tire) / self.model.m2

        return [x1dot, x1ddot, x2dot, x2ddot]

    def calculate(self, doCalc=True):
        # Read values from GUI widgets and convert to float
        self.model.m1 = float(self.le_m1.text())
        self.model.m2 = float(self.le_m2.text())
        self.model.c1 = float(self.le_c1.text())
        self.model.k1 = float(self.le_k1.text())
        self.model.k2 = float(self.le_k2.text())
        self.model.v = float(self.le_v.text())

        # Recalculate min/max spring constants based on current masses
        g = 9.81
        # Suspension springs (k1)
        self.model.mink1 = (self.model.m1 * g) / 0.1524  # 6" deflection
        self.model.maxk1 = (self.model.m1 * g) / 0.0762  # 3" deflection

        # Tire springs (k2)
        total_mass = self.model.m1 + self.model.m2
        self.model.mink2 = (total_mass * g) / 0.0381  # 1.5" deflection
        self.model.maxk2 = (total_mass * g) / 0.01905  # 0.75" deflection

        ymag=6.0/(12.0*3.3)   #This is the height of the ramp in m
        if ymag is not None:
            self.model.ymag = ymag
        self.model.yangdeg = float(self.le_ang.text())
        self.model.tmax = float(self.le_tmax.text())
        if(doCalc):
            self.doCalc()
        self.SSE((self.model.k1, self.model.c1, self.model.k2), optimizing=False)
        self.view.updateView(self.model)

    def setWidgets(self, w):
        self.view.setWidgets(w)
        self.chk_IncludeAccel=self.view.chk_IncludeAccel

    def doCalc(self, doPlot=True, doAccel=True):
        """
        This solves the differential equations for the quarter car model.
        :param doPlot:
        :param doAccel:
        :return:
        """
        v = 1000 * self.model.v / 3600  # convert speed to m/s from kph
        self.model.angrad = self.model.yangdeg * math.pi / 180.0  # convert angle to radians
        self.model.tramp = self.model.ymag / (math.sin(self.model.angrad) * v)  # calculate time to traverse ramp

        self.model.t=np.linspace(0,self.model.tmax,2000)
        ic = [0, 0, 0, 0]
        # run odeint solver
        self.model.results = odeint(self.ode_system, ic, self.model.t)
        if doAccel:
            self.calcAccel()
        if doPlot:
            self.doPlot()

    def calcAccel(self):
        """
        Calculate the acceleration in the vertical direction using the forward difference formula.
        """
        N=len(self.model.t)
        self.model.accel=np.zeros(shape=N)
        vel=self.model.results[:,1]
        for i in range(N):
            if i==N-1:
                h = self.model.t[i] - self.model.t[i-1]
                self.model.accel[i]=(vel[i]-vel[i-1])/(9.81*h)  # backward difference of velocity
            else:
                h = self.model.t[i + 1] - self.model.t[i]
                self.model.accel[i] = (vel[i + 1] - vel[i]) / (9.81 * h)  # forward difference of velocity
            # else:
            #     self.model.accel[i]=(vel[i+1]-vel[i-1])/(9.81*2.0*h)  # central difference of velocity
        self.model.accelMax=self.model.accel.max()
        return True

    def OptimizeSuspension(self):
        """
        OptimizeSuspension
        ------------------
        Main method to optimize the suspension system (k1, c1, k2) by minimizing SSE.

        Steps:
        1. Read user parameters from GUI but do NOT solve yet (calculate(doCalc=False)).
        2. Make an initial guess (x0) from current parameter values.
        3. Use scipy.optimize.minimize with Nelder-Mead method to minimize SSE.
        4. Update model and GUI with optimized suspension values.
        """

        # Step 1: Set parameters from GUI without calculation
        self.calculate(doCalc=False)

        # Step 2: Initial guess using current values
        x0 = np.array([self.model.k1, self.model.c1, self.model.k2])

        # Step 3: Perform optimization using Nelder-Mead
        answer = minimize(self.SSE, x0, method='Nelder-Mead')

        # Update model with optimized values
        self.model.k1, self.model.c1, self.model.k2 = answer.x

        # Update GUI and plot
        self.view.updateView(self.model)

    # End of Region

    def SSE(self, vals, optimizing=True):
        """
        Calculates the sum of square errors between the contour of the road and the car body.
        :param vals:
        :param optimizing:
        :return:
        """
        k1, c1, k2=vals  #unpack the new values for k1, c1, k2
        self.model.k1=k1
        self.model.c1=c1
        self.model.k2=k2
        self.doCalc(doPlot=False)  #solve the odesystem with the new values of k1, c1, k2
        SSE=0
        for i in range(len(self.model.results[:,0])):
            t=self.model.t[i]
            y=self.model.results[:,0][i]
            if t < self.model.tramp:
                ytarget = self.model.ymag * (t / self.model.tramp)
            else:
                ytarget = self.model.ymag
            SSE+=(y-ytarget)**2

        #some penalty functions if the constants are too small
        if optimizing:
            if k1<self.model.mink1 or k1>self.model.maxk1:
                SSE+=100
            if c1<10:
                SSE+=100
            if k2<self.model.mink2 or k2>self.model.maxk2:
                SSE+=100

            # I'm overlaying a gradient in the acceleration limit that scales with distance from a target squared.
            if self.model.accelMax>self.model.accelLim and self.chk_IncludeAccel.isChecked():
                # need to soften suspension
                SSE+=(self.model.accelMax-self.model.accelLim)**2
        self.model.SSE=SSE
        return SSE

    def doPlot(self):
        self.view.doPlot(self.model)
#endregion
#endregion

def main():
    QCM = CarController()
    QCM.doCalc()

if __name__ == '__main__':
    main()

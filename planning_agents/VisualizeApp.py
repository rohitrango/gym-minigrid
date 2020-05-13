from functools import partial
import numpy as np
from matplotlib import pyplot as plt
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
import pickle as pkl

from gym_minigrid.minigrid import Grid
from gym_minigrid import minigrid as M
TILE = M.TILE_PIXELS

ACTION_TO_NUM_MAPPING = {
    'Left': 0,
    'Right': 1,
    'Forward': 2,
    'Toggle': 5,
    'Done': 6,
}
ACTIONMAPPING = dict()
for k, v in ACTION_TO_NUM_MAPPING.items():
    ACTIONMAPPING[v] = k

############################################################################################################
# Part for plotter widget
############################################################################################################
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=10, height=10, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class VisualizerApp(QtWidgets.QWidget):

    """Docstring for VisualizerApp. """

    def __init__(self):
        super().__init__()
        # Create file for
        self.filename = None
        self.setWindowTitle('App')
        self.setUI()
        self.resize(500, 500)
        self.show()

    def setUI(self):
        # Set the choose pickle button
        btn = QPushButton('Choose file', self)
        btn.clicked.connect(self.openFileNameDialog)
        # Widgets for everything else
        self.trajbtnleft = QPushButton(' Prev traj ', self)
        self.trajbtnright = QPushButton(' Next traj ', self)
        self.seqbtnleft = QPushButton(' Prev obs ', self)
        self.seqbtnright = QPushButton(' Next obs ', self)
        # Add functionality
        self.trajbtnleft.clicked.connect(partial(self.changetraj, -1))
        self.trajbtnright.clicked.connect(partial(self.changetraj, 1))
        self.seqbtnleft.clicked.connect(partial(self.changestate, -1))
        self.seqbtnright.clicked.connect(partial(self.changestate, 1))
        # Add info of trajectory
        self.imagebar = MplCanvas(self)
        self.infobar = QLabel()
        self.actionbar = QLabel()
        #self.imagebar.setAlignment(QtCore.Qt.AlignCenter)
        self.actionbar.setAlignment(QtCore.Qt.AlignCenter)
        self.infobar.setAlignment(QtCore.Qt.AlignCenter)

        # Set this into a hbox inside a vbox
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(btn)

        # Set vertical box there
        vbox = QVBoxLayout()
        vbox.addStretch(1)

        # Add other widgets here
        btnbox = QHBoxLayout()
        btnbox.addWidget(self.trajbtnleft)
        btnbox.addWidget(self.trajbtnright)
        btnbox.addWidget(self.seqbtnleft)
        btnbox.addWidget(self.seqbtnright)

        vbox.addWidget(self.imagebar)
        vbox.addWidget(self.actionbar)
        vbox.addWidget(self.infobar)
        vbox.addLayout(btnbox)
        vbox.addLayout(hbox)
        # Set the layout of the app
        self.setLayout(vbox)

    def changetraj(self, delta):
        assert delta in [-1, 1]
        if delta == -1:
            if self.trajcur == 0:
                return
            self.trajcur -= 1
            self.seqcur = 0
            self.seqcount = len(self.data[self.trajcur]['act'])
            self.updateImageUI(self.trajcur, self.seqcur)
        else:
            if self.trajcur == self.trajcount - 1:
                return
            self.trajcur += 1
            self.seqcur = 0
            self.seqcount = len(self.data[self.trajcur]['act'])
            self.updateImageUI(self.trajcur, self.seqcur)

    def changestate(self, delta):
        assert delta in [-1, 1]
        if delta == -1:
            if self.seqcur == 0:
                return
            self.seqcur -= 1
            self.updateImageUI(self.trajcur, self.seqcur)
        else:
            if self.seqcur == self.seqcount - 1:
                return
            self.seqcur += 1
            self.updateImageUI(self.trajcur, self.seqcur)

    def openFileNameDialog(self):
        # Check for files opened
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Pickle files (*.pkl)", options=options)

        # Update the UI based on file name
        if fileName:
            self.filename = fileName
            with open(self.filename, 'rb') as fi:
                self.data = pkl.load(fi)
                # Trajectory ID, and sequence ID
                self.trajcount = len(self.data)
                self.trajcur = 0
                self.seqcount = len(self.data[0]['obs'])
                self.seqcur = 0
                self.updateImageUI(0, 0)


    def updateImageUI(self, trajidx, stateidx):
        traj = self.data[trajidx]
        # Given this trajectory, get observation, action, reward
        act = traj['act'][stateidx]
        obs = traj['obs'][stateidx]
        rew = traj['rew'][stateidx]
        # Get image rendering from obs

        # Given this, print action and reward
        self.actionbar.setText('Action taken: {}, Reward: {}\n'.format(ACTIONMAPPING[act], rew))
        self.infobar.setText('Trajectory {}/{} , Observation {}/{}'.format(trajidx, self.trajcount, stateidx, self.seqcount))
        img = self.get_image_from_encoded(obs).astype(int)
        self.imagebar.axes.cla()
        self.imagebar.axes.imshow(img)
        self.imagebar.draw()


    def get_image_from_encoded(self, observation):
        if isinstance(observation, dict):
            obs = observation['image']
        else:
            obs = observation
        H, W, C = obs.shape
        img = np.zeros((H*TILE, W*TILE, 3))
        # Build the image
        for i in range(H):
            for j in range(W):
                typ, col, sta = obs[i, j]
                if typ == M.OBJECT_TO_IDX['agent']:
                    obj = M.WorldObj.decode(M.OBJECT_TO_IDX['empty'], 0, 0)
                    tileimg = M.Grid.render_tile(obj, agent_dir=sta).transpose(1, 0, 2)
                else:
                    obj = M.WorldObj.decode(typ, col, sta)
                    if M.OBJECT_TO_IDX['goal'] == typ:
                        obj.color = M.IDX_TO_COLOR[col]
                    tileimg = M.Grid.render_tile(obj)
                img[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE, :] = tileimg
        #plt.imshow(img.astype(int).transpose(1, 0, 2))
        #plt.show()
        return img.transpose(1, 0, 2)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viz = VisualizerApp()
    sys.exit(app.exec_())

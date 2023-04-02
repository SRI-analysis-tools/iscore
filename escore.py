#!/usr/bin/env python3
"""
This program display polysomnography signals and allows to score epochs into
behavioral states either manually or semi automatically.
It also computes and plot statisitics of the sleep architecture
"""
# Author: Jaime Heiss
# Date created: MArch 25, 2023
# License: Free with permission

import sys
import os
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QGraphicsScene, QGraphicsPixmapItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import xgboost as xgb
import time
import pickle
import pandas as pd
# from datetime import datetime, timedelta
from sklearn.utils import class_weight
from matplotlib import pyplot as plt
import scipy.io
import scipy.io as sio
from scipy.stats import mode
from PyQt6.QtGui import QIcon, QPixmap, QImage, QKeyEvent
from PyQt6.QtCore import Qt
from collections import defaultdict
import h5py
import mne
# from xml.dom import minidom
from scipy import signal
from scipy import stats
# import cv2
import matplotlib.pyplot as plt
# import random, math
# import matplotlib.image as mpimg
import datetime
from iscore import *
from pyqtgraph import TextItem
# QCoreApplication::setAttribute(Qt::AA_DisableHighDpiScaling);
#     if (qgetenv("QT_FONT_DPI").isEmpty()) {
#         qputenv("QT_FONT_DPI", "84");
#     }
# Qt.ApplicationAttribute(Qt.AA_DisableHighDpiScaling)
# Qt.ApplicationAttribute(Qt.AA_Use96Dpi)

# import pandas as pd
# from shutil import copyfile
"""If there is EEG, traces need to by synced with EEG. If traces include dropped frames, the they should match
the N frames from the mat files, else they should match the nu,ber of detected frames form the sync signal.
A vector with the time of each frame will be generated, matching the EEG time of the first frame in every chunk.
Convert ui with pyuic5 demoLineEdit.ui -o demoLineEdit.py"""

# auxilliary functions and classes:


def getsc(x, d):
    if x in d:
        return (d[x])
    else:
        return 'U'


def getbouts(score, st):
    # Given score and state st, returns list with the duration of each bout
    index2 = np.where(np.round(score) == st)
    boutdur = []
    if len(index2) > 0:
        index2 = index2[0]
        dindex = np.diff(index2)
        # EVery time the diff is not 1, we start counting the bd of a new bout
        nbout = 1
        pos_index = 0
        while pos_index < len(dindex):
            if dindex[pos_index] == 1:
                nbout += 1
                # Add case for last bout
                if pos_index == len(dindex) - 1:
                    boutdur.append(nbout)
            else:
                boutdur.append(nbout)
                nbout = 1
                # Add case for last bout of only 1 epoch
                if pos_index == len(dindex) - 1:
                    boutdur.append(nbout)
            pos_index += 1
    return np.array(boutdur)


def getptime(score):
    # Returns the percentage of time in each state
    indw = np.where(np.round(score) == 0)
    if len(indw) > 0:
        indw = indw[0]
    indnr = np.where(np.round(score) == 1)
    if len(indnr) > 0:
        indnr = indnr[0]
    indr = np.where(np.round(score) == 2)
    if len(indr) > 0:
        indr = indr[0]
    neps = len(score)
    return len(indw) / neps, len(indnr) / neps, len(indr) / neps


def expandx(x):
    # EXpand feature vecr x to include 3 previous and next
    neps, nfeat = x.shape
    prev1 = np.ones((neps, nfeat)) * -1
    prev1[1:, :] = x[:-1, :]
    prev1[0, :] = prev1[1, :]
    prev2 = np.ones((neps, nfeat)) * -1
    prev2[2:, :] = x[:-2, :]
    prev2[0, :] = prev2[2, :]
    prev2[1, :] = prev2[2, :]
    prev3 = np.ones((neps, nfeat)) * -1
    prev3[3:, :] = x[:-3, :]
    prev3[0, :] = prev3[3, :]
    prev3[1, :] = prev3[3, :]
    prev3[2, :] = prev3[3, :]
    nextx = np.ones((neps, nfeat)) * -1
    nextx[:-1, :] = x[1:, :]
    nextx[-1, :] = nextx[-2, :]
    # Concatenate all inputs into one matrix
    return np.hstack([x, prev1, prev2, prev3, nextx])


def halve(vect):
    # Convert the vector into half of it by averaging every two values
    if len(vect) // 2 == len(vect) / 2:
        return vect.reshape(-1, 2).mean(axis=1)
    else:
        return vect[:-1].reshape(-1, 2).mean(axis=1)

# core classes


class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # specify all the controls here:
        self.ui.pushButton.clicked.connect(self.loadEDF)
        # self.ui.ImgView.ui.roiBtn.hide()
        # self.ui.ImgView.ui.histogram.hide()
        # self.ui.ImgView.ui.menuBtn.hide()
        self.ui.pushButton_3.clicked.connect(self.loadScoring)
        self.ui.pushButton_6.clicked.connect(self.saveScoring)
        self.ui.pushButton_11.clicked.connect(self.loadtextScore)
        self.ui.pushButton_9.clicked.connect(self.autoScore)
        self.ui.pushButton_13.clicked.connect(self.evaluateScore)
        self.ui.pushButton_10.clicked.connect(self.rescore)
        self.ui.pushButton_12.clicked.connect(self.stats)
        self.ui.pushButton_16.clicked.connect(self.undo)
        self.ui.pushButton_17.clicked.connect(self.plotTrace)
        self.ui.pushButton_14.clicked.connect(self.restoreEEG_scale)
        self.ui.pushButton_15.clicked.connect(self.restoreEMG_scale)
        self.ui.pushButton_19.clicked.connect(self.closeap)
        self.ui.pushButton_20.clicked.connect(self.apply_notch)
        self.ui.pushButton_22.clicked.connect(self.help)
        self.ui.horizontalScrollBar.valueChanged.connect(self.scrollbar_moved)
        self.ui.epoch_length.valueChanged.connect(self.val_ch_epl)
        # self.ui.epoch_length.editingFinished.connect(self.update_epl) not
        # working well for now
        self.ui.epoch_length_2.valueChanged.connect(self.update_tracel)
        self.ui.lineEdit.textChanged.connect(self.update_currep)
        self.ui.dateTimeEdit_4.dateTimeChanged.connect(self.update_epocht)
        self.ui.timeEdit_3.dateTimeChanged.connect(self.update_ZT0)
        self.ui.checkBox_5.stateChanged.connect(self.update_useZT)
        self.ui.checkBox_3.stateChanged.connect(self.update_fixEEG)
        self.ui.checkBox_4.stateChanged.connect(self.update_fixEMG)
        self.ui.checkBox_6.stateChanged.connect(self.update_fixFFT)
        self.ui.listWidget.itemSelectionChanged.connect(self.update_selchan)
        self.ui.lineEdit_2.editingFinished.connect(self.update_maxEEG)
        self.ui.lineEdit_3.editingFinished.connect(self.update_maxEMG)
        self.ui.EMGch.valueChanged.connect(self.update_emg)
        self.ui.listWidget_2.itemSelectionChanged.connect(self.update_emg)
        self.ui.listWidget_3.itemSelectionChanged.connect(
            self.update_timescale)
        self.show()

        # Global variables:
        self.currep = 0
        # self.manual_score = []
        self.score = []
        self.auto_score = []
        self.manual_score2s = []
        self.auto_score2s = []
        self.currentpath = ""
        self.epochl = 4
        self.tracedur = 40
        self.edfname = ""
        self.scorefile = ""
        self.selchan2p = []
        self.t0 = 0
        self.maxep = 100
        self.tstart = 0
        self.eegmat = []
        self.fixedEEG = False
        self.fixedEMG = False
        self.maxFFT = 4
        self.t = 0
        self.font = QtGui.QFont("SansSerif", 14)
        self.dictsc = {0: 'W', 0.1: 'WA', 1: 'NR', 1.1: 'NA',
                       2: 'R', 2.1: 'RA', -1: 'U', 3: 'C', 3.1: 'CA', 4: 'S'}
        self.totalp = 0
        self.scplt = None
        self.hypcrsr = None
        self.pfft = None
        self.faxis = []
        self.fftl = []
        self.fn = ''
        self.thw = [np.nan, np.nan]
        self.thnr = [np.nan, np.nan]
        self.thr = [np.nan, np.nan]
        self.thwa = [np.nan, np.nan]
        self.notch_applied=False

        # now def all the functions
    def val_ch_epl(self):
        return None

    def getbd(self, score, minbd=3):
        # mean bout duration in min of W,NR and REM
        # score is numeric where 0=W, 1=NR and 2 =R
        epochd = self.epochl
        w = getbouts(score, 0)
        nr = getbouts(score, 1)
        r = getbouts(score, 2)
        w = w[w >= minbd]
        nr = nr[nr >= minbd]
        r = r[r >= minbd]
        return np.mean(w) * epochd / 60, np.mean(nr) * \
            epochd / 60, np.mean(r) * epochd / 60

    def help(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Information)
        # msg_box.setInformativeText('Additional information can be displayed here.')
        msg_box.setWindowTitle('Iscore V 0.1 (Under development)')

        # set the detailed text with markup text
        msg_box.setText(
            '<font size="6"><html><table border="1" cellpadding="5">\
        <tr><th>Key</th><th>Action</th></tr>\
        <tr><td>right</td><td>next epoch</td></tr>\
        <tr><td>left</td><td>previous epoch</td></tr>\
        <tr><td>up</td><td>9 epochs forward</td></tr>\
        <tr><td>down</td><td>9 epochs backwards</td></tr>\
        <tr><td>n</td><td>next unscored epoch</td></tr>\
        <tr><td>q</td><td>next NREM epoch</td></tr>\
        <tr><td>r</td><td>next REM epoch</td></tr>\
        <tr><td>w</td><td>Next W epoch</td></tr>\
        <tr><td>0,M</td><td>W</td></tr>\
        <tr><td>9,O</td><td>WA</td></tr>\
        <tr><td>1,J</td><td>NR</td></tr>\
        <tr><td>7,U</td><td>NA</td></tr>\
        <tr><td>5,K</td><td>REM</td></tr>\
        <tr><td>8,I</td><td>RA</td></tr>\
        <tr><td>3,.</td><td>C</td></tr>\
        <tr><td>2,,</td><td>CA</td></tr>\
        <tr><td>4,S</td><td>S</td></tr>\
        <tr><td>6,;</td><td>U</td></tr>\
        </table></html></font>')
        msg_box.exec()
        return None
    def apply_notch(self):
        if ~self.notch_applied:
            # Apply 60 Hz notch filter to EEG and EMG
            f0 = 60.0
            Q = 100
            # Calculate the notch filter coefficients using a second-order Butterworth filter
            w0 = f0 / (self.sr / 2)
            b, a = signal.iirnotch(w0, Q)
            # Apply the filter to the signals array
            signal_array = self.edfmat[self.ui.EEGch.value() - 1, :]
            self.edfmat[self.ui.EEGch.value() - 1, :] = signal.filtfilt(b, a, signal_array)
            signal_array = self.edfmat[self.ui.EMGch.value() - 1, :]
            self.edfmat[self.ui.EMGch.value() - 1, :] = signal.filtfilt(b, a, signal_array)
            self.preprocessing()
            self.ui.label_13.setText("Notch filter applied")
            self.notch_applied=True
        return None


    def update_epl(self):
        self.ui.label_13.setText("Recalculating...")
        self.epochl = self.ui.epoch_length.value()
        self.maxep = int(self.edf.times[-1] // self.epochl)
        self.tracel = self.ui.epoch_length_2.value()
        self.edfmat = np.asarray(self.edf.get_data())
        npep = int(self.epochl * self.sr)
        leeg = len(self.edfmat[0, :])
        # self.edfmat[sch, fp:lp], pen=c[p])
        self.eegmat = self.edfmat[self.ui.EEGch.value(
        ) - 1, 0:npep * (leeg // npep)].reshape(leeg // npep, npep)
        self.fftmat = np.zeros(np.shape(self.eegmat))
        self.freqs = np.fft.fftfreq(npep, 1 / self.sr)
        self.score = -1 * np.ones(self.maxep)
        self.autoscore = np.ones(len(self.score) * int(self.epochl / 2)) * -1
        for epoch in range(leeg // npep):
            self.fftmat[epoch, :] = np.abs(np.fft.fft(self.eegmat[epoch, :]))
            # We only one the freqs between 0 and the max freq FFT

        # now finding the pos of the 0 and maxfrec
        pos0 = np.argmin(np.abs(self.freqs))
        posmax = np.argmin(np.abs(self.freqs - self.ui.maxF.value()))
        self.fftmat = self.fftmat[:, pos0:posmax]
        self.freqsp = self.freqs[pos0:posmax]

        self.neps = self.tracel // self.epochl  # epochs per page
        self.halfneps = self.neps // 2
        # current time, start date, end date, start time, end time, scroll bar
        # settings
        self.update_plots()
        newdt = time.gmtime(time.mktime(self.tstart) + self.t)

        # plotting for the first time
        histCanv = self.ui.PlotWidget_hypnogram
        histCanv.clear()
        self.scplt = histCanv.plot(self.score)
        histCanv.setYRange(-1.1, 2.2, padding=0)
        histCanv.setXRange(0, len(self.score), padding=0)
        self.hypcrsr = histCanv.plot(
            [self.currep, self.currep], [-1, 2], pen='r')
        # FFT:
        plotCanvFFT = self.ui.PlotWidget_FFT
        plotCanvFFT.clear()
        if self.currep > self.halfneps:
            if self.currep < self.maxep - self.neps:
                fft2pl = self.fftmat[self.currep -
                                     self.halfneps:self.currep +
                                     self.halfneps, :].flatten()
            else:
                fft2pl = self.fftmat[self.maxep -
                                     self.neps:self.maxep, :].flatten()
        else:
            fft2pl = self.fftmat[0:2 * self.halfneps, :].flatten()
        faxis = list(self.freqsp) * (2 * self.halfneps)
        # print(type(fft2pl))
        # print(len(fft2pl),len(faxis))
        indx = np.arange(len(fft2pl))
        self.pfft = plotCanvFFT.plot(
            indx, fft2pl, fillLevel=0, brush=(
                50, 50, 200, 100))
        plotCanvFFT.setXRange(indx[0], indx[-1], padding=0)
        if self.ui.checkBox_6.isChecked():
            ylim = self.maxFFT
        else:
            ylim = np.max(fft2pl)
        plotCanvFFT.setYRange(0, ylim, padding=0)
        indxb1 = indx[faxis == list(self.freqsp)[0]]
        for x in indxb1:
            plotCanvFFT.plot([x, x], [0, ylim], pen='k')
        ax = plotCanvFFT.getAxis('bottom')
        ticks = []
        for i in range(len(faxis)):
            if faxis[i] % 2 < 0.001:  # label only even frecs
                ticks.append('%.0f' % faxis[i])
            else:
                ticks.append("")
        xdict = dict(enumerate(ticks))
        ax.setTicks([xdict.items()])
        ax = plotCanvFFT.getAxis('left')
        ax.setTicks([])
        plotCanvFFT.setMouseEnabled(x=False, y=False)
        self.ui.label_13.setText("Ready")

    def closeap(self):
        self.close()

    def update_ZT0(self):
        pass

    def update_emg(self):
        # called upon change on list
        item = self.ui.listWidget_2.currentItem()
        self.emgtype = self.ui.listWidget_2.row(item)

        if self.emgtype == 2:
            self.emg = signal.medfilt(
                self.edfmat[self.ui.EMGch.value() - 1, :], 5)
        if self.emgtype == 0:
            self.emg = self.edfmat[self.ui.EMGch.value() - 1, :]
        if self.emgtype == 1:
            self.emg = self.edfmat[self.ui.EMGch.value() - 1, :] ** 2

    def update_fixEEG(self):
        self.fixedEEG = self.ui.checkBox_3.isChecked()
        if self.currep + self.halfneps < self.maxep:
            fp = int(
                max([0, ((self.currep * self.epochl) - (self.tracedur / 2)) * self.sr]))
        else:
            fp = int(
                np.floor(
                    (self.maxep -
                     self.neps) *
                    self.epochl *
                    self.sr))
        lp = int(np.floor(fp + self.tracedur * self.sr))
        sch = self.ui.EEGch.value() - 1  # assuming first chan is EEG
        self.maxEEG = np.max(self.edfmat[sch, fp:lp])

    def update_fixEMG(self):
        self.fixedEMG = self.ui.checkBox_4.isChecked()
        if self.currep + self.halfneps < self.maxep:
            fp = int(
                max([0, ((self.currep * self.epochl) - (self.tracedur / 2)) * self.sr]))
        else:
            fp = int(
                np.floor(
                    (self.maxep -
                     self.neps) *
                    self.epochl *
                    self.sr))
        lp = int(np.floor(fp + self.tracedur * self.sr))
        self.maxEMG = np.max(self.emg[fp:lp])

    def update_fixFFT(self):
        self.fixedFFT = self.ui.checkBox_6.isChecked()
        if self.currep > self.halfneps:
            if self.currep < self.maxep - self.neps:
                fft2pl = self.fftmat[self.currep -
                                     self.halfneps:self.currep +
                                     self.halfneps, :].flatten()
            else:
                fft2pl = self.fftmat[self.maxep -
                                     self.neps:self.maxep, :].flatten()
        else:
            fft2pl = self.fftmat[0:2 * self.halfneps, :].flatten()
        self.maxFFT = np.max(fft2pl)

    def update_useZT(self):
        self.useZT = self.ui.checkBox_5.isChecked()
        self.update_plots()

    def update_currep(self):
        self.currep = int(float(self.ui.lineEdit.text()))
        self.update_epocht()
        # update time and call updteplot

    def scrollbar_moved(self):
        v = self.ui.horizontalScrollBar.value()
        self.currep = int((v / 1000) * self.maxep)
        self.ui.lineEdit.setText(str(self.currep))
        self.update_epocht()

    def update_timescale(self):
        pass

    def update_selchan(self):
        self.selchan2p = [self.ui.listWidget.row(
            item) for item in self.ui.listWidget.selectedItems()]
        if len(self.selchan2p) > 0:
            self.update_plots()

    def update_epocht(self):
        self.t = self.epochl * self.currep
        # newdt = time.gmtime(time.mktime(self.tstart) + self.t)
        newdt = time.gmtime(self.tstart.timestamp() + self.t)
        self.ui.dateTimeEdit_4.setDateTime(
            datetime.datetime.fromtimestamp(
                time.mktime(newdt)))
        self.update_plots()

    def update_maxEEG(self):
        pass

    def update_maxEMG(self):
        pass

    def loadScoring(self):
        filelist2 = os.listdir()
        print(self.fn)
        fn = self.fn[:-4] + '.pkl'
        for item in filelist2:
            if item == self.fn[:-4] + '.pkl':
                with open(fn, 'rb') as handle:
                    print('loading scoring...')
                    self.scoredict = pickle.load(handle)
                    self.score = self.scoredict['score']
                break
# save(fn, 'sc', 'vali', 'startp', 'endp', 'zt0', 'scalefft', 'rangefft',
# 'epocl', 't0');endswith(".scr"):
        else:
            fileName = QFileDialog.getOpenFileName(
                self, 'Open scoring file', '', "scr files (*.scr), mat files (*.mat)")
            self.scorefile = fileName[0]
            if self.scorefile.endswith('.mat'):
                arrays = {}
                try:
                    f = sio.loadmat(self.scorefile)
                    for k, v in f.items():
                        arrays[k] = np.array(v)
                    m = arrays['zt0'].flatten()
                    self.scoredict['zt0'] = m[0]
                except NotImplementedError:
                    f = h5py.File(self.scorefile)
                    for k, v in f.items():
                        arrays[k] = np.array(v)
                    m = arrays['zt0'].flatten()
                    self.scoredict['zt0'] = ''.join([str(chr(c)) for c in m])
                    # if 't0' in arrays.keys():
                    #     m = arrays['t0'].flatten()
                    #     self.scoredict['t0'] = ''.join([str(chr(c)) for c in m])

                except BaseException:
                    ValueError('could not read scoring mat file...')

                self.scoredict['score'] = arrays['sc'].flatten()
                self.scoredict['el'] = arrays['epocl'].flatten()[0]
                for i, s in enumerate(self.scoredict['score']):
                    if not np.isnan(s):
                        self.score[i] = s
            else:
                if len(self.scorefile) >= 1:
                    f = open(self.scorefile, 'r')
                    lines = f.read().split()
                    f.close()
                    for linen in range(len(lines)):
                        if linen == 0:
                            self.epochl = int(lines[linen])
                        elif linen == 1:
                            edfstart = time.strptime(
                                lines[linen], '%d_%m_%Y_%H:%M:%S')
                            self.ui.dateTimeEdit_4.setValue(
                                edfstart)  # check this
                        else:
                            self.manual_score.append(lines[linen])
        self.update_plots()

    def update_autoscore(self):  # copies score into autoscore with 2 s epochs
        ind = np.arange(0, len(self.autoscore), int(self.epochl / 2))
        for n in range(int(self.epochl / 2)):
            self.autoscore[ind + n] = self.score

    def saveScoring(self):
        fn = self.fn[:-4] + '.pkl'
        print(fn)
        self.scoredict['score'] = self.score
        # add automatic update of dict if zto, t0 or epoch len changes
        with open(fn, 'wb') as handle:
            pickle.dump(
                self.scoredict,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
        print('pickled!')
        print(self.scoredict)
        f = open(fn[:-4] + '.txt', 'w')
        for s in self.dictsc['score']:
            if s in list(self.dict.keys()):
                f.write(dictslp[s] + '\n')
            else:
                f.write('' + '\n')
        f.close()

    def loadtextScore(self):
        return None

    def autoScore(self):
        # Compute featues in 2s epochs: RMS of EEG and EMG and EEG power
        # Use gamma and EMG to identify clear sleep and wake epochs
        # Trains XGBoost on easy cases and scores the rest
        # Identifies wake artifacts as high EMG, high Delta
        # Converts 2s score into 4s score adding artifact epochs
        # bands:0-4, 6-10, 125-135, 285-295
        # Adding 2s epoch FFT
        self.ui.label_13.setText("Computing Features...")
        np2sep = int(np.floor(2 * self.sr))  # Number of points in 2 s epochs
        eegmat2s = self.edfmat[self.ui.EEGch.value() -
                               1, 0:np2sep *int(self.epochl/2)*len(self.score)].reshape(-1,np2sep)
                            
        emgmat2s = self.edfmat[self.ui.EMGch.value() -
                               1, 0:np2sep *int(self.epochl/2)*len(self.score)].reshape(-1, np2sep)
        #    (self.edfmat.shape[1] //
        #     np2sep)].reshape(self.edfmat.shape[1] //
        #                      np2sep, np2sep)
        freqs2s = np.fft.fftfreq(np2sep, 1 / self.sr)
        indx = np.where(freqs2s >= 0)[0]
        fftmat2s = np.zeros((eegmat2s.shape[0], len(indx)))
        freqs2s = freqs2s[indx]
        for epoch in range(len(self.score)*int(self.epochl/2)):
            fftmat2s[epoch, :] = np.abs(np.fft.fft(eegmat2s[epoch, :]))[indx]
        # GEt features:
        # 1- Power bands
        g1i = 125
        g1f = 135
        g2i = 285
        g2f = 295
        if self.sr < 600:
            g1i = 30
            g1f = 100
            g2i = min([150, (self.sr / 2) - 20])
            g2f = min([170, (self.sr / 2)])
        print(f'Gamma1:{g1i}-{g1f}')
        print(f'Gamma2:{g2i}-{g2f}')
        deltafr = np.where(freqs2s < 4)[0]
        gamma1fr = np.where((freqs2s >= g1i) & (freqs2s <= g1f))[0]
        gamma2fr = np.where((freqs2s >= g2i) & (freqs2s <= g2f))[0]
        thetafr = np.where((freqs2s >= 6) & (freqs2s <= 10))[0]
        self.delta = np.mean(fftmat2s[:, deltafr], axis=1)
        self.gamma1 = np.mean(fftmat2s[:, gamma1fr], axis=1)
        self.gamma2 = np.mean(fftmat2s[:, gamma2fr], axis=1)
        self.theta = np.mean(fftmat2s[:, thetafr], axis=1)
        self.delta[self.delta < 1E-10] = 1E-10
        self.thetad = self.theta / self.delta
        # 2- RMS value
        self.eegrms = np.zeros(eegmat2s.shape[0])
        self.emgrms = np.zeros(eegmat2s.shape[0])
        for e in range(eegmat2s.shape[0]):
            self.eegrms[e] = np.sqrt(np.mean(eegmat2s[e, :]**2))
            self.emgrms[e] = np.sqrt(np.mean(emgmat2s[e, :]**2))
        self.ui.label_13.setText("Autoscoring...")
        self.ui.label_13.setStyleSheet("color: green;")
        self.train_mode = True
        self.autoscore = np.ones(len(self.delta)) * -1
        # prevent zeros
        self.emgrms[self.emgrms < 1E-10] = 1E-10
        self.eegrms[self.eegrms < 1E-10] = 1E-10
        self.delta[self.delta < 1E-10] = 1E-10

        # Index for each state, including wake artifact
        self.indnr = (self.delta**3) / \
            (self.emgrms * self.gamma1 * self.thetad)
        self.indw = self.emgrms * self.gamma2 / self.eegrms
        self.indwa = self.emgrms * self.gamma1 * self.delta**2
        self.indrem = self.theta * (self.thetad**3) / self.emgrms

        if self.ui.checkBox_2.isChecked():  # Full auto.
            # Requires >4% of the epochs to be REM, 2% to be WA and 20% to be
            # NR and W

            nreps = np.where(self.indnr > np.percentile(self.indnr, 80))[0]
            self.autoscore[nreps] = 1
            weps = np.where(self.indw > np.percentile(self.indw, 80))[0]
            self.autoscore[weps] = 0

            # getting represetative values for delta and mt
            deltasleep = np.mean(self.delta[nreps])
            mtw = np.mean(self.emgrms[weps])
            # Using the values for filtering WA
            filt_delta = np.ones(len(self.indwa))
            filt_delta[self.delta < deltasleep] = 0
            filt_notdelta = np.ones(len(self.indrem))
            filt_notdelta[self.delta > deltasleep] = 0
            filt_mt = np.ones(len(self.indwa))
            filt_mt[self.emgrms < mtw] = 0
            filt_notmt = np.ones(len(self.indrem))
            filt_notmt[self.emgrms > mtw] = 0
            self.indwa = self.indwa * filt_delta * filt_mt
            self.indrem = self.indrem * filt_notdelta * filt_notmt
            reps = np.where(self.indrem > np.percentile(self.indrem, 96))[0]
            self.autoscore[reps] = 2
            mtrem = np.mean(self.emgrms[reps])
            deltarem = np.mean(self.delta[reps])
            tethar = np.mean(self.theta[reps])
            waeps = np.where(self.indwa > np.percentile(self.indwa, 98))[0]
            # neeed to be an int for now. Will be converted to 0.1
            self.autoscore[waeps] = 0.1
        else:  # Ask user to score examples based on the index values for each state
            # Start with the middle value and go up or down until the state changes
            # Initialize training examples
            # We need to find two thresholds for each index, one that gurantees
            # presence and one that gurantees absence

            # self.evaluateScore()
            None

        indtrain = np.where(self.autoscore >= 0)[0]
        ind_test = np.where(self.autoscore < 0)[0]

        x_train = np.vstack([self.delta[indtrain],
                             self.theta[indtrain],
                             self.thetad[indtrain],
                             self.gamma1[indtrain],
                             self.gamma2[indtrain],
                             self.eegrms[indtrain],
                             self.emgrms[indtrain]]).T
        x_test = np.vstack([self.delta[ind_test],
                            self.theta[ind_test],
                            self.thetad[ind_test],
                            self.gamma1[ind_test],
                            self.gamma2[ind_test],
                            self.eegrms[ind_test],
                            self.emgrms[ind_test]]).T
        y_train = self.autoscore[indtrain]
        # Add two previous and one next
        x_train = expandx(x_train)
        x_test = expandx(x_test)

        # Train ML
        # Define the XGBoost model
        y_train[y_train == 0.1] = 3

        classes_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )

        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(set(y_train)),
            learning_rate=0.1,
            n_estimators=100,
            max_depth=3,
            min_child_weight=1,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42)
        # Train the model

        xgb_model.fit(x_train, y_train, sample_weight=classes_weights)
        # predict the remaining epochs
        y_pred = xgb_model.predict(x_test)
        y_pred[y_pred == 3] = 0.1
        self.autoscore[ind_test] = y_pred

        # Now convert autoscore to 4s epochs and add artifacts for mixed scores
        if len(self.autoscore) // 2 != len(self.autoscore) / 2:
            # Trim last epoch if it is not even
            self.autoscore = self.autoscore[:-1]
        autos = self.autoscore.reshape(-1, 2)
        print('autos shape:',autos.shape,' score shape:',self.score.shape)
        for e in range(autos.shape[0] - 1):
            if np.min(autos[e, :]) >= 0:
                # If any one is WA, score as WA
                if np.max(autos[e, :] == 0.1):
                    self.score[e] = 0.1
                else:
                    if np.round(autos[e, 0]) == np.round(autos[e, 1]):
                        self.score[e] = np.max(autos[e, :])
                    else:
                        if e > 0:
                            if np.max(autos[e, :] == 3):
                                self.score[e] = 3
                            else:
                                # If one of the pair have same score of
                                # previous epoch, score as previous epoch with
                                # artifact
                                if autos[e, 0] == np.round(self.score[e - 1]):
                                    self.score[e] = np.round(
                                        self.score[e - 1]) + 0.1
                                elif autos[e, 1] == np.round(self.score[e - 1]):
                                    self.score[e] = np.round(
                                        self.score[e - 1]) + 0.1
                                elif e < len(self.score) - 1:
                                    if autos[e, 0] == np.round(
                                            self.score[e + 1]):
                                        self.score[e] = np.round(
                                            self.score[e + 1]) + 0.1
                                    elif autos[e, 1] == np.round(self.score[e + 1]):
                                        self.score[e] = np.round(
                                            self.score[e + 1]) + 0.1
                                else:
                                    self.score[e] = np.round(
                                        self.score[e - 1]) + 0.1
                        else:
                            self.score[e] = mode(np.hstack(
                                [autos[e, :], autos[e + 1, :]]),keepdims=False)[0][0]
        # replace all NR that have low delta and high tethad by R if there is a R epoch nearby
        # Convert 2s vectors into 4s
        delta4 = halve(self.delta)
        emg4 = halve(self.emgrms)
        tetha4 = halve(self.delta)
        newindr = np.where((np.round(self.score) == 1) | (self.score < 0))[0]
        mtrem = 1.4 * np.mean(self.emgrms[reps])
        deltarem = 1.4 * np.mean(self.delta[reps])
        tethar = 0.5 * np.mean(self.theta[reps])
        for e in newindr:
            if np.max(
                    np.round(self.score[max([0, e - 5]):min([e + 5, len(self.score) - 1])]) == 2):
                if (delta4[e] < deltarem) and (
                        emg4[e] < mtrem) and (tetha4[e] > tethar):
                    self.score[e] = 2
        # fix remaining unscored epochs
        unsc = np.where(self.score < 0)
        if len(unsc) > 0:
            for e in unsc[0]:
                ne = e.copy()
                while (ne > 0) and (self.score[ne] < 0):
                    ne -= 1
                self.score[e] = self.score[ne]
        # Replace epochs flanked by REM into REM
        for ep in range(1, len(self.score) - 1):
            if (round(self.score[ep - 1]) ==
                    2) and (round(self.score[ep + 1]) == 2):
                self.score[ep] = 2

        self.update_plots()
        return None

    # Ask for examples and find lower and upper threshold of each state index
    def evaluateScore(self):
        # as half the distance between current percentile and max value (or min)
        # setting W thresholds
        if np.isnan(np.sum(self.thw)):
            if np.isnan(self.thwa[0]):  # lower threshold
                # If last score was W, reduce percentile and ask again
                if self.score[self.currep] == 0:
                    if self.cper == 0:
                        # All epochs w
                        self.thw[0] = 0
                        self.cper = 50
                    elif self.cper > 1:
                        self.cper = self.cper / 2
                    else:
                        self.cper = 0
                    indp = np.where(self.indw > np.percentile(self.indw, self.cper))[
                        0]  # Index of the values above cper
                    ce = indp[np.argmin(self.indw[indp])]
                    self.ui.label_13.setText(
                        "Please score current epoch\n and add to train")
                    self.ui.lineEdit.setText(str(int(ce)))
                    self.update_currep()
                else:  # Set lower threshold
                    self.thw[0] = np.percentile(self.indw, self.cper / 2)
                    self.cper = 50
            else:
                # If last score was not W, increase percentile and ask again
                if self.score[self.currep] > 0.5:
                    if self.cper == 100:
                        # No W epochs
                        self.thw[1] = np.max(self.indw) + 1
                        self.cper = 50

                    elif self.cper > 1:
                        self.cper += ((100 - self.cper) / 2)
                    else:
                        self.cper = 100
                    indp = np.where(
                        self.indw > np.percentile(
                            self.indw, self.cper))[0]
                    ce = indp[np.argmin(self.indw[indp])]
                    self.ui.label_13.setText(
                        "Please score current epoch \n and add to train")
                    self.ui.lineEdit.setText(str(int(ce)))
                    self.update_currep()
                else:  # Set upper threshold
                    self.thw[1] = np.percentile(
                        self.indw, (100 - self.cper) / 2)
                    self.cper = 50
        # setting NR thresholds
        if np.isnan(np.sum(self.thnr)):
            if np.isnan(self.thnr[0]):  # lower threshold
                # If last score was NR, reduce percentile and ask again
                if self.score[self.currep] == 1:
                    if self.cper == 0:
                        # All epochs NR
                        self.thnr[0] = 0
                        self.cper = 50

                    elif self.cper > 1:
                        self.cper = self.cper / 2
                    else:
                        self.cper = 0
                    indp = np.where(
                        self.indnr > np.percentile(
                            self.indnr, self.cper))[0]
                    ce = indp[np.argmin(self.indnr[indp])]
                    self.ui.label_13.setText("Please score current epoch")
                    self.ui.lineEdit.setText(str(int(ce)))
                    self.update_currep()
                else:  # Set lower threshold
                    self.thnr[0] = np.percentile(self.indnr, self.cper / 2)
                    self.cper = 50
            else:
                # If last score was not NR, increase percentile and ask again
                if np.round(self.score[self.currep]) != 1:
                    if self.cper == 100:
                        # No NR epochs
                        self.thnr[1] = np.max(self.indw) + 1
                        self.cper = 50
                    elif self.cper > 1:
                        self.cper += ((100 - self.cper) / 2)
                    else:
                        self.cper = 100
                    indp = np.where(
                        self.indnr > np.percentile(
                            self.indnr, self.cper))[0]
                    ce = indp[np.argmin(self.indnr[indp])]
                    self.ui.label_13.setText("Please score current epoch")
                    self.ui.lineEdit.setText(str(int(ce)))
                    self.update_currep()
                else:  # Set upper threshold
                    self.thnr[1] = np.percentile(
                        self.indnr, (100 - self.cper) / 2)
                    self.cper = 50
        # Adjust index of WA and R based on W and NR:
        self.autoscore = np.ones(len(self.score) * 2) * -1
        for e in range(len(self.score)):
            self.autoscore[2 * e] = self.score[e]
            self.autoscore[2 * e + 1] = self.score[e]
        epw = np.where(self.indw > self.thw[1])[0]
        epnr = np.where(self.indnr > self.thnr[1])[0]
        self.autoscorep[epw] = 0
        self.autoscore[epnr] = 1
        # getting represetative values for delta and mt
        deltasleep = np.mean(self.delta[epnr])
        mtw = np.mean(self.emgrms[epw])
        # Using the values for filtering WA
        filt_delta = np.ones(len(self.indwa))
        filt_delta[self.delta < deltasleep] = 0
        filt_notdelta = np.ones(len(self.indrem))
        filt_notdelta[self.delta > deltasleep] = 0
        filt_mt = np.ones(len(self.indwa))
        filt_mt[self.emgrms < mtw] = 0
        filt_notmt = np.ones(len(indrem))
        filt_notmt[self.emgrms > mtw] = 0
        self.indwa = self.indwa * filt_delta * filt_mt
        self.indrem = self.indrem * filt_notdelta * filt_notmt

        if np.isnan(np.sum(self.thwa)):  # setting WA thresholds
            if np.isnan(self.thwa[0]):  # lower threshold
                # If last score was W, reduce percentile and ask again
                if self.score[self.currep] == 0.1:
                    if self.cper == 0:
                        # All epochs wa
                        self.thwa[0] = 0
                        self.cper = 50
                    elif self.cper > 1:
                        self.cper = self.cper / 2
                    else:
                        self.cper = 0
                    indp = np.where(
                        self.indwa > np.percentile(
                            self.indwa, self.cper))[0]
                    ce = indp[np.argmin(self.indwa[indp])]
                    self.ui.label_13.setText("Please score current epoch")
                    self.ui.lineEdit.setText(str(int(ce)))
                    self.update_currep()
                else:  # Set lower threshold
                    self.thwa[0] = np.percentile(self.indwa, self.cper / 2)
                    self.cper = 50
            else:
                # If last score was not WA, increase percentile and ask again
                if self.score[self.currep] != 0.1:
                    if self.cper == 100:
                        # No WA epochs
                        self.thwa[1] = np.max(self.indwa) + 1
                        self.cper = 50
                    elif self.cper > 1:
                        self.cper += ((100 - self.cper) / 2)
                    else:
                        self.cper = 100
                    indp = np.where(
                        self.indwa > np.percentile(
                            self.indwa, self.cper))[0]
                    ce = indp[np.argmin(self.indwa[indp])]
                    self.ui.label_13.setText("Please score current epoch")
                    self.ui.lineEdit.setText(str(int(ce)))
                    self.update_currep()
                else:  # Set upper threshold
                    self.thwa[1] = np.percentile(
                        self.indwa, (100 - self.cper) / 2)
                    self.cper = 50

        if np.isnan(np.sum(self.threm)):  # setting REM thresholds
            if np.isnan(self.threm[0]):  # lower threshold
                if np.round(
                        self.score[self.currep]) == 2:  # If last score was R, reduce percentile and ask again
                    if self.cper == 0:
                        # All epochs rem
                        self.thr[0] = 0
                        self.cper = 50
                    elif self.cper > 1:
                        self.cper = self.cper / 2
                    else:
                        self.cper = 0
                    indp = np.where(
                        self.indrem > np.percentile(
                            self.indrem, self.cper))[0]
                    ce = indp[np.argmin(self.indrem[indp])]
                    self.ui.label_13.setText("Please score current epoch")
                    self.ui.lineEdit.setText(str(int(ce)))
                    self.update_currep()
                else:  # Set lower threshold
                    self.thr[0] = np.percentile(self.indrem, self.cper / 2)
                    self.cper = 50
            else:
                # If last score was not WA, increase percentile and ask again
                if self.score[self.currep] != 2:
                    if self.cper == 100:
                        # No WA epochs
                        self.thwr[1] = np.max(self.indrem) + 1
                        self.cper = 50
                    elif self.cper > 1:
                        self.cper += ((100 - self.cper) / 2)
                    else:
                        self.cper = 100
                    indp = np.where(
                        self.indrem > np.percentile(
                            self.indrem, self.cper))[0]
                    ce = indp[np.argmin(self.indrem[indp])]
                    self.ui.label_13.setText("Please score current epoch")
                    self.ui.lineEdit.setText(str(int(ce)))
                    self.update_currep()
                else:  # Set upper threshold
                    self.thr[1] = np.percentile(
                        self.indrem, (100 - self.cper) / 2)
                    self.cper = 50

            ce = np.where(self.indw > np.percentile(indw, 50))[0]
            self.ui.label_13.setText("Please score current epoch")
            self.ui.lineEdit.setText(str(int(ce)))
            self.update_currep()
        return None

    def rescore(self):
        return None

    def stats(self):
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams.update({'font.size': 14})
        # Displays mean fft per state ,mean BD, % time in each state and save
        # csv file with score and power every 2 Hz
        indw = np.where(self.score == 0)
        indnr = np.where(self.score == 1)
        indr = np.where(self.score == 2)
        # First plot log FFT per state across all freqs
        plt.figure()
        if len(indw) > 0:
            indw = indw[0]
            fftw = np.mean(self.fftstats[indw, :], axis=0)
            plt.plot(self.freqstats, np.log(fftw), 'g-', label='W')
        else:
            indw = []
        if len(indnr) > 0:
            indnr = indnr[0]
            fftnr = np.mean(self.fftstats[indnr, :], axis=0)
            plt.plot(self.freqstats, np.log(fftnr), 'b-', label='NR')
        else:
            indnr = []
        if len(indr) > 0:
            indr = indr[0]
            fftr = np.mean(self.fftstats[indr, :], axis=0)
            plt.plot(self.freqstats, np.log(fftr), 'r-', label='REM')
        else:
            indnr = []
        plt.ylabel('Log(|FFT|) (Log $V$)', fontsize=15)
        plt.xlabel('Frequency (Hz)', fontsize=15)
        plt.title('FFT by state', fontsize=18)
        plt.legend()
        plt.show()
        # Plot again for selected freqs in linear scale
        plt.figure(figsize=(15, 11))
        plt.subplot(2, 1, 1)
        if len(indw) > 0:
            fftw = np.mean(self.fftmat[indw, :], axis=0)
            plt.plot(self.freqsp, fftw, '#50C878', label='W')
        if len(indnr) > 0:
            fftnr = np.mean(self.fftmat[indnr, :], axis=0)
            plt.plot(self.freqsp, fftnr, '#4169E1', label='NR')
        if len(indr) > 0:
            fftr = np.mean(self.fftmat[indr, :], axis=0)
            plt.plot(self.freqsp, fftr, '#FF6347', label='REM')
        plt.ylabel(' |FFT| ($V$)', fontsize=15)
        plt.xlabel('Frequency (Hz)', fontsize=15)
        plt.title('FFT by state', fontsize=18)
        plt.legend()
        # now BD and PT
        bd = self.getbd(self.score)
        pt = getptime(self.score)
        plt.subplot(2, 2, 3)
        x_labels = ['WAKE', 'NREM', 'REM']
        # colors=['green', 'blue', 'red']
        colors = ['#50C878', '#4169E1', '#FF6347']
        plt.bar(x_labels, bd, color=colors)
        ax2 = plt.gca()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        plt.title('Mean bout duration', fontsize=19)
        plt.ylabel('Duration (Min)', fontsize=15)
        plt.subplot(2, 2, 4)
        x_labels = ['WAKE', 'NREM', 'REM']
        plt.pie(pt, labels=x_labels, colors=colors, autopct='%1.1f%%')
        plt.title('Percentage of time', fontsize=19)
        plt.tight_layout()
        plt.show()
        # Save csv file with epoch num, Zt time, time, and FFT for each freq in
        # 2 Hz bins
        # set tick values to the x-coordinates of the data points
        tick_values = np.arange(self.t0[0], self.t0[-1], 4).astype(int)
        startt = time.gmtime(self.tstart.timestamp())
        inits = startt.tm_hour * 3600 + startt.tm_min * 60 + startt.tm_sec
        difsec = inits
        self.z0 = self.ui.timeEdit_3.time()
        z0secs = self.z0.hour() * 3600 + self.z0.minute() * 60 + self.z0.second()
        difseczt0 = inits - z0secs
        if difseczt0 < 0:
            difseczt0 = 24 * 3600 + difseczt0
        tickszt0 = [QtCore.QTime(int(np.floor(t / 3600)), int((t % 3600) // 60), int(
            (t % 3600) % 60)).addSecs(difseczt0).toString("hh:mm:ss") for t in tick_values]
        ticks = [QtCore.QTime(int(np.floor(t / 3600)), int((t %
                                                            3600) // 60), int((t %
                                                                               3600) %
                                                                              60)).addSecs(difsec).toString("hh:mm:ss") for t in tick_values]
        neps = len(self.score)
        if len(ticks) < neps:
            # removing excess epochs
            self.score = self.score[:len(ticks)]
            neps = len(self.score)
        cols = ['Time', 'ZT', 'Epoch', 'Score']
        freqs = [
            str(f) for f in np.arange(
                1, np.floor(
                    np.max(
                        self.freqstats)))]
        ffts = pd.DataFrame(columns=cols + self.freqsp)
        ffts['Epoch'] = 1 + np.arange(len(self.score))
        ffts['Time'] = ticks[0:neps]
        ffts['ZT'] = tickszt0[:neps]
        ffts['Score'] = pd.Series(self.score).apply(lambda x: self.dictsc[x])
        for i, f in enumerate(np.arange(1, np.floor(np.max(self.freqstats)))):
            f0 = int(np.where(self.freqstats > f - 1)[0][0])
            ff = int(np.where(self.freqstats > f)[0][0])
            ffts[freqs[i]] = np.mean(self.fftstats[:neps, f0:ff], axis=1)
        ffts.to_csv(self.fn[:-4] + '.csv', index=False)
        print(self.fn[:-4] + '.csv Saved')

        # Now show sleep arch for each ZT hour: for each state: N bouts, %
        # time, BD, mean power

        statsdic = {
            'ZT': [],
            'WBD': [],
            'NRBD': [],
            'RBD': [],
            'NW': [],
            'NNR': [],
            'NR': [],
            'PTW': [],
            'PTNR': [],
            'PTR': [],
            'PW': [],
            'PNR': [],
            'PR': []}
        # Find first and last epoch within integer part of ZT
        mincol = ffts['ZT'].apply(lambda x: x[3:5]).values
        fe = np.where(mincol == '00')[0][0]  # first epoch
        nepsph = 3600 // self.epochl
        nh = (neps - fe) // nepsph  # Number of hours
        le = fe + nh * nepsph  # last epoch
        # Compute stats for every ZT hour
        for ep in np.arange(fe, le + 1, nepsph):
            indw = np.where(self.score[ep:ep + nepsph] == 0)
            indnr = np.where(self.score[ep:ep + nepsph] == 1)
            indr = np.where(self.score[ep:ep + nepsph] == 2)
            if len(indw) > 0:
                indw = indw[0]
                fftw = np.mean(self.fftstats[indw, :], axis=0)
            else:
                indw = []
                fftw = np.zeros(self.fftstats.shape[1])
            if len(indnr) > 0:
                indnr = indnr[0]
                fftnr = np.mean(self.fftstats[indnr, :], axis=0)
            else:
                indnr = []
                fftnr = np.zeros(self.fftstats.shape[1])
            if len(indr) > 0:
                indr = indr[0]
                fftr = np.mean(self.fftstats[indr, :], axis=0)
            else:
                indnr = []
                fftr = np.zeros(self.fftstats.shape[1])
            statsdic['ZT'].append(ffts['ZT'].values[ep + 1][0:2])  # ZT hour
            statsdic['PW'].append(fftw)
            statsdic['PNR'].append(fftnr)
            statsdic['PR'].append(fftr)
            bd = self.getbd(self.score[ep:ep + nepsph])
            pt = getptime(self.score[ep:ep + nepsph])
            statsdic['WBD'].append(bd[0])
            statsdic['NRBD'].append(bd[1])
            statsdic['RBD'].append(bd[2])
            statsdic['PTW'].append(pt[0])
            statsdic['PTNR'].append(pt[1])
            statsdic['PTR'].append(pt[2])
            w = getbouts(self.score[ep:ep + nepsph], 0)
            w = w[w >= 3]
            statsdic['NW'].append(len(w))
            nr = getbouts(self.score[ep:ep + nepsph], 1)
            nr = nr[nr >= 3]
            statsdic['NNR'].append(len(nr))
            r = getbouts(self.score[ep:ep + nepsph], 2)
            r = r[r >= 3]
            statsdic['NR'].append(len(r))
        hourlysts = pd.DataFrame(statsdic)
        # Save and plot
        hourlysts.to_csv(self.fn[:-4] + 'hrly_stats.csv', index=False)
        print(self.fn[:-4] + 'hrly_stats.csv Saved')
        # For frequency bands, a baseline power needs to be computed so
        # skipping that
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.plot(
            hourlysts['ZT'].values,
            hourlysts['WBD'].values,
            'o-',
            color='#50C878',
            label='WAKE')
        plt.plot(
            hourlysts['ZT'].values,
            hourlysts['NRBD'].values,
            'o-',
            color='#4169E1',
            label='NREM')
        plt.plot(
            hourlysts['ZT'].values,
            hourlysts['RBD'].values,
            'o-',
            color='#FF6347',
            label='REM')
        plt.title('Bout Duration', fontsize=19)
        plt.xlabel('ZT', fontsize=15)
        plt.ylabel('Duration (Min.)', fontsize=15)
        plt.subplot(1, 3, 2)
        plt.plot(
            hourlysts['ZT'].values,
            hourlysts['PTW'].values,
            'o-',
            color='#50C878',
            label='WAKE')
        plt.plot(
            hourlysts['ZT'].values,
            hourlysts['PTNR'].values,
            'o-',
            color='#4169E1',
            label='NREM')
        plt.plot(
            hourlysts['ZT'].values,
            hourlysts['PTR'].values,
            'o-',
            color='#FF6347',
            label='REM')
        plt.title('Percentage Time', fontsize=19)
        plt.xlabel('ZT', fontsize=15)
        plt.ylabel('(%)', fontsize=15)
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(
            hourlysts['ZT'].values,
            hourlysts['NW'].values,
            'o-',
            color='#50C878',
            label='WAKE')
        plt.plot(
            hourlysts['ZT'].values,
            hourlysts['NNR'].values,
            'o-',
            color='#4169E1',
            label='NREM')
        plt.plot(
            hourlysts['ZT'].values,
            hourlysts['NR'].values,
            'o-',
            color='#FF6347',
            label='REM')
        plt.title('Number of Bouts', fontsize=19)
        plt.xlabel('ZT', fontsize=15)
        plt.ylabel('N', fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return None

    def undo(self):
        return None

    def plotTrace(self):
        plt.figure()
        plt.subplot(6, 1, 1)
        plt.plot(self.delta)
        plt.ylabel('Delta')
        plt.subplot(6, 1, 2)
        plt.plot(self.gamma1)
        plt.ylabel('$\\gamma$1')
        plt.subplot(6, 1, 3)
        plt.plot(self.gamma2)
        plt.ylabel('$\\gamma$2')
        plt.subplot(6, 1, 4)
        plt.plot(self.theta)
        plt.ylabel('$\\theta$')
        plt.subplot(6, 1, 5)
        plt.plot(self.eegrms)
        plt.ylabel('EEGrms')
        plt.subplot(6, 1, 6)
        plt.plot(self.emgrms)
        plt.ylabel('EMGrms')
        # Plots mean power for bands
        print()
        plt.figure()
        plt.plot(self.gamma2, self.emgrms, 'k.', alpha=0.4)
        plt.ylabel('EMG', fontsize=15)
        plt.xlabel('Gamma 285-295 Hz', fontsize=15)
        plt.title(f'r={np.round(np.corrcoef(self.gamma2,self.emgrms)[0,1],2)}')
        plt.figure()
        deltafr = np.where(self.freqstats < 4)[0]
        gammafr = np.where((self.freqstats > 30) & (self.freqstats < 59))[0]
        gammarem = np.where(
            (self.freqstats >= 125) & (
                self.freqstats <= 135))[0]
        gammaw = np.where((self.freqstats >= 285) & (self.freqstats <= 295))[0]
        window_size = 15
        window = np.ones(int(window_size)) / float(window_size)

        delta = np.mean(self.fftstats[:, deltafr], axis=1)
        gamma = signal.medfilt(
            np.mean(self.fftstats[:, gammafr], axis=1))
        gammarem = np.mean(self.fftstats[:, gammarem], axis=1)
        gammaw = np.mean(self.fftstats[:, gammaw], axis=1)
        plt.subplot(4, 1, 1)
        plt.plot(np.convolve(delta, window, 'same'), 'k-', linewidth=0.5)
        plt.ylim(0, np.percentile(delta, 95))
        plt.ylabel('$\\delta$', fontsize=16)
        plt.subplot(4, 1, 2)
        plt.plot(np.convolve(gamma, window, 'same'), 'k-', linewidth=0.5)
        plt.ylim(0, np.percentile(gamma, 95))
        plt.ylabel('$\\gamma$', fontsize=16)
        plt.subplot(4, 1, 3)
        plt.plot(np.convolve(gammarem, window, 'same'), 'k-', linewidth=0.5)
        plt.ylim(0, np.percentile(gammarem, 95))
        plt.ylabel('$\\gamma$_Rem', fontsize=16)
        plt.subplot(4, 1, 4)
        plt.plot(np.convolve(gammaw, window, 'same'), 'k-', linewidth=0.5)
        plt.ylim(0, np.percentile(gammaw, 95))
        plt.ylabel('$\\gamma$_W', fontsize=16)
        plt.show()
        return None

    def update_tracel(self):
        self.tracel = self.ui.epoch_length_2.value()
        self.update_plots()

    def restoreEEG_scale(self):
        return None

    def restoreEMG_scale(self):
        return None

    def update_plots(self):
        # plots data according to : current epoch,
        # epoch length, trace length, channels 2 plot, channel for EMG, EMG
        # signal type, timescale type, ZT0, EEg scaling, EEG scale, EMG
        # scaling, EMG scales,
        histCanv = self.ui.PlotWidget_hypnogram
        histCanv.removeItem(self.scplt)
        histCanv.removeItem(self.hypcrsr)
        self.scplt = histCanv.plot(self.score, pen='k')
        histCanv.setYRange(-1.1, 2.2, padding=0)
        histCanv.setXRange(0, len(self.score), padding=0)
        self.hypcrsr = histCanv.plot(
            [self.currep, self.currep], [-1, 2], pen='r')
        # print(len(self.selchan2p))
        # ampFactor = self.sr * self.epochl
        if self.currep + self.halfneps < self.maxep:
            fp = int(
                max([0, ((self.currep * self.epochl) - (self.tracedur / 2)) * self.sr]))
        else:
            fp = int(
                np.floor(
                    (self.maxep -
                     self.neps) *
                    self.epochl *
                    self.sr))
        # lp = min([self.totalp-1,int(fp + self.tracedur*self.sr)-1])
        lp = int(np.floor(fp + self.tracedur * self.sr))
        # beginnning of each epoch:
        indxtep = self.t0[list(range(fp, lp, int(self.epochl * self.sr)))]
        indxeps = np.round(indxtep / self.epochl).astype(int)
        # print(indxeps)
        # print(fp,lp,self.selchan2p)
        c = 'kbrkbrkbrkbr'
        p = 0
        plotCanv = self.ui.PlotWidget_signals
        plotCanv.clear()
        if len(self.selchan2p) > 0:
            for sch in self.selchan2p:
                plotCanv.plot(self.t0[fp:lp],
                              self.edfmat[sch, fp:lp], pen=c[p])
                p += 1
            plotCanv.setXRange(self.t0[fp], self.t0[lp], padding=0)

            axis = plotCanv.getAxis('bottom')
            # Set custom ticks with labels formatted as hh:mm:ss
            # set tick values to the x-coordinates of the data points
            tick_values = np.arange(self.t0[fp], self.t0[lp], 4).astype(int)
            startt = time.gmtime(self.tstart.timestamp())
            inits = startt.tm_hour * 3600 + startt.tm_min * 60 + startt.tm_sec
            difsec = inits
            if self.ui.checkBox_5.isChecked():  # SUbtract ZT0
                self.z0 = self.ui.timeEdit_3.time()
                z0secs = self.z0.hour() * 3600 + self.z0.minute() * 60 + self.z0.second()
                difsec = inits - z0secs
                if difsec < 0:
                    difsec = 24 * 3600 + difsec

            ticks = [(t, QtCore.QTime(int(np.floor(t / 3600)), (t %
                                                                3600) // 60, (t %
                                                                              3600) %
                                      60).addSecs(difsec).toString("hh:mm:ss")) for t in tick_values]
            axis.setTicks([ticks])
            if self.ui.checkBox_3.isChecked():
                ylim = self.maxEEG
            else:
                ylim = np.max(self.edfmat[self.selchan2p[0], fp:lp])
            plotCanv.setYRange(-ylim, ylim, padding=0)
            midp = indxtep[len(indxtep) // 2]
            for i, x in enumerate(indxtep):
                plotCanv.plot([x, x], [-ylim, ylim], pen='g')
                if indxeps[i] < self.maxep:
                    text = TextItem(
                        getsc(self.score[indxeps[i]], self.dictsc), color=(100, 200, 0))
                    plotCanv.addItem(text)
                    text.setPos(x + 1, 0.98 * ylim)
                    text.setFont(self.font)
            # if x == midp:
            # plotCanv.plot([x,x,x+self.epochl,x+self.epochl], [-ylim,ylim,ylim,-ylim], fillLevel=-1, brush=(150, 50, 200, 15))
        x = self.t0[int(self.t * self.sr)]
        plotCanv.plot([x, x, x + self.epochl, x + self.epochl], [-ylim,
                      ylim, ylim, -ylim], fillLevel=-1, brush=(150, 50, 200, 15))
        ax = plotCanv.getAxis('left')
        ax.setTicks([])
        if self.fixedEEG:
            plotCanv.setMouseEnabled(x=False, y=False)
        else:
            plotCanv.setMouseEnabled(x=False, y=True)
        plotCanvMT = self.ui.PlotWidget_MT
        plotCanvMT.clear()
        ax = plotCanvMT.getAxis('left')
        ax.setTicks([])
        mt2p = self.emg[fp:lp]

        # if self.ui.listWidget_2.currentRow() == 1:
        #     mt2p = mt2p **2

        # if self.ui.listWidget_2.currentRow() == 2:
        #     mt2p = signal.medfilt(mt2p,5)
        plotCanvMT.plot(self.t0[fp:lp], mt2p, pen='k')
        plotCanvMT.setXRange(self.t0[fp], self.t0[lp], padding=0)
        if self.ui.checkBox_4.isChecked():
            ylim = self.maxEMG
        else:
            ylim = np.max(mt2p)
        plotCanvMT.setYRange(0, ylim, padding=0)
        for x in indxtep:
            plotCanvMT.plot([x, x], [0, ylim], pen='k')
        plotCanvMT.setXLink(plotCanv)
        # FFT:
        plotCanvFFT = self.ui.PlotWidget_FFT
        plotCanvFFT.removeItem(self.pfft)
        if self.currep > self.halfneps:
            if self.currep < self.maxep - self.neps:
                fft2pl = self.fftmat[self.currep -
                                     self.halfneps:self.currep +
                                     self.halfneps, :].flatten()
            else:
                fft2pl = self.fftmat[self.maxep -
                                     self.neps:self.maxep, :].flatten()
        else:
            fft2pl = self.fftmat[0:2 * self.halfneps, :].flatten()

        # print(type(fft2pl))
        # print(len(fft2pl),len(faxis))
        indx = np.arange(len(fft2pl))
        self.pfft = plotCanvFFT.plot(
            indx, fft2pl, fillLevel=0, brush=(
                50, 50, 200, 100))
        plotCanvFFT.setXRange(indx[0], indx[-1], padding=0)
        if self.ui.checkBox_6.isChecked():
            ylim = self.maxFFT
        else:
            ylim = np.max(fft2pl)
        plotCanvFFT.setYRange(0, ylim, padding=0)
        indxb1 = indx[self.faxis == list(self.freqs)[0]]
        for item in self.fftl:
            self.fftl.removeItem(item)
        self.fftl = []
        for x in indxb1:
            self.fftl.append(plotCanvFFT.plot([x, x], [0, ylim], pen='k'))
        self.ui.label_13.setText("Ready")
        

    def loadEDF(self):  # read the EDF
        fileName = QFileDialog.getOpenFileName(
            self, 'Open EDF with sync data', '.', "EDF files (*.edf)")
        print(fileName)
        len(fileName)
        fileName = fileName[0]
        if len(fileName) >= 1:
            self.fn = fileName.split('/')[-1]
            os.chdir(fileName.rsplit('/', 1)[0])
            self.ui.label_5.setText(fileName)
            self.ui.label_13.setText("Reading EDF...")
            self.edf = mne.io.read_raw_edf(
                fileName, preload=True, stim_channel=None)
            self.sr = float(self.edf.info["sfreq"])
            # getting start and end time as tuple
            if isinstance(self.edf.info["meas_date"], tuple):
                self.tstart = time.gmtime(self.edf.info["meas_date"][0])
            else:
                if type(self.edf.info["meas_date"]) == datetime.datetime:
                    self.tstart = self.edf.info["meas_date"]
                else:
                    # print(edf.info["meas_date"],type(edf.info["meas_date"]))
                    self.tstart = time.gmtime(self.edf.info["meas_date"])

            # dt2 = time.gmtime(time.mktime(self.tstart)+edf.times[-1])
            dt2 = time.gmtime(self.tstart.timestamp() + self.edf.times[-1])
            print("Duration:", self.edf.times[-1])
            # formating time
            self.ui.label_8.setText(
                "Start date:" +
                time.strftime(
                    "%m/%d/%Y %H:%M:%S ",
                    time.gmtime(
                        self.tstart.timestamp())))
            self.ui.label_10.setText(
                "End date:" +
                time.strftime(
                    "%m/%d/%Y %H:%M:%S ",
                    dt2))
            # print(edf.info["meas_date"][1])
            self.t0 = self.edf.times  # In seconds since start?
            self.totalp = len(self.t0)
            self.maxep = int(self.edf.times[-1] // self.epochl)
            print('Max eps:', self.maxep)
            self.tracel = self.ui.epoch_length_2.value()
            self.edfmat = np.asarray(self.edf.get_data())
            #Select only the integer multiple of the epoch length
            dur01 = self.edfmat.shape[1] /self.sr
            adj_dur = self.epochl*(dur01//self.epochl)
            neps = adj_dur//self.epochl
            self.npep = int(np.floor(self.epochl * self.sr))
            totalpts01 =int(np.round(neps * self.npep))
            #totalpts01 = int(np.ceil(adj_dur * self.sr))
            self.edfmat = self.edfmat[:,:totalpts01]
            self.leeg = len(self.edfmat[0, :])
            print("Calculating FFT from channel", self.ui.EEGch.value())
            print("last EEG point:", self.npep * (self.leeg // self.npep))
            # self.edfmat[sch, fp:lp], pen=c[p])
            self.eegmat = self.edfmat[self.ui.EEGch.value(
            ) - 1, 0:self.npep * (self.leeg // self.npep)].reshape(self.leeg // self.npep, self.npep)
            print('Shape EEGmat:',np.shape(self.eegmat))
            self.fftmat = np.zeros(np.shape(self.eegmat))
            self.freqs = np.fft.fftfreq(self.npep, 1 / self.sr)
            self.score = -1 * np.ones(self.maxep)
            self.emg = self.edfmat[self.ui.EMGch.value() - 1, :]
            
            self.preprocessing()
            # current time, start date, end date, start time, end time, scroll
            # bar settings
            self.selchan2p.append(0)
            for n in self.edf.info["ch_names"]:
                self.ui.listWidget.addItem(n)
            self.ui.listWidget.setCurrentRow(0)
            print('Shape fftstats:',np.shape(self.fftstats))
            print('Ready!')
            #self.update_plots()
        else:
            print("No File selected")
        # newdt = time.gmtime(time.mktime(self.tstart) + self.t)
        newdt = time.gmtime(self.tstart.timestamp() + self.t)
        self.ui.dateTimeEdit_2.setDateTime(
            datetime.datetime.fromtimestamp(
                time.mktime(newdt)))
        self.ui.dateTimeEdit.setDateTime(
            datetime.datetime.fromtimestamp(
                time.mktime(dt2)))
        # plotting the first time
        histCanv = self.ui.PlotWidget_hypnogram
        histCanv.clear()
        self.scplt = histCanv.plot(self.score)
        histCanv.setYRange(-1.1, 2.2, padding=0)
        histCanv.setXRange(0, len(self.score), padding=0)
        self.hypcrsr = histCanv.plot(
            [self.currep, self.currep], [-1, 2], pen='r')
        # FFT:
        plotCanvFFT = self.ui.PlotWidget_FFT
        plotCanvFFT.clear()
        if self.currep > self.halfneps:
            if self.currep < self.maxep - self.neps:
                fft2pl = self.fftmat[self.currep -
                                     self.halfneps:self.currep +
                                     self.halfneps, :].flatten()
            else:
                fft2pl = self.fftmat[self.maxep -
                                     self.neps:self.maxep, :].flatten()
        else:
            fft2pl = self.fftmat[0:2 * self.halfneps, :].flatten()
        faxis = list(self.freqsp) * (2 * self.halfneps)
        # print(type(fft2pl))
        # print(len(fft2pl),len(faxis))
        indx = np.arange(len(fft2pl))
        self.pfft = plotCanvFFT.plot(
            indx, fft2pl, fillLevel=0, brush=(
                50, 50, 200, 100))
        plotCanvFFT.setXRange(indx[0], indx[-1], padding=0)
        if self.ui.checkBox_6.isChecked():
            ylim = self.maxFFT
        else:
            ylim = np.max(fft2pl)
        plotCanvFFT.setYRange(0, ylim, padding=0)
        indxb1 = indx[faxis == list(self.freqsp)[0]]
        for x in indxb1:
            plotCanvFFT.plot([x, x], [0, ylim], pen='k')
        ax = plotCanvFFT.getAxis('bottom')
        ticks = []
        for i in range(len(faxis)):
            if faxis[i] % 2 < 0.001:  # label only even frecs
                ticks.append('%.0f' % faxis[i])
            else:
                ticks.append("")
        xdict = dict(enumerate(ticks))
        ax.setTicks([xdict.items()])
        ax = plotCanvFFT.getAxis('left')
        ax.setTicks([])
        plotCanvFFT.setMouseEnabled(x=False, y=False)
        self.ui.label_13.setText("Ready")
        self.scoredict = {
            'score': self.score,
            'zt0': self.ui.timeEdit_3.time(),
            't0': self.ui.dateTimeEdit_2.dateTime(),
            'el': 4}

    def preprocessing(self):
        self.fftmat = np.zeros(np.shape(self.eegmat))
        for epoch in range(self.leeg // self.npep):
            self.fftmat[epoch, :] = np.abs(
                np.fft.fft(self.eegmat[epoch, :]))
            # We only one the freqs between 0 and the max freq FFT

        # now finding the pos of the 0 and maxfrec
        pos0 = np.argmin(np.abs(self.freqs))
        posmax = np.argmin(np.abs(self.freqs - self.ui.maxF.value()))
        posp = np.where(self.freqs >= 0)[0]
        self.fftstats = self.fftmat[:, posp].copy()
        print('Shape fftstats:',np.shape(self.fftstats))
        self.freqstats = self.freqs[posp].copy()
        self.fftmat = self.fftmat[:, pos0:posmax]
        self.freqsp = self.freqs[pos0:posmax]
        self.neps = self.tracel // self.epochl  # epochs per page
        self.halfneps = self.neps // 2
        
        

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q.value:
            print("Going to the next NREM epoch...")
        if event.key() == Qt.Key.Key_Down.value:
            if (self.currep - self.neps) >= 0:
                self.ui.lineEdit.setText(str(int(self.currep - self.neps + 1)))
                self.update_currep()
            else:
                self.ui.lineEdit.setText(str(int(self.halfneps)))
                self.update_currep()
        if event.key() == Qt.Key.Key_Up.value:
            if (self.currep + self.neps) < (self.maxep):
                self.ui.lineEdit.setText(str(int(self.currep + self.neps - 1)))
                self.update_currep()
            else:
                self.ui.lineEdit.setText(str(int(self.maxep - self.halfneps)))
                self.update_currep()

        if event.key() == Qt.Key.Key_Right.value:
            if (self.currep + 1) < (self.maxep):
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
                self.update_currep()
        if event.key() == Qt.Key.Key_Left.value:
            if self.currep > 0:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep - 1, self.maxep - 1]))))
                self.update_currep()
                # Scoring
        if event.key() == Qt.Key.Key_0.value:
            self.score[self.cur0rep] = 0
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_1.value:
            self.score[self.currep] = 1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_5.value:
            self.score[self.currep] = 2
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_7.value:
            self.score[self.currep] = 1.1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_8.value:
            self.score[self.currep] = 2.1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_9.value:
            self.score[self.currep] = 0.1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_3.value:
            self.score[self.currep] = 3
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_Period.value:
            self.score[self.currep] = 3
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_2.value:
            self.score[self.currep] = 3.1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_Comma.value:
            self.score[self.currep] = 3.1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if (event.key() == Qt.Key.Key_4.value) or (
                event.key() == Qt.Key.Key_S.value):
            self.score[self.currep] = 4
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_Semicolon.value:
            self.score[self.currep] = -1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if (event.key() == Qt.Key.Key_6.value):
            self.score[self.currep] = -1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        # same for normal keyboard in case there is no numpad:
        if event.key() == Qt.Key.Key_M.value:
            self.score[self.currep] = 0
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_J.value:
            self.score[self.currep] = 1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_K.value:
            self.score[self.currep] = 2
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_U.value:
            self.score[self.currep] = 1.1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()

        if event.key() == Qt.Key.Key_I.value:
            self.score[self.currep] = 2.1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_O.value:
            self.score[self.currep] = 0.1
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(
                    str(int(min([self.currep + 1, self.maxep - 1]))))
            self.update_currep()

        # jumps to next non scored epoch
        if event.key() == Qt.Key.Key_N.value:
            lnz = np.where(self.score[self.currep:] < 0)[
                0][0] + self.currep + 1
            self.currep = lnz
            self.ui.lineEdit.setText(
                str(int(min([self.currep - 1, self.maxep - 1]))))
            self.update_currep()

        if event.key() == Qt.Key.Key_Q.value:
            c1 = self.score[self.currep:] > 0.9
            c2 = self.score[self.currep:] < 1.9
            self.currep = np.argwhere(c1 & c2)[0] + self.currep
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.update_currep()
        if event.key() == Qt.Key.Key_R.value:
            c1 = self.score[self.currep:] > 1.9
            c2 = self.score[self.currep:] < 2.9
            self.currep = np.argwhere(c1 & c2)[0] + self.currep
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.update_currep()
        if event.key() == Qt.Key.Key_W.value:
            c1 = self.score[self.currep:] >= 0
            c2 = self.score[self.currep:] < 1
            self.currep = np.argwhere(c1 & c2)[0] + self.currep
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.update_currep()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec())

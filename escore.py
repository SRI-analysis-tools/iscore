import sys, os
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import xgboost as xgb
import time
import pickle
#from datetime import datetime, timedelta

from matplotlib import pyplot as plt
import scipy.io
import scipy.io as sio
from scipy.stats import mode
from PyQt6.QtGui import QIcon, QPixmap, QImage, QKeyEvent
from PyQt6.QtCore import Qt
from collections import defaultdict
import h5py
import mne
#from xml.dom import minidom
import scipy.signal
from scipy import stats
#import cv2
import matplotlib.pyplot as plt
#import random, math
#import matplotlib.image as mpimg
import datetime
from iscore import *
from pyqtgraph import TextItem
# QCoreApplication::setAttribute(Qt::AA_DisableHighDpiScaling);
#     if (qgetenv("QT_FONT_DPI").isEmpty()) {
#         qputenv("QT_FONT_DPI", "84");
#     }
#Qt.ApplicationAttribute(Qt.AA_DisableHighDpiScaling)
#Qt.ApplicationAttribute(Qt.AA_Use96Dpi)

#import pandas as pd
#from shutil import copyfile
"""If there is EEG, traces need to by synced with EEG. If traces include dropped frames, the they should match 
the N frames from the mat files, else they should match the nu,ber of detected frames form the sync signal.
A vector with the time of each frame will be generated, matching the EEG time of the first frame in every chunk.
Convert ui with pyuic5 demoLineEdit.ui -o demoLineEdit.py"""

#auxilliary functions and classes:
def getsc(x,d):
    if x in d:
        return(d[x])
    else:
        return 'U'
    
def getbouts(score,st):
    #Given score and state st, returns list with the duration of each bout
    index2 = np.where(np.round(score) == st)
    boutdur = []
    if len(index2)>0:
        index2 = index2[0]
        dindex = np.diff(index2)
        #EVery time the diff is not 1, we start counting the bd of a new bout
        nbout = 1
        pos_index = 0
        while pos_index < len(dindex):
            if dindex[pos_index]==1:
                nbout+=1
                #Add case for last bout
                if pos_index == len(dindex)-1:
                    boutdur.append(nbout)
            else:
                boutdur.append(nbout)
                nbout=1
                #Add case for last bout of only 1 epoch
                if pos_index == len(dindex)-1:
                    boutdur.append(nbout)
            pos_index+=1
    return np.array(boutdur)
                
def getbd(score,minbd=3, epochd=4):
    #mean bout duration in min of W,NR and REM
    #score is numeric where 0=W, 1=NR and 2 =R
    w = getbouts(score,0)
    nr = getbouts(score,1)
    r = getbouts(score,2)
    w = w[w>=minbd]
    nr = nr[nr>=minbd]
    r = r[r>=minbd]
    return np.mean(w)*epochd/60,np.mean(nr)*epochd/60,np.mean(r)*epochd/60
    
def getptime(score):
    #Returns the percentage of time in each state
    indw = np.where(np.round(score) == 0)
    if len(indw)>0:
        indw=indw[0]
    indnr = np.where(np.round(score) == 1)
    if len(indnr)>0:
        indnr=indnr[0]
    indr = np.where(np.round(score) == 2)
    if len(indr)>0:
        indr=indr[0]
    neps=len(score)
    return len(indw)/neps,len(indnr)/neps,len(indr)/neps
def expandx(x):
        #EXpand feature vecr x to include 3 previous and next
        neps,nfeat = x.shape
        prev1 = np.ones((neps,nfeat))*-1
        prev1[1:,:] = x[:-1,:]
        prev1[0,:] = prev1[1,:]
        prev2 = np.ones((neps,nfeat))*-1
        prev2[2:,:] = x[:-2,:]
        prev2[0,:] = prev2[2,:]
        prev2[1,:] = prev2[2,:]
        prev3 = np.ones((neps,nfeat))*-1
        prev3[3:,:] = x[:-3,:]
        prev3[0,:] = prev3[3,:]
        prev3[1,:] = prev3[3,:]
        prev3[2,:] = prev3[3,:]
        nextx=np.ones((neps,nfeat))*-1
        nextx[:-1,:] = x[1:,:]
        nextx[-1,:] = nextx[-2,:]
        #Concatenate all inputs into one matrix
        return np.hstack([x,prev1,prev2,prev3,nextx])
def halve(vect):
    #Convert the vector into half of it by averaging every two values
    if len(vect)//2==len(vect)/2:
        return vect.reshape(-1,2).mean(axis=1) 
    else:
        return vect[:-1].reshape(-1,2).mean(axis=1) 

#core classes
class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #specify all the controls here:
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
        self.ui.horizontalScrollBar.valueChanged.connect(self.scrollbar_moved)
        self.ui.epoch_length.valueChanged.connect(self.val_ch_epl)
        #self.ui.epoch_length.editingFinished.connect(self.update_epl) not working well for now
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
        self.ui.listWidget_3.itemSelectionChanged.connect(self.update_timescale)
        self.show()

        #Global variables:
        self.currep = 0
        #self.manual_score = []
        self.score = []
        self.auto_score = []
        self.manual_score2s = []
        self.auto_score2s = []
        self.currentpath = ""
        self.epochl = 4
        self.tracedur = 40
        self.edfname = ""
        self.scorefile = ""
        self.selchan2p=[]
        self.t0=0
        self.maxep=100
        self.tstart=0
        self.eegmat=[]
        self.fixedEEG = False
        self.fixedEMG = False
        self.maxFFT = 4
        self.t = 0
        self.font = QtGui.QFont("SansSerif", 14)
        self.dictsc ={0:'W', 0.1:'WA',1:'NR',1.1:'NA',2:'R',2.1:'RA'}
        self.totalp = 0
        self.scplt = None
        self.hypcrsr = None
        self.pfft = None
        self.faxis = []
        self.fftl =[]
        self.fn = ''
        self.train_mode = False

        #now def all the functions
    def val_ch_epl(self):
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
        self.eegmat = self.edfmat[self.ui.EEGch.value() - 1, 0:npep * (leeg // npep)].reshape(leeg // npep, npep)
        self.fftmat = np.zeros(np.shape(self.eegmat))
        self.freqs = np.fft.fftfreq(npep, 1 / self.sr)
        self.score = -1 * np.ones(self.maxep)
        for epoch in range(leeg // npep):
            self.fftmat[epoch, :] = np.abs(np.fft.fft(self.eegmat[epoch, :]))
            # We only one the freqs between 0 and the max freq FFT

        # now finding the pos of the 0 and maxfrec
        pos0 = np.argmin(np.abs(self.freqs))
        posmax = np.argmin(np.abs(self.freqs - self.ui.maxF.value()))
        self.fftmat = self.fftmat[:, pos0:posmax]
        self.freqs = self.freqs[pos0:posmax]

        self.neps = self.tracel // self.epochl  # epochs per page
        self.halfneps = self.neps // 2
        # current time, start date, end date, start time, end time, scroll bar settings
        self.update_plots()
        newdt = time.gmtime(time.mktime(self.tstart) + self.t)

        # plotting for the first time
        histCanv = self.ui.PlotWidget_hypnogram
        histCanv.clear()
        self.scplt = histCanv.plot(self.score)
        histCanv.setYRange(-1.1, 2.2, padding=0)
        histCanv.setXRange(0, len(self.score), padding=0)
        self.hypcrsr = histCanv.plot([self.currep, self.currep], [-1, 2], pen='r')
        # FFT:
        plotCanvFFT = self.ui.PlotWidget_FFT
        plotCanvFFT.clear()
        if self.currep > self.halfneps:
            if self.currep < self.maxep - self.neps:
                fft2pl = self.fftmat[self.currep - self.halfneps:self.currep + self.halfneps, :].flatten()
            else:
                fft2pl = self.fftmat[self.maxep - self.neps:self.maxep, :].flatten()
        else:
            fft2pl = self.fftmat[0:2 * self.halfneps, :].flatten()
        faxis = list(self.freqs) * (2 * self.halfneps)
        # print(type(fft2pl))
        # print(len(fft2pl),len(faxis))
        indx = np.arange(len(fft2pl))
        self.pfft = plotCanvFFT.plot(indx, fft2pl, fillLevel=0, brush=(50, 50, 200, 100))
        plotCanvFFT.setXRange(indx[0], indx[-1], padding=0)
        if self.ui.checkBox_6.isChecked():
            ylim = self.maxFFT
        else:
            ylim = np.max(fft2pl)
        plotCanvFFT.setYRange(0, ylim, padding=0)
        indxb1 = indx[faxis == list(self.freqs)[0]]
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
        #called upon change on list
        item = self.ui.listWidget_2.currentItem()
        self.emgtype = self.ui.listWidget_2.row(item)
        
        if self.emgtype == 2:
            self.emg = scipy.signal.medfilt(self.edfmat[self.ui.EMGch.value()-1, :] ,5)
        if self.emgtype == 0:
            self.emg = self.edfmat[self.ui.EMGch.value()-1, :]
        if self.emgtype == 1:
            self.emg = self.edfmat[self.ui.EMGch.value()-1, :] **2
            
    def update_fixEEG(self):
        self.fixedEEG = self.ui.checkBox_3.isChecked()
        if self.currep + self.halfneps < self.maxep:
            fp = int(max([0,((self.currep * self.epochl) - (self.tracedur/2))*self.sr]))
        else:
            fp = int(np.floor((self.maxep-self.neps)*self.epochl*self.sr))
        lp = int(np.floor(fp + self.tracedur * self.sr))
        sch=self.ui.EEGch.value()-1 #assuming first chan is EEG
        self.maxEEG=np.max(self.edfmat[sch, fp:lp])

    def update_fixEMG(self):
        self.fixedEMG = self.ui.checkBox_4.isChecked()
        if self.currep + self.halfneps < self.maxep:
            fp = int(max([0,((self.currep * self.epochl) - (self.tracedur/2))*self.sr]))
        else:
            fp = int(np.floor((self.maxep-self.neps)*self.epochl*self.sr))
        lp = int(np.floor(fp + self.tracedur * self.sr))
        self.maxEMG=np.max(self.emg[fp:lp])

    def update_fixFFT(self):
        self.fixedFFT=self.ui.checkBox_6.isChecked()
        if self.currep > self.halfneps:
            if self.currep < self.maxep - self.neps:
                fft2pl = self.fftmat[self.currep - self.halfneps:self.currep + self.halfneps, :].flatten()
            else:
                fft2pl = self.fftmat[self.maxep - self.neps:self.maxep, :].flatten()
        else:
            fft2pl = self.fftmat[0:2 * self.halfneps, :].flatten()
        self.maxFFT=np.max(fft2pl)

    def update_useZT(self):
        self.useZT = self.ui.checkBox_5.isChecked()
    def update_currep(self):
        self.currep = int(float(self.ui.lineEdit.text()))
        self.update_epocht()
        #update time and call updteplot
    def scrollbar_moved(self):
        v = self.ui.horizontalScrollBar.value()
        self.currep = int((v/1000)*self.maxep)
        self.ui.lineEdit.setText(str(self.currep))
        self.update_epocht()
    def update_timescale(self):
        pass
    def update_selchan(self):
        self.selchan2p= [self.ui.listWidget.row(item) for item in self.ui.listWidget.selectedItems()]
        if len(self.selchan2p)>0:
            self.update_plots()
    def update_epocht(self):
        self.t = self.epochl * self.currep
        #newdt = time.gmtime(time.mktime(self.tstart) + self.t)
        newdt = time.gmtime(self.tstart.timestamp() + self.t)
        self.ui.dateTimeEdit_4.setDateTime(datetime.datetime.fromtimestamp(time.mktime(newdt)))
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
            if item==self.fn[:-4]+'.pkl':
                with open(fn, 'rb') as handle:
                    print('loading scoring...')
                    self.scoredict = pickle.load(handle)
                    self.score=self.scoredict['score']
                break
#save(fn, 'sc', 'vali', 'startp', 'endp', 'zt0', 'scalefft', 'rangefft', 'epocl', 't0');endswith(".scr"):
        else:
            fileName = QFileDialog.getOpenFileName(self, 'Open scoring file', '', "scr files (*.scr), mat files (*.mat)")
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
                    
                except:
                    ValueError('could not read scoring mat file...')

                self.scoredict['score']=arrays['sc'].flatten()
                self.scoredict['el']=arrays['epocl'].flatten()[0]
                print(len(self.scoredict['score']))
                print(len(self.score))
                for i,s in enumerate(self.scoredict['score']):
                    if not np.isnan(s):
                        self.score[i]=s
            else:
                if len(self.scorefile) >= 1:
                    f = open(self.scorefile,'r')
                    lines = f.read().split()
                    f.close()
                    for linen in range(len(lines)):
                        if linen == 0:
                            self.epochl = int(lines[linen])
                        elif linen == 1:
                            edfstart = time.strptime(lines[linen], '%d_%m_%Y_%H:%M:%S')
                            self.ui.dateTimeEdit_4.setValue(edfstart) #check this
                        else:
                            self.manual_score.append(lines[linen])
        self.update_plots()

    def saveScoring(self):
        fn=self.fn[:-4]+'.pkl'
        print(fn)
        self.scoredict['score']=self.score
        #add automatic update of dict if zto, t0 or epoch len changes
        with open(fn, 'wb') as handle:
            pickle.dump(self.scoredict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('pickled!')
        print(self.scoredict)
        dictslp = {0: 'W', 0.1: 'WA', 1: 'NR', 1.1: 'NA', 2: 'R', 2.1: 'RA'}
        f=open(fn[:-4]+'.txt','w')
        for s in self.scoredict['score']:
            if s in list(dictslp.keys()):
                f.write(dictslp[s] + '\n')
            else:
                f.write('' + '\n')
        f.close()
        # if len(self.scorefile) < 1:
        #     self.scorefile=self.fn+ ".scr" #check this
        #     filelist = os.listdir()
        #     if self.scorefile in filelist:
        #         copyfile(self.scorefile,"backup_"+self.scorefile)
        #     f = open(self.scorefile, 'w')
        #     f.write(self.ui.dateTimeEdit_4.value())
        #     for st in self.manual_score:
        #         f.write(st)
        #     f.close()
            # fn = get(handles.filename, 'string');
            # fn = strcat(fn(1:end - 4), '.mat');
            # k = 0;
            # backupfn = fn;
            # while exist(backupfn)
            #     k = k + 1;
            #     backupfn = get(handles.filename, 'string');
            #     backupfn = strcat(fn(1:end - 4), '_', num2str(k), '.mat');
            #     end
            #     if k > 0
            #         copyfile(fn, backupfn)
            #     end
            #
            #     if isnan(handles.score(end))
            #         handles.score(end) = handles.score(end - 1);
            #     end
            #     startp = str2double(get(handles.startp, 'string'));
            #     endp = str2double(get(handles.endp, 'string'));
            #     sc = handles.score;
            #     vali = handles.validatedsc;
            #     scalefft = get(handles.limemg2, 'string');
            #     zt0 = get(handles.edit4, 'string');
            #     rangefft = handles.emgscale;
            #     epocl = handles.el;
            #     t0 = strcat(get(handles.stdate, 'string'), '_', get(handles.toedit, 'string'));
            #     t0(find(t0 == '/')) = '_';
            #     t0 = t0(4:end);
            #     save(fn, 'sc', 'vali', 'startp', 'endp', 'zt0', 'scalefft', 'rangefft', 'epocl', 't0');
            #     helpdlg('Score saved')


            #print('Epoch length:',self.epochl)
            #plotsc = self.ui.PlotWidget2
            #print(self.starttimes[0])
            #edfstart = time.strptime(self.t0, '%d_%m_%Y_%H:%M:%S')
            #print(edfstart)
            #dt = time.mktime(edfstart)
            #print(dt)
            #self.timesscore.append(dt)
            #for i in range(len(self.score)):
            #    if i>0:
            #        self.timesscore.append(self.timesscore[i-1]+float(self.epochl))

            #print(len(self.timesscore))
            #taxis = #time of each epoch//(np.arange(len(self.score))) * self.epochl
            #plotsc.plot(self.timesscore, self.score)
            #also plotting in the traces window (requires sync)
            #taxis = (np.arange(self.nframes)) / self.sr
            #plotCanv = self.ui.PlotWidget_tr
            #ampFactor = self.sr * self.epochl
            #hrscoring = ampfun(self.score, ampFactor)
            #plotCanv.plot(taxis, hrscoring,  pen='k')
    def loadtextScore(self):
        return None
    def autoScore(self):
        #Compute featues in 2s epochs: RMS of EEG and EMG and EEG power
        #Use gamma and EMG to identify clear sleep and wake epochs
        #Trains XGBoost on easy cases and scores the rest
        #Identifies wake artifacts as high EMG, high Delta
        #Converts 2s score into 4s score adding artifact epochs
        #bands:0-4, 6-10, 125-135, 285-295
        #Adding 2s epoch FFT
        self.ui.label_13.setText("Computing Features...")
        np2sep = int(2 * self.sr)#Number of points in 2 s epochs
        eegmat2s = self.edfmat[self.ui.EEGch.value()-1,0:np2sep * (self.edfmat.shape[1]//np2sep)].reshape(self.edfmat.shape[1]//np2sep,np2sep)
        emgmat2s = self.edfmat[self.ui.EMGch.value()-1,0:np2sep * (self.edfmat.shape[1]//np2sep)].reshape(self.edfmat.shape[1]//np2sep,np2sep)
        freqs2s = np.fft.fftfreq(np2sep, 1/self.sr)
        indx = np.where(freqs2s>=0)[0]
        fftmat2s = np.zeros((eegmat2s.shape[0],len(indx)))
        freqs2s=freqs2s[indx]
        for epoch in range(eegmat2s.shape[0]):
            fftmat2s[epoch,:] = np.abs(np.fft.fft(eegmat2s[epoch,:]))[indx]
        #GEt features:
        #1- Power bands
        g1i = 125
        g1f = 135
        g2i = 285
        g2f = 295
        if self.sr<600:
            g1i-=90
            g1f-=90
            g2i-=200
            g2f-=200
        deltafr = np.where(freqs2s<4)[0]
        gamma1fr = np.where((freqs2s>=g1i)&(freqs2s<=g1f))[0]
        gamma2fr = np.where((freqs2s>=g2i)&(freqs2s<=g2f))[0]
        thetafr = np.where((freqs2s>=6)&(freqs2s<=10))[0]
        self.delta = np.mean(fftmat2s[:,deltafr],axis=1)
        self.gamma1 = np.mean(fftmat2s[:,gamma1fr],axis=1)
        self.gamma2 = np.mean(fftmat2s[:,gamma2fr],axis=1)
        self.theta = np.mean(fftmat2s[:,thetafr],axis=1)
        self.delta[self.delta<1E-10]=1E-10
        self.thetad = self.theta/self.delta
        #2- RMS value
        self.eegrms = np.zeros(eegmat2s.shape[0])
        self.emgrms = np.zeros(eegmat2s.shape[0])
        for e in range(eegmat2s.shape[0]):
            self.eegrms[e] = np.sqrt(np.mean(eegmat2s[e,:]**2))
            self.emgrms[e] = np.sqrt(np.mean(emgmat2s[e,:]**2))
        self.ui.label_13.setText("Autoscoring...")
        self.ui.label_13.setStyleSheet("color: green;")
        self.train_mode = True
        autoscore = np.ones(len(self.delta))*-1
        #prevent zeros
        self.emgrms[self.emgrms<1E-10]=1E-10
        self.eegrms[self.eegrms<1E-10]=1E-10
        self.delta[self.delta<1E-10]=1E-10
        
        #Index for each state, including wake artifact
        indnr = (self.delta**3)/(self.emgrms*self.gamma1*self.thetad)
        indw = self.emgrms*self.gamma2/self.eegrms
        indwa = self.emgrms*self.gamma1*self.delta**2
        indrem = self.theta * (self.thetad**3) /self.emgrms
        

        nreps = np.where(indnr>np.percentile(indnr,80))[0]
        autoscore[nreps] = 1
        weps = np.where(indw>np.percentile(indw,80))[0]
        autoscore[weps] = 0

        #getting represetative values for delta and mt
        deltasleep = np.mean(self.delta[nreps])
        mtw = np.mean(self.emgrms[weps])
        #Using the values for filtering WA
        filt_delta = np.ones(len(indwa))
        filt_delta[self.delta<deltasleep]=0
        filt_notdelta = np.ones(len(indrem))
        filt_notdelta[self.delta>deltasleep]=0
        filt_mt = np.ones(len(indwa))
        filt_mt[self.emgrms<mtw]=0
        filt_notmt = np.ones(len(indrem))
        filt_notmt[self.emgrms>mtw]=0
        indwa = indwa*filt_delta *filt_mt
        indrem = indrem * filt_notdelta *filt_notmt

        reps = np.where(indrem>np.percentile(indrem,96))[0]
        autoscore[reps] = 2
        mtrem  = np.mean(self.emgrms[reps])
        deltarem = np.mean(self.delta[reps])
        tethar = np.mean(self.theta[reps])
        waeps = np.where(indwa>np.percentile(indwa,98))[0]
        autoscore[waeps] = 0.1 #neeed to be an int for now. Will be converted to 0.1
        indtrain = np.where(autoscore>=0)[0]
        ind_test = np.where(autoscore<0)[0]
        print(len(self.theta[indtrain]))
        x_train = np.vstack([self.delta[indtrain],self.theta[indtrain],self.thetad[indtrain],self.gamma1[indtrain],self.gamma2[indtrain],
                                        self.eegrms[indtrain],self.emgrms[indtrain]]).T
        x_test = np.vstack([self.delta[ind_test],self.theta[ind_test],self.thetad[ind_test],self.gamma1[ind_test],self.gamma2[ind_test],
                                        self.eegrms[ind_test],self.emgrms[ind_test]]).T
        y_train = autoscore[indtrain]
        #Add two previous and one next
        x_train = expandx(x_train)
        x_test = expandx(x_test)
        
        #Train ML
        # Define the XGBoost model
        y_train[y_train==0.1]=3
        # Determine class weights
        class_weights = len(y_train) / (int(len(set(y_train))) * np.bincount(y_train.astype(int)))
        print(class_weights)
        xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weights[0],
                              objective='multi:softmax',
                              num_class=4,
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
        
        xgb_model.fit(x_train, y_train)
        # predict the remaining epochs
        y_pred = xgb_model.predict(x_test)
        y_pred[y_pred==3]=0.1
        autoscore[ind_test] = y_pred

        #Now convert autoscore to 4s epochs and add artifacts for mixed scores
        if len(autoscore)//2 != len(autoscore)/2:
            autoscore=autoscore[:-1] #Trim last epoch if it is not even
        autos = autoscore.reshape(-1,2)
        for e in range(autos.shape[0]-1):
            if np.min(autos[e,:])>=0:
                #If any one is WA, score as WA
                if np.max(autos[e,:]==0.1):
                    self.score[e]=0.1
                else:
                    if np.round(autos[e,0])==np.round(autos[e,1]):
                        self.score[e] = np.max(autos[e,:])
                    else:
                        if e>0:
                            if np.max(autos[e,:]==3):
                                self.score[e] = 3
                            else:
                                #If one of the pair have same score of previous epoch, score as previous epoch with artifact
                                if autos[e,0] == np.round(self.score[e-1]):
                                    self.score[e] = np.round(self.score[e-1])+0.1
                                elif autos[e,1] == np.round(self.score[e-1]):
                                    self.score[e] = np.round(self.score[e-1])+0.1
                                elif e<len(self.score)-1:
                                    if autos[e,0] == np.round(self.score[e+1]):
                                        self.score[e] = np.round(self.score[e+1])+0.1
                                    elif autos[e,1] == np.round(self.score[e+1]):
                                        self.score[e] = np.round(self.score[e+1])+0.1
                                else:
                                    self.score[e] = np.round(self.score[e-1])+0.1
                        else:
                            self.score[e] = mode(np.hstack(autos[e,:],autos[e+1,:])).mode[0]
        #replace all NR that have low delta and high tethad by R if there is a R epoch nearby
        #Convert 2s vectors into 4s
        delta4 = halve(self.delta) 
        emg4 = halve(self.emgrms)
        tetha4 = halve(self.delta)
        newindr = np.where((np.round(self.score)==1) | (self.score<0))[0]
        mtrem  = 1.4*np.mean(self.emgrms[reps])
        deltarem = 1.4*np.mean(self.delta[reps])
        tethar = 0.5*np.mean(self.theta[reps])
        for e in newindr:
            if np.max(np.round(self.score[max([0,e-5]):min([e+5,len(self.score)-1])])==2):
                if (delta4[e]<deltarem) and (emg4[e]<mtrem) and (tetha4[e]>tethar):
                    self.score[e]=2
        #fix remaining unscored epochs
        unsc = np.where(self.score<0)
        if len(unsc)>0:
            for e in unsc[0]:
                ne=e.copy()
                while (ne>0) and (self.score[ne]<0):
                    ne-=1
                self.score[e] = self.score[ne]
        #Replace epochs flanked by REM into REM
        for ep in range(1,len(self.score)-1):
            if (round(self.score[ep-1])==2) and (round(self.score[ep+1])==2):
                self.score[ep]=2

        
        self.update_plots()
        return None
    def evaluateScore(self):
        return None
    def rescore(self):
        return None
    def stats(self):
        #Displays mean fft per state ,mean BD, % time in each state and save csv file with score and power every 2 Hz
        indw = np.where(self.score==0)
        indnr = np.where(self.score==1)
        indr = np.where(self.score==2)
        plt.figure()
        if len(indw)>0:
            indw = indw[0]
            fftw = np.mean(self.fftstats[indw,:],axis=0)
            plt.plot(self.freqstats,np.log(fftw),'g-',label='W')
        else:
            indw=[]
        if len(indnr)>0:
            indnr = indnr[0]
            fftnr = np.mean(self.fftstats[indnr,:],axis=0)
            plt.plot(self.freqstats,np.log(fftnr),'b-',label='NR')
        else:
            indnr=[]
        if len(indr)>0:
            indr = indr[0]
            fftr = np.mean(self.fftstats[indr,:],axis=0)
            plt.plot(self.freqstats,np.log(fftr),'r-',label='REM')
        else:
            indnr=[]
        plt.ylabel('Log(|FFT|) (Log $V$)',fontsize=15)
        plt.xlabel('Frequency (Hz)',fontsize=15)
        plt.title('FFT by state',fontsize=18)
        plt.legend()
        plt.show()
        #Plot again for selected freqs in linear scale
        plt.figure(figsize=(12,9))
        plt.subplot(2,1,1)
        if len(indw)>0:
            fftw = np.mean(self.fftmat[indw,:],axis=0)
            plt.plot(self.freqs,fftw,'#50C878',label='W')
        if len(indnr)>0:
            fftnr = np.mean(self.fftmat[indnr,:],axis=0)
            plt.plot(self.freqs,fftnr,'#4169E1',label='NR')
        if len(indr)>0:
            fftr = np.mean(self.fftmat[indr,:],axis=0)
            plt.plot(self.freqs,fftr,'#FF6347',label='REM')
        plt.ylabel(' |FFT| ($V$)',fontsize=15)
        plt.xlabel('Frequency (Hz)',fontsize=15)
        plt.title('FFT by state',fontsize=18)
        plt.legend()
        #now BD and PT
        bd = getbd(self.score)
        pt = getptime(self.score)
        plt.subplot(2,2,3)
        x_labels = ['WAKE', 'NREM', 'REM']
        #colors=['green', 'blue', 'red']
        colors = ['#50C878', '#4169E1', '#FF6347']
        plt.bar(x_labels, bd, color=colors)
        ax2 = plt.gca()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        plt.title('Mean bout duration',fontsize=19)
        plt.ylabel('Duration (Min)',fontsize=15)
        plt.subplot(2,2,4)
        x_labels = ['WAKE', 'NREM', 'REM']
        plt.pie(pt,labels=x_labels,colors=colors, autopct='%1.1f%%')
        plt.title('Percentage of time',fontsize=19)
        plt.tight_layout()
        plt.show()

        return None
    def undo(self):
        return None
    def plotTrace(self):
        plt.figure()
        plt.subplot(6,1,1)
        plt.plot(self.delta)
        plt.ylabel('Delta')
        plt.subplot(6,1,2)
        plt.plot(self.gamma1)
        plt.ylabel('$\gamma$1')
        plt.subplot(6,1,3)
        plt.plot(self.gamma2)
        plt.ylabel('$\gamma$2')
        plt.subplot(6,1,4)
        plt.plot(self.theta)
        plt.ylabel('$\\theta$')
        plt.subplot(6,1,5)
        plt.plot(self.eegrms)
        plt.ylabel('EEGrms')
        plt.subplot(6,1,6)
        plt.plot(self.emgrms)
        plt.ylabel('EMGrms')
        #Plots mean power for bands
        print()
        plt.figure()
        plt.plot(self.gamma2,self.emgrms,'k.',alpha=0.4)
        plt.ylabel('EMG',fontsize=15)
        plt.xlabel('Gamma 285-295 Hz',fontsize=15)
        plt.title(f'r={np.round(np.corrcoef(self.gamma2,self.emgrms)[0,1],2)}')
        plt.figure()
        deltafr = np.where(self.freqstats<4)[0]
        gammafr = np.where((self.freqstats>30)&(self.freqstats<59))[0]
        gammarem = np.where((self.freqstats>=125)&(self.freqstats<=135))[0]
        gammaw = np.where((self.freqstats>=285)&(self.freqstats<=295))[0]
        window_size = 15
        window = np.ones(int(window_size))/float(window_size)
        
        delta = np.mean(self.fftstats[:,deltafr],axis=1)
        gamma = scipy.signal.medfilt(np.mean(self.fftstats[:,gammafr],axis=1))
        gammarem = np.mean(self.fftstats[:,gammarem],axis=1)
        gammaw = np.mean(self.fftstats[:,gammaw],axis=1)
        plt.subplot(4,1,1)
        plt.plot(np.convolve(delta, window, 'same'),'k-',linewidth=0.5)
        plt.ylim(0,np.percentile(delta,95))
        plt.ylabel('$\delta$',fontsize=16)
        plt.subplot(4,1,2)
        plt.plot(np.convolve(gamma, window, 'same'),'k-',linewidth=0.5)
        plt.ylim(0,np.percentile(gamma,95))
        plt.ylabel('$\gamma$',fontsize=16)
        plt.subplot(4,1,3)
        plt.plot(np.convolve(gammarem, window, 'same'),'k-',linewidth=0.5)
        plt.ylim(0,np.percentile(gammarem,95))
        plt.ylabel('$\gamma$_Rem',fontsize=16)
        plt.subplot(4,1,4)
        plt.plot(np.convolve(gammaw, window, 'same'),'k-',linewidth=0.5)
        plt.ylim(0,np.percentile(gammaw,95))
        plt.ylabel('$\gamma$_W',fontsize=16)
        plt.show()
        return None
    def update_tracel(self):
        self.tracel=self.ui.epoch_length_2.value()
        self.update_plots()
    def restoreEEG_scale(self):
        return None
    def restoreEMG_scale(self):
        return None
    def update_plots(self):
        #plots data according to : current epoch,
        #epoch length, trace length, channels 2 plot, channel for EMG, EMG
        # signal type, timescale type, ZT0, EEg scaling, EEG scale, EMG scaling, EMG scales,
        histCanv = self.ui.PlotWidget_hypnogram
        histCanv.removeItem(self.scplt)
        histCanv.removeItem(self.hypcrsr)
        self.scplt = histCanv.plot(self.score,pen='k')
        histCanv.setYRange(-1.1, 2.2, padding=0)
        histCanv.setXRange(0, len(self.score), padding=0)
        self.hypcrsr = histCanv.plot([self.currep, self.currep], [-1, 2], pen='r')
        #print(len(self.selchan2p))
        # ampFactor = self.sr * self.epochl
        if self.currep + self.halfneps < self.maxep:
            fp = int(max([0,((self.currep * self.epochl) - (self.tracedur/2))*self.sr]))
        else:
            fp = int(np.floor((self.maxep-self.neps)*self.epochl*self.sr))
        #lp = min([self.totalp-1,int(fp + self.tracedur*self.sr)-1])
        lp = int(np.floor(fp + self.tracedur * self.sr))
        #beginnning of each epoch:
        indxtep = self.t0[list(range(fp,lp,int(self.epochl * self.sr)))]
        indxeps = np.round(indxtep/self.epochl).astype(int)
        #print(indxeps)
        #print(fp,lp,self.selchan2p)
        c = 'kbrkbrkbrkbr'
        p = 0
        plotCanv = self.ui.PlotWidget_signals
        plotCanv.clear()
        if len(self.selchan2p)>0:
            for sch in self.selchan2p:
                plotCanv.plot(self.t0[fp:lp], self.edfmat[sch, fp:lp],  pen=c[p])
                p += 1
            plotCanv.setXRange(self.t0[fp],self.t0[lp], padding=0)
            if self.ui.checkBox_3.isChecked():
                ylim = self.maxEEG
            else:
                ylim = np.max(self.edfmat[self.selchan2p[0], fp:lp])
            plotCanv.setYRange(-ylim, ylim, padding=0)
            midp = indxtep[len(indxtep)//2]
            for i,x in enumerate(indxtep):
                plotCanv.plot([x,x], [-ylim,ylim], pen='g')
                if indxeps[i] < self.maxep:
                    text = TextItem(getsc(self.score[indxeps[i]],self.dictsc),color=(100, 200, 0))
                    plotCanv.addItem(text)
                    text.setPos(x+1,0.98*ylim)
                    text.setFont(self.font)
            #if x == midp:
            #plotCanv.plot([x,x,x+self.epochl,x+self.epochl], [-ylim,ylim,ylim,-ylim], fillLevel=-1, brush=(150, 50, 200, 15))
        x = self.t0[int(self.t * self.sr)]
        plotCanv.plot([x, x, x + self.epochl, x + self.epochl], [-ylim, ylim, ylim, -ylim], fillLevel=-1,
                      brush=(150, 50, 200, 15))
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
        #     mt2p = scipy.signal.medfilt(mt2p,5)
        plotCanvMT.plot(self.t0[fp:lp], mt2p, pen='k')
        plotCanvMT.setXRange(self.t0[fp], self.t0[lp], padding=0)
        if self.ui.checkBox_4.isChecked():
            ylim = self.maxEMG
        else:
            ylim = np.max(mt2p)
        plotCanvMT.setYRange(0, ylim, padding=0)
        for x in indxtep:
            plotCanvMT.plot([x,x], [0,ylim], pen='k')
        plotCanvMT.setXLink(plotCanv)
        #FFT:
        plotCanvFFT = self.ui.PlotWidget_FFT
        plotCanvFFT.removeItem(self.pfft)
        if self.currep > self.halfneps:
            if self.currep < self.maxep-self.neps:
                fft2pl = self.fftmat[self.currep-self.halfneps:self.currep+self.halfneps,:].flatten()
            else:
                fft2pl = self.fftmat[self.maxep - self.neps:self.maxep, :].flatten()
        else:
            fft2pl = self.fftmat[0:2 * self.halfneps,:].flatten()

        #print(type(fft2pl))
        #print(len(fft2pl),len(faxis))
        indx=np.arange(len(fft2pl))
        self.pfft = plotCanvFFT.plot(indx, fft2pl,  fillLevel=0, brush=(50,50,200,100))
        plotCanvFFT.setXRange(indx[0],indx[-1], padding=0)
        if self.ui.checkBox_6.isChecked():
            ylim = self.maxFFT
        else:
            ylim = np.max(fft2pl)
        plotCanvFFT.setYRange(0, ylim, padding=0)
        indxb1 = indx[self.faxis == list(self.freqs)[0]]
        for item in self.fftl:
            self.fftl.removeItem(item)
        self.fftl=[]
        for x in indxb1:
            self.fftl.append(plotCanvFFT.plot([x,x], [0,ylim], pen='k'))
        


    def loadEDF(self):#read the EDF
        fileName = QFileDialog.getOpenFileName(self, 'Open EDF with sync data', '.', "EDF files (*.edf)")
        print(fileName)
        len(fileName)
        fileName = fileName[0]
        if len(fileName) >= 1:
            self.fn=fileName.split('/')[-1]
            os.chdir(fileName.rsplit('/',1)[0])
            self.ui.label_5.setText(fileName)
            self.ui.label_13.setText("Reading EDF...")
            edf = mne.io.read_raw_edf(fileName, preload=True, stim_channel=None)
            self.edf = edf
            self.sr = float(edf.info["sfreq"])
            #getting start and end time as tuple
            if isinstance(edf.info["meas_date"],tuple):
                self.tstart = time.gmtime(edf.info["meas_date"][0])
            else:
                if type(edf.info["meas_date"])==datetime.datetime:
                    self.tstart = edf.info["meas_date"]
                else:
                    #print(edf.info["meas_date"],type(edf.info["meas_date"]))
                    self.tstart = time.gmtime(edf.info["meas_date"])
            
            #dt2 = time.gmtime(time.mktime(self.tstart)+edf.times[-1])
            dt2 = time.gmtime(self.tstart.timestamp()+edf.times[-1])
            print("Duration:",edf.times[-1])
            #formating time
            self.ui.label_8.setText("Start date:" + time.strftime("%m/%d/%Y %H:%M:%S ",time.gmtime(self.tstart.timestamp())))
            self.ui.label_10.setText("End date:" + time.strftime("%m/%d/%Y %H:%M:%S ", dt2))
            #print(edf.info["meas_date"][1])
            self.t0 = edf.times
            self.totalp = len(self.t0)
            self.maxep = int(edf.times[-1]//self.epochl)
            print('Max eps:',self.maxep)
            self.tracel = self.ui.epoch_length_2.value()
            self.edfmat = np.asarray(edf.get_data())
            npep = int(self.epochl * self.sr)
            
            leeg= len(self.edfmat[0,:])
            print("Calculating FFT from channel",self.ui.EEGch.value())
            print("last EEG point:",npep * (leeg//npep))
            #self.edfmat[sch, fp:lp], pen=c[p])
            self.eegmat = self.edfmat[self.ui.EEGch.value()-1,0:npep * (leeg//npep)].reshape(leeg//npep,npep)
            print(np.shape(self.eegmat))
            self.fftmat = np.zeros(np.shape(self.eegmat))
            self.freqs = np.fft.fftfreq(npep, 1/self.sr)
            self.score= -1* np.ones(self.maxep)
            self.emg = self.edfmat[self.ui.EMGch.value()-1, :]
            for epoch in range(leeg//npep):
                self.fftmat[epoch,:] = np.abs(np.fft.fft(self.eegmat[epoch,:]))
                #We only one the freqs between 0 and the max freq FFT
            
            #now finding the pos of the 0 and maxfrec
            pos0 = np.argmin(np.abs(self.freqs))
            posmax = np.argmin(np.abs(self.freqs - self.ui.maxF.value()))
            posp = np.where(self.freqs>=0)[0]
            self.fftstats = self.fftmat[:,posp].copy()
            self.freqstats = self.freqs[posp].copy()
            self.fftmat = self.fftmat[:,pos0:posmax]
            self.freqs = self.freqs[pos0:posmax]
            self.neps = self.tracel // self.epochl #epochs per page
            self.halfneps = self.neps //2
            #current time, start date, end date, start time, end time, scroll bar settings
            self.selchan2p.append(0)
            for n in edf.info["ch_names"]:
                self.ui.listWidget.addItem(n)
            self.ui.listWidget.setCurrentRow(0)
            print('Ready!')
            self.update_plots()
        else:
            print("No File selected")
        #newdt = time.gmtime(time.mktime(self.tstart) + self.t)
        newdt = time.gmtime(self.tstart.timestamp() + self.t)
        self.ui.dateTimeEdit_2.setDateTime(datetime.datetime.fromtimestamp(time.mktime(newdt)))
        self.ui.dateTimeEdit.setDateTime(datetime.datetime.fromtimestamp(time.mktime(dt2)))
        #plotting the first time
        histCanv = self.ui.PlotWidget_hypnogram
        histCanv.clear()
        self.scplt = histCanv.plot(self.score)
        histCanv.setYRange(-1.1, 2.2, padding=0)
        histCanv.setXRange(0, len(self.score), padding=0)
        self.hypcrsr=histCanv.plot([self.currep, self.currep], [-1, 2], pen='r')
        #FFT:
        plotCanvFFT = self.ui.PlotWidget_FFT
        plotCanvFFT.clear()
        if self.currep > self.halfneps:
            if self.currep < self.maxep - self.neps:
                fft2pl = self.fftmat[self.currep - self.halfneps:self.currep + self.halfneps, :].flatten()
            else:
                fft2pl = self.fftmat[self.maxep - self.neps:self.maxep, :].flatten()
        else:
            fft2pl = self.fftmat[0:2 * self.halfneps, :].flatten()
        faxis = list(self.freqs) * (2 * self.halfneps)
        # print(type(fft2pl))
        # print(len(fft2pl),len(faxis))
        indx = np.arange(len(fft2pl))
        self.pfft = plotCanvFFT.plot(indx, fft2pl, fillLevel=0, brush=(50, 50, 200, 100))
        plotCanvFFT.setXRange(indx[0], indx[-1], padding=0)
        if self.ui.checkBox_6.isChecked():
            ylim = self.maxFFT
        else:
            ylim = np.max(fft2pl)
        plotCanvFFT.setYRange(0, ylim, padding=0)
        indxb1 = indx[faxis == list(self.freqs)[0]]
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
        self.scoredict = {'score': self.score, 'zt0': self.ui.timeEdit_3.time(), 't0': self.ui.dateTimeEdit_2.dateTime(), 'el': 4}

    # def saveAll(self): #this is from ca imager
    #     #making data frame with the Z scores of ever cell for every state, assuming 1 period
    #     #For each cell we need a table with the Zscore of an epoch and it's state
    #     mat_aux=[]
    #     k0=1
    #     lstate = ['Wake'] * len(self.activityW[1, :]) + ['NREM'] * len(self.activityN[1, :]) + ['REM'] * len(
    #         self.activityR[1, :])
    #     #there can be statistical differences due to larger than the rest or smaller than the rest activity
    #     wakeL =[]
    #     nremL = []
    #     remL = []
    #     wakeS = []
    #     nremS = []
    #     remS = []
    #     indep = []
    #     faL=[]
    #     oaL=[]
    #     faS=[]

    #     lfwa1 = int(len(self.fwa)/2)
    #     lowa1 = int(len(self.owa)/2)
    #     for n in range(self.Ncells):
    #         lact=list(self.activityW[n,:]) + list(self.activityN[n,:])+list(self.activityR[n,:])

    #         #lstate = ('WAKE ' * len(self.activityW)).split()+ ('NREM ' * len(self.activityN)).split() + ('REM ' * len(self.activityR)).split()
    #         F, p = stats.f_oneway(self.activityW[n,:],self.activityN[n,:],self.activityR[n,:])
    #         #print(self.activityW[n,:].mean(),self.activityN[n,:].mean(),self.activityR[n,:].mean())
    #         if p<0.05:
    #             print("Significant! N=",k0)
    #             k0+=1
    #             if (self.activityW[n,:].mean()>self.activityN[n,:].mean()) and (self.activityW[n,:].mean()>self.activityR[n,:].mean()):
    #                 wakeL.append(n)
    #             elif (self.activityW[n,:].mean()<self.activityN[n,:].mean()) and (self.activityW[n,:].mean()<self.activityR[n,:].mean()):
    #                 wakeS.append(n)
    #             elif (self.activityN[n,:].mean()<self.activityW[n,:].mean()) and (self.activityN[n,:].mean()<self.activityR[n,:].mean()):
    #                 nremS.append(n)
    #             elif (self.activityN[n,:].mean()>self.activityW[n,:].mean()) and (self.activityN[n,:].mean()>self.activityR[n,:].mean()):
    #                 nremL.append(n)
    #             elif (self.activityR[n,:].mean()<self.activityW[n,:].mean()) and (self.activityR[n,:].mean()<self.activityN[n,:].mean()):
    #                 remS.append(n)
    #             elif (self.activityR[n,:].mean()>self.activityW[n,:].mean()) and (self.activityR[n,:].mean()>self.activityN[n,:].mean()):
    #                 remL.append(n)
    #         else:
    #             indep.append(n)
    #         F, p1 = stats.f_oneway(self.fwa[n,0:lfwa1],self.owa[n,0:lowa1]) # Checking only the first half ogf the data
    #         F, p2 = stats.f_oneway(self.fwa[n, lfwa1:],
    #                                self.owa[n, lowa1:])  # Checking only the second half ogf the data
    #         if (p1<0.05) and (p2<0.05):
    #             if (self.fwa[n,0:lfwa1].mean()>self.owa[n,0:lowa1].mean()) and (self.fwa[n,lfwa1:].mean()>self.owa[n,lowa1:].mean()):
    #                 faL.append(n) #saves the number of the cell that has higher activity at the onset of W
    #             else:
    #                 faS.append(n)
    #         else:
    #             oaL.append(n)
    #     print("percentage of active cells at onset: ",100*len(faL)/self.Ncells)
    #     print("percentage of less active cells at onset: ", 100 * len(faS) / self.Ncells)
    #     print("percentage of cells indifferent to onset: ", 100 * len(oaL) / self.Ncells)
    #     mixed = wakeS +nremS +remS
    #     if len(wakeL +nremL +remL +mixed +indep ) != self.Ncells:
    #         print("missing cells!!")
    #     #Making final figure with traces and hypnogram, jellybeans and pie chart
    #     labels =[]
    #     sizes =[]
    #     explode = []
    #     colors =[]
    #     colorlist = 'g', 'r', 'b', (0.5,0.5,0.5), 'y'
    #     labellist = 'W', 'R', 'NR', 'Mixed', 'Ind'
    #     i=0
    #     for m in [wakeL,remL, nremL, mixed,indep]:
    #         if len(m)>0:
    #             labels.append(labellist[i])
    #             sizes.append(100*len(m)/self.Ncells)
    #             explode.append(0)
    #             colors.append(colorlist[i])
    #         i+=1

    #     #To do: add first vs other activity during W
    #     #self.foa

    #     #making summary figure with jellybeans,
    #     fig=plt.figure()
    #     grid = plt.GridSpec(2, 3, wspace=0.0, hspace=0.1)
    #     plt.subplot(grid[0, 0])
    #     plt.imshow(self.mat2plot)
    #     plt.axis("off")
    #     plt.draw()
    #     ax1 = plt.subplot(grid[0, 1])
    #     ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
    #             shadow=False, textprops={'size': 'x-large', 'weight':'bold'}, startangle=90)
    #     ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    #     ax=plt.subplot(grid[0, 2]) #plotting bar plot with the difference in activity for cells who had larger activity during hte first W bout
    #     meanw,meann,meanr = self.activityW.mean(axis=1), self.activityN.mean(axis=1),self.activityR.mean(axis=1) #mean activity for every cell
    #     error = [(0,0,0),[stats.sem(meanw),stats.sem(meann), stats.sem(meanr)]]
    #     bp=plt.bar([0,1,2], [meanw.mean(),meann.mean(),meanr.mean()],
    #                yerr=error,align='center',alpha=1, ecolor='k',capsize=5)
    #     plt.xticks([0, 1,2], ('WAKE', 'NREM','REM'))
    #     plt.ylabel('Mean Z score')
    #     bp[0].set_color('g')
    #     bp[1].set_color('b')
    #     bp[2].set_color('r')

    #     # plt.bar([0,1],[self.fwa[faL, :].mean(), self.owa[faL, :].mean()])
    #     # plt.xticks([0,1], ('Onset W', 'Within W'))
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     plt.tight_layout()
    #     #plt.box(on=None)
    #     ax = plt.subplot(grid[1, 0:])
    #     #now plotting hypnogram and traces
    #     tscore =[t*self.epochl for t in range(len(self.chunk_score))]
    #     taxis = (np.arange(self.nframes)) / self.sr
    #     plt.plot(tscore[3:], self.chunk_score[3:]*4.5)
    #     indxt = list(range(self.Ncells))
    #     random.shuffle(indxt)
    #     traceindx = indxt[0:8]
    #     for i in range(len(traceindx)):
    #         plt.plot(taxis[int(self.sr*12):], self.Traces[i,int(self.sr*12):].T + 10 * (i+1),linewidth=0.5)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #     ax.tick_params(left=False)
    #     ax.tick_params(labelleft=False)
    #     #plt.box(on=None)
    #     #ax.set_frame_on(False)
    #     #ax.add_axes([0., 1., 1., 0])
    #     #ax = plt.axes([0, 1., 0, 1.])
    #     #ax.get_xaxis().set_visible(True)
    #     #ax.get_yaxis().set_visible(False)
    #     plt.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q.value:
            print("Going to the next NREM epoch...")
        if event.key() == Qt.Key.Key_Down.value:
            if (self.currep - self.neps) >= 0:
                self.ui.lineEdit.setText(str(int(self.currep - self.neps+1)))
                self.update_currep()
            else:
                self.ui.lineEdit.setText(str(int(self.halfneps)))
                self.update_currep()
        if event.key() == Qt.Key.Key_Up.value:
            if (self.currep + self.neps) < (self.maxep):
                self.ui.lineEdit.setText(str(int(self.currep + self.neps-1)))
                self.update_currep()

            else:

                self.ui.lineEdit.setText(str(int(self.maxep-self.halfneps)))
                self.update_currep()

        if event.key() == Qt.Key.Key_Right.value:
            if (self.currep + 1) < (self.maxep):
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
                self.update_currep()
        if event.key() == Qt.Key.Key_Left.value:
            if self.currep > 0:
                self.ui.lineEdit.setText(str(int(min([self.currep - 1,self.maxep-1]))))
                self.update_currep()
                # Scoring
        if event.key() == Qt.Key.Key_0.value:
            self.score[self.currep] = 0
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_1.value:
            self.score[self.currep] = 1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_5.value:
            self.score[self.currep] = 2
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_7.value:
            self.score[self.currep] = 1.1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_8.value:
            self.score[self.currep] = 2.1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_9.value:
            self.score[self.currep] = 0.1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        #same for normal keyboard in case there is no numpad:
        if event.key() == Qt.Key.Key_M.value:
            self.score[self.currep] = 0
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_J.value:
            self.score[self.currep] = 1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_K.value:
            self.score[self.currep] = 2
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        
        if event.key() == Qt.Key.Key_I.value:
            self.score[self.currep] = 2.1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key.Key_O.value:
            self.score[self.currep] = 0.1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        #jumps to next non scored epoch
        if event.key() == Qt.Key.Key_N.value:
            lnz = np.where(self.score[self.currep:] < 0)[0][0]+self.currep
            self.currep = lnz
            self.ui.lineEdit.setText(str(int(min([self.currep - 1, self.maxep - 1]))))
            self.update_currep()

        if event.key() == Qt.Key.Key_Q.value:
            c1=self.score[self.currep:] > 0.9
            c2=self.score[self.currep:] < 1.9
            self.currep = np.argwhere(c1 & c2)[0]+self.currep
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.update_currep()
        if event.key() == Qt.Key.Key_R.value:
            c1 = self.score[self.currep:] > 1.9
            c2 = self.score[self.currep:] < 2.9
            self.currep = np.argwhere(c1 & c2)[0] + self.currep
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.update_currep()
        if event.key() == Qt.Key.Key_W.value:
            c1 = self.score[self.currep:] >=0
            c2 = self.score[self.currep:] < 1
            self.currep = np.argwhere(c1 & c2)[0] + self.currep
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.update_currep()
        if event.key() == Qt.Key.Key_U.value:
            c1 = self.score[self.currep:] <0
            c2 = self.score[self.currep:] > 3
            self.currep = np.argwhere(c1 | c2)[0] + self.currep
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.update_currep()

if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec())

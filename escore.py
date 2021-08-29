import sys, os, numpy
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
import pickle
#from datetime import datetime, timedelta

from matplotlib import pyplot as plt
import scipy.io
import scipy.io as sio
from PyQt5.QtGui import QIcon, QPixmap, QImage, QKeyEvent
from PyQt5.QtCore import Qt
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
Qt.ApplicationAttribute(Qt.AA_Use96Dpi)
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

        #now def all the functions
    def val_ch_epl(self):
        return None

    def update_epl(self):
        self.ui.label_11.setText("Recalculating...")
        self.epochl = self.ui.epoch_length.value()
        self.maxep = int(self.edf.times[-1] // self.epochl)
        self.tracel = self.ui.epoch_length_2.value()
        self.edfmat = numpy.asarray(self.edf.get_data())
        npep = int(self.epochl * self.sr)
        leeg = len(self.edfmat[0, :])
        # self.edfmat[sch, fp:lp], pen=c[p])
        self.eegmat = self.edfmat[self.ui.EEGch.value() - 1, 0:npep * (leeg // npep)].reshape(leeg // npep, npep)
        self.fftmat = numpy.zeros(numpy.shape(self.eegmat))
        self.freqs = numpy.fft.fftfreq(npep, 1 / self.sr)
        self.score = -1 * numpy.ones(self.maxep)
        for epoch in range(leeg // npep):
            self.fftmat[epoch, :] = numpy.abs(numpy.fft.fft(self.eegmat[epoch, :])) ** 2
            # We only one the freqs between 0 and the max freq FFT

        # now finding the pos of the 0 and maxfrec
        pos0 = numpy.argmin(numpy.abs(self.freqs))
        posmax = numpy.argmin(numpy.abs(self.freqs - self.ui.maxF.value()))
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
        indx = numpy.arange(len(fft2pl))
        self.pfft = plotCanvFFT.plot(indx, fft2pl, fillLevel=0, brush=(50, 50, 200, 100))
        plotCanvFFT.setXRange(indx[0], indx[-1], padding=0)
        if self.ui.checkBox_6.isChecked():
            ylim = self.maxFFT
        else:
            ylim = numpy.max(fft2pl)
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
        self.ui.label_11.setText("Ready")

    def closeap(self):
        self.close()
    def update_ZT0(self):
        pass
    def update_emg(self):
        item = self.ui.listWidget_2.currentItem()
        self.emgtype = self.ui.listWidget_2.row(item)
    def update_fixEEG(self):
        self.fixedEEG = self.ui.checkBox_3.isChecked()
        if self.currep + self.halfneps < self.maxep:
            fp = int(max([0,((self.currep * self.epochl) - (self.tracedur/2))*self.sr]))
        else:
            fp = int(numpy.floor((self.maxep-self.neps)*self.epochl*self.sr))
        lp = int(numpy.floor(fp + self.tracedur * self.sr))
        sch=0 #assuming first chan is EEG
        self.maxEEG=numpy.max(self.edfmat[sch, fp:lp])
    def update_fixEMG(self):
        self.fixedEMG = self.ui.checkBox_4.isChecked()
        if self.currep + self.halfneps < self.maxep:
            fp = int(max([0,((self.currep * self.epochl) - (self.tracedur/2))*self.sr]))
        else:
            fp = int(numpy.floor((self.maxep-self.neps)*self.epochl*self.sr))
        lp = int(numpy.floor(fp + self.tracedur * self.sr))
        sch=2 #assuming ch2 is EMG
        self.maxEMG=numpy.max(self.edfmat[sch, fp:lp])**2

    def update_fixFFT(self):
        self.fixedFFT=self.ui.checkBox_6.isChecked()
        if self.currep > self.halfneps:
            if self.currep < self.maxep - self.neps:
                fft2pl = self.fftmat[self.currep - self.halfneps:self.currep + self.halfneps, :].flatten()
            else:
                fft2pl = self.fftmat[self.maxep - self.neps:self.maxep, :].flatten()
        else:
            fft2pl = self.fftmat[0:2 * self.halfneps, :].flatten()
        self.maxFFT=numpy.max(fft2pl)

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
                        arrays[k] = numpy.array(v)
                    m = arrays['zt0'].flatten()
                    self.scoredict['zt0'] = m[0]
                except NotImplementedError:    
                    f = h5py.File(self.scorefile)
                    for k, v in f.items():
                        arrays[k] = numpy.array(v)
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
                    if not numpy.isnan(s):
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
            #taxis = #time of each epoch//(numpy.arange(len(self.score))) * self.epochl
            #plotsc.plot(self.timesscore, self.score)
            #also plotting in the traces window (requires sync)
            #taxis = (numpy.arange(self.nframes)) / self.sr
            #plotCanv = self.ui.PlotWidget_tr
            #ampFactor = self.sr * self.epochl
            #hrscoring = ampfun(self.score, ampFactor)
            #plotCanv.plot(taxis, hrscoring,  pen='k')
    def loadtextScore(self):
        return None
    def autoScore(self):
        return None
    def evaluateScore(self):
        return None
    def rescore(self):
        return None
    def stats(self):
        #print mean delta power across the whole recording

        return None
    def undo(self):
        return None
    def plotTrace(self):
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
        self.scplt = histCanv.plot(self.score)
        histCanv.setYRange(-1.1, 2.2, padding=0)
        histCanv.setXRange(0, len(self.score), padding=0)
        self.hypcrsr = histCanv.plot([self.currep, self.currep], [-1, 2], pen='r')
        #print(len(self.selchan2p))
        # ampFactor = self.sr * self.epochl
        if self.currep + self.halfneps < self.maxep:
            fp = int(max([0,((self.currep * self.epochl) - (self.tracedur/2))*self.sr]))
        else:
            fp = int(numpy.floor((self.maxep-self.neps)*self.epochl*self.sr))
        #lp = min([self.totalp-1,int(fp + self.tracedur*self.sr)-1])
        lp = int(numpy.floor(fp + self.tracedur * self.sr))
        #beginnning of each epoch:
        indxtep = self.t0[list(range(fp,lp,int(self.epochl * self.sr)))]
        indxeps = numpy.round(indxtep/self.epochl).astype(int)
        #print(indxeps)
        #print(fp,lp,self.selchan2p)
        c = 'kbrkbrkbrkbr'
        p = 0
        plotCanv = self.ui.PlotWidget_signals
        plotCanv.clear()
        for sch in self.selchan2p:
            plotCanv.plot(self.t0[fp:lp], self.edfmat[sch, fp:lp],  pen=c[p])
            p += 1
        plotCanv.setXRange(self.t0[fp],self.t0[lp], padding=0)
        if self.ui.checkBox_3.isChecked():
            ylim = self.maxEEG
        else:
            ylim = numpy.max(self.edfmat[sch, fp:lp])
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
        mt2p = self.edfmat[self.ui.EMGch.value()-1, fp:lp]
        if self.ui.listWidget_2.currentRow() == 1:
            mt2p = mt2p **2
        if self.ui.listWidget_2.currentRow() == 2:
            mt2p = scipy.signal.medfilt(mt2p,5)
        plotCanvMT.plot(self.t0[fp:lp], mt2p, pen='k')
        plotCanvMT.setXRange(self.t0[fp], self.t0[lp], padding=0)
        if self.ui.checkBox_4.isChecked():
            ylim = self.maxEMG
        else:
            ylim = numpy.max(mt2p)
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
        indx=numpy.arange(len(fft2pl))
        self.pfft = plotCanvFFT.plot(indx, fft2pl,  fillLevel=0, brush=(50,50,200,100))
        plotCanvFFT.setXRange(indx[0],indx[-1], padding=0)
        if self.ui.checkBox_6.isChecked():
            ylim = self.maxFFT
        else:
            ylim = numpy.max(fft2pl)
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
            self.ui.label_11.setText("Reading EDF...")
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
            self.edfmat = numpy.asarray(edf.get_data())
            npep = int(self.epochl * self.sr)
            leeg= len(self.edfmat[0,:])
            print("Calculating FFT from channel",self.ui.EEGch.value())
            print("last EEG point:",npep * (leeg//npep))
            #self.edfmat[sch, fp:lp], pen=c[p])
            self.eegmat = self.edfmat[self.ui.EEGch.value()-1,0:npep * (leeg//npep)].reshape(leeg//npep,npep)
            print(numpy.shape(self.eegmat))
            self.fftmat = numpy.zeros(numpy.shape(self.eegmat))
            self.freqs = numpy.fft.fftfreq(npep, 1/self.sr)
            self.score= -1* numpy.ones(self.maxep)
            for epoch in range(leeg//npep):
                self.fftmat[epoch,:] = numpy.abs(numpy.fft.fft(self.eegmat[epoch,:]))**2
                #We only one the freqs between 0 and the max freq FFT

            #now finding the pos of the 0 and maxfrec
            pos0 = numpy.argmin(numpy.abs(self.freqs))
            posmax = numpy.argmin(numpy.abs(self.freqs - self.ui.maxF.value()))
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
        indx = numpy.arange(len(fft2pl))
        self.pfft = plotCanvFFT.plot(indx, fft2pl, fillLevel=0, brush=(50, 50, 200, 100))
        plotCanvFFT.setXRange(indx[0], indx[-1], padding=0)
        if self.ui.checkBox_6.isChecked():
            ylim = self.maxFFT
        else:
            ylim = numpy.max(fft2pl)
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
        self.ui.label_11.setText("Ready")
        self.scoredict = {'score': self.score, 'zt0': self.ui.timeEdit_3.time(), 't0': self.ui.dateTimeEdit_2.dateTime(), 'el': 4}

    def saveAll(self): #this is from ca imager
        #making data frame with the Z scores of ever cell for every state, assuming 1 period
        #For each cell we need a table with the Zscore of an epoch and it's state
        mat_aux=[]
        k0=1
        lstate = ['Wake'] * len(self.activityW[1, :]) + ['NREM'] * len(self.activityN[1, :]) + ['REM'] * len(
            self.activityR[1, :])
        #there can be statistical differences due to larger than the rest or smaller than the rest activity
        wakeL =[]
        nremL = []
        remL = []
        wakeS = []
        nremS = []
        remS = []
        indep = []
        faL=[]
        oaL=[]
        faS=[]

        lfwa1 = int(len(self.fwa)/2)
        lowa1 = int(len(self.owa)/2)
        for n in range(self.Ncells):
            lact=list(self.activityW[n,:]) + list(self.activityN[n,:])+list(self.activityR[n,:])

            #lstate = ('WAKE ' * len(self.activityW)).split()+ ('NREM ' * len(self.activityN)).split() + ('REM ' * len(self.activityR)).split()
            F, p = stats.f_oneway(self.activityW[n,:],self.activityN[n,:],self.activityR[n,:])
            #print(self.activityW[n,:].mean(),self.activityN[n,:].mean(),self.activityR[n,:].mean())
            if p<0.05:
                print("Significant! N=",k0)
                k0+=1
                if (self.activityW[n,:].mean()>self.activityN[n,:].mean()) and (self.activityW[n,:].mean()>self.activityR[n,:].mean()):
                    wakeL.append(n)
                elif (self.activityW[n,:].mean()<self.activityN[n,:].mean()) and (self.activityW[n,:].mean()<self.activityR[n,:].mean()):
                    wakeS.append(n)
                elif (self.activityN[n,:].mean()<self.activityW[n,:].mean()) and (self.activityN[n,:].mean()<self.activityR[n,:].mean()):
                    nremS.append(n)
                elif (self.activityN[n,:].mean()>self.activityW[n,:].mean()) and (self.activityN[n,:].mean()>self.activityR[n,:].mean()):
                    nremL.append(n)
                elif (self.activityR[n,:].mean()<self.activityW[n,:].mean()) and (self.activityR[n,:].mean()<self.activityN[n,:].mean()):
                    remS.append(n)
                elif (self.activityR[n,:].mean()>self.activityW[n,:].mean()) and (self.activityR[n,:].mean()>self.activityN[n,:].mean()):
                    remL.append(n)
            else:
                indep.append(n)
            F, p1 = stats.f_oneway(self.fwa[n,0:lfwa1],self.owa[n,0:lowa1]) # Checking only the first half ogf the data
            F, p2 = stats.f_oneway(self.fwa[n, lfwa1:],
                                   self.owa[n, lowa1:])  # Checking only the second half ogf the data
            if (p1<0.05) and (p2<0.05):
                if (self.fwa[n,0:lfwa1].mean()>self.owa[n,0:lowa1].mean()) and (self.fwa[n,lfwa1:].mean()>self.owa[n,lowa1:].mean()):
                    faL.append(n) #saves the number of the cell that has higher activity at the onset of W
                else:
                    faS.append(n)
            else:
                oaL.append(n)
        print("percentage of active cells at onset: ",100*len(faL)/self.Ncells)
        print("percentage of less active cells at onset: ", 100 * len(faS) / self.Ncells)
        print("percentage of cells indifferent to onset: ", 100 * len(oaL) / self.Ncells)
        mixed = wakeS +nremS +remS
        if len(wakeL +nremL +remL +mixed +indep ) != self.Ncells:
            print("missing cells!!")
        #Making final figure with traces and hypnogram, jellybeans and pie chart
        labels =[]
        sizes =[]
        explode = []
        colors =[]
        colorlist = 'g', 'r', 'b', (0.5,0.5,0.5), 'y'
        labellist = 'W', 'R', 'NR', 'Mixed', 'Ind'
        i=0
        for m in [wakeL,remL, nremL, mixed,indep]:
            if len(m)>0:
                labels.append(labellist[i])
                sizes.append(100*len(m)/self.Ncells)
                explode.append(0)
                colors.append(colorlist[i])
            i+=1

        #To do: add first vs other activity during W
        #self.foa

        #making summary figure with jellybeans,
        fig=plt.figure()
        grid = plt.GridSpec(2, 3, wspace=0.0, hspace=0.1)
        plt.subplot(grid[0, 0])
        plt.imshow(self.mat2plot)
        plt.axis("off")
        plt.draw()
        ax1 = plt.subplot(grid[0, 1])
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=False, textprops={'size': 'x-large', 'weight':'bold'}, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax=plt.subplot(grid[0, 2]) #plotting bar plot with the difference in activity for cells who had larger activity during hte first W bout
        meanw,meann,meanr = self.activityW.mean(axis=1), self.activityN.mean(axis=1),self.activityR.mean(axis=1) #mean activity for every cell
        error = [(0,0,0),[stats.sem(meanw),stats.sem(meann), stats.sem(meanr)]]
        bp=plt.bar([0,1,2], [meanw.mean(),meann.mean(),meanr.mean()],
                   yerr=error,align='center',alpha=1, ecolor='k',capsize=5)
        plt.xticks([0, 1,2], ('WAKE', 'NREM','REM'))
        plt.ylabel('Mean Z score')
        bp[0].set_color('g')
        bp[1].set_color('b')
        bp[2].set_color('r')

        # plt.bar([0,1],[self.fwa[faL, :].mean(), self.owa[faL, :].mean()])
        # plt.xticks([0,1], ('Onset W', 'Within W'))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        #plt.box(on=None)
        ax = plt.subplot(grid[1, 0:])
        #now plotting hypnogram and traces
        tscore =[t*self.epochl for t in range(len(self.chunk_score))]
        taxis = (numpy.arange(self.nframes)) / self.sr
        plt.plot(tscore[3:], self.chunk_score[3:]*4.5)
        indxt = list(range(self.Ncells))
        random.shuffle(indxt)
        traceindx = indxt[0:8]
        for i in range(len(traceindx)):
            plt.plot(taxis[int(self.sr*12):], self.Traces[i,int(self.sr*12):].T + 10 * (i+1),linewidth=0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
        ax.tick_params(labelleft=False)
        #plt.box(on=None)
        #ax.set_frame_on(False)
        #ax.add_axes([0., 1., 1., 0])
        #ax = plt.axes([0, 1., 0, 1.])
        #ax.get_xaxis().set_visible(True)
        #ax.get_yaxis().set_visible(False)
        plt.show()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Q:
            print("Going to the next NREM epoch...")
        if event.key() == Qt.Key_Down:
            if (self.currep - self.neps) >= 0:
                self.ui.lineEdit.setText(str(int(self.currep - self.neps+1)))
                self.update_currep()
            else:
                self.ui.lineEdit.setText(str(int(self.halfneps)))
                self.update_currep()
        if event.key() == Qt.Key_Up:
            if (self.currep + self.neps) < (self.maxep):
                self.ui.lineEdit.setText(str(int(self.currep + self.neps-1)))
                self.update_currep()

            else:

                self.ui.lineEdit.setText(str(int(self.maxep-self.halfneps)))
                self.update_currep()

        if event.key() == Qt.Key_Right:
            if (self.currep + 1) < (self.maxep):
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
                self.update_currep()
        if event.key() == Qt.Key_Left:
            if self.currep > 0:
                self.ui.lineEdit.setText(str(int(min([self.currep - 1,self.maxep-1]))))
                self.update_currep()
                # Scoring
        if event.key() == Qt.Key_0:
            self.score[self.currep] = 0
            if self.currep < self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key_1:
            self.score[self.currep] = 1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key_5:
            self.score[self.currep] = 2
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key_7:
            self.score[self.currep] = 1.1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key_8:
            self.score[self.currep] = 2.1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key_9:
            self.score[self.currep] = 0.1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        #same for normal keyboard in case there is no numpad:
        if event.key() == Qt.Key_M:
            self.score[self.currep] = 0
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key_J:
            self.score[self.currep] = 1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key_K:
            self.score[self.currep] = 2
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        
        if event.key() == Qt.Key_I:
            self.score[self.currep] = 2.1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        if event.key() == Qt.Key_O:
            self.score[self.currep] = 0.1
            if self.currep <self.maxep:
                self.ui.lineEdit.setText(str(int(min([self.currep + 1,self.maxep-1]))))
            self.update_currep()
        #jumps to next non scored epoch
        if event.key() == Qt.Key_N:
            lnz = numpy.where(self.score[self.currep:] < 0)[0][0]+self.currep
            self.currep = lnz
            self.ui.lineEdit.setText(str(int(min([self.currep - 1, self.maxep - 1]))))
            self.update_currep()


        if event.key() == Qt.Key_Q:
            c1=self.score[self.currep:] > 0.9
            c2=self.score[self.currep:] < 1.9
            self.currep = numpy.argwhere(c1 & c2)[0]+self.currep
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.update_currep()
        if event.key() == Qt.Key_R:
            c1 = self.score[self.currep:] > 1.9
            c2 = self.score[self.currep:] < 2.9
            self.currep = numpy.argwhere(c1 & c2)[0] + self.currep
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.update_currep()
        if event.key() == Qt.Key_W:
            c1 = self.score[self.currep:] >=0
            c2 = self.score[self.currep:] < 1
            self.currep = numpy.argwhere(c1 & c2)[0] + self.currep
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.update_currep()
        if event.key() == Qt.Key_U:
            c1 = self.score[self.currep:] <0
            c2 = self.score[self.currep:] > 3
            self.currep = numpy.argwhere(c1 | c2)[0] + self.currep
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.ui.lineEdit.setText(str(int(self.currep)))
            self.update_currep()

if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())

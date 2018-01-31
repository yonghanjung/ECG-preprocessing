import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import pickle

class PrepECG(object):
    def __init__(self):
        self.MITBIH_idx = [100, 101, 102, 103, 104,
                      105, 106, 108, 109,
                      111, 112, 113, 114, 115,
                      116, 117, 118, 119, 120,
                      121, 122, 123, 124,
                      200, 201, 202, 203, 205,
                      207, 208, 209, 210,
                      212, 213, 214, 215, 217,
                      219, 220, 221, 222, 223,
                      228, 230, 231, 232, 233,
                      234
                      ]
        self.dist_r = 180 # 128 points to the left and right from R peak
        self.num_beat_limit = 30


    def ReadECG(self,num_record):
        '''
        Reading ECG record from 'Data' folder
        :param num_record: index number of the record
        :return: ECG record as array
        '''
        file_path = 'Data/' + str(num_record) + '_file.mat'
        ECG_record = scipy.io.loadmat(file_path)['val'][0]
        return ECG_record

    def ReadAnno(self, num_record):
        '''
        Reading ECG record from 'Data' folder
        :param num_record: index number of the record
        :return: Annotation
        '''
        file_path = 'Data/' + str(num_record) + '_anno.txt'
        ECG_anno = pd.read_fwf(file_path)
        return ECG_anno

    def Anno_Epi(self, Anno):
        '''
        Adding Episode label for every beat
        :param Anno: Annotation file
        :return:
        '''
        epi = Anno.iloc[0]['Aux']
        len_Anno = len(Anno)

        for idx in range(1,len_Anno):
            if pd.isnull(Anno.iloc[idx]['Aux']) == True:
                Anno._set_value(idx, 'Aux', epi)
            else:
                epi = Anno.iloc[idx]['Aux']
                Anno._set_value(idx, 'Aux', epi)
        return Anno

    def label_converter(self, beat_label):
        if beat_label in ['(N']:
            return 'N'
        elif beat_label in ['(AF','(AFIB','(AFL']:
            return 'AF'
        elif beat_label in ['(P','(B','(VT','(T',
                            '(SV','(IV','(NOD','(SVTA',
                            '(VFL','(IVR','(SVT','(AB',
                            '(PREX','(BII','(SBR']:
            return 'O'
        else:
            return 0

    def InitSeg(self):
        SegDict = dict()
        label_counter = dict()
        for label in ['N', 'AF', 'O']:
            SegDict[label] = [[]]
            label_counter[label] = 0
        return SegDict, label_counter


    def MakeRoomSegDict(self, SegDict, prev_label_counter, curr_label_counter):
        for label in ['N', 'AF', 'O']:
            if prev_label_counter[label] < curr_label_counter[label]:
                SegDict[label].append([])
            else:
                pass

        return SegDict

    def SegECG(self, ECG, Anno, SegDict, prev_label_counter):
        '''
        Given ECG and Annotaiton file,
        :param ECG: ECG record from ReadECG
        :param Anno: Annotation file from SegECG (after Anno-Epi)
        :return:
        '''

        R_list = list(Anno['Sample #'])
        ECG_len = len(ECG)

        prev_r = 0
        init_beat = False
        record_label_counter = dict()
        curr_label_counter = dict( )
        for label in ['N','AF','O']:
            record_label_counter[label] = 0
            curr_label_counter[label] = prev_label_counter[label]

        for r in R_list:
            if r > self.dist_r and r + self.dist_r < ECG_len: # if r is in the middle of the record
                if init_beat == False:
                    beat = list(ECG[range(r-self.dist_r, r+self.dist_r)])
                    init_beat = True
                elif init_beat == True:
                    beat = list(ECG[range( prev_r+self.dist_r,
                                           r + self.dist_r)])

                beat_label = self.label_converter( list(Anno[Anno['Sample #'] == r]['Aux'])[0] )
                if beat_label == 0:
                    prev_r = r
                    continue
                else:
                    if record_label_counter[beat_label] < self.num_beat_limit:
                        SegDict[beat_label][prev_label_counter[beat_label]] += beat
                        record_label_counter[beat_label] += 1
                        prev_r = r
                    else:
                        prev_r = r
                        continue

            else: # If r is at the very beginning stage or very end phase
                continue

        # del (prev_label_counter)
        for label in ['N','AF','O']:
            if record_label_counter[label] > 0:
                curr_label_counter[label] += 1
            else:
                pass

        return SegDict, curr_label_counter

    def Preprocessing(self):
        SegDict, prev_label_counter = self.InitSeg()
        for num_record in self.MITBIH_idx:
            print(num_record)
            ECG = self.ReadECG(num_record)
            Anno = self.Anno_Epi( self.ReadAnno(num_record) )
            SegDict, curr_label_counter = self.SegECG(ECG,Anno,SegDict, prev_label_counter)
            SegDict = self.MakeRoomSegDict(SegDict,prev_label_counter, curr_label_counter)
            prev_label_counter = curr_label_counter
        return SegDict

    def Graph_ECG(self, ECG):
        f = plt.figure()
        plot_ECG = f.add_subplot(111)
        plot_ECG.plot(ECG)
        return f

prep = PrepECG()
SegDict = prep.Preprocessing()
Label= ['N'] * len(SegDict['N']) + ['AF']*len(SegDict['AF']) + ['O']*len(SegDict['O'])
Data = SegDict['N'] + SegDict['AF'] + SegDict['O']

inclusion_idx = []
for idx in range(len(Data)):
    if len(Data[idx]) == 0:
        print(idx)
    else:
        inclusion_idx.append(idx)

Data = list(np.array(Data)[inclusion_idx])
Label = list(np.array(Label)[inclusion_idx])

pickle.dump(SegDict,open('SegECG.pkl','wb'))
pickle.dump(Label,open('label.pkl','wb'))
pickle.dump(Data,open('data.pkl','wb'))



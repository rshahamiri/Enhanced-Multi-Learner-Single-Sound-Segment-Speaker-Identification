from pydub import AudioSegment
from pydub.silence import split_on_silence
import subprocess
from os import path
import os
import pickle
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta
import pyprog
from PickleLargeFile import PickleLargeFile

SETTINGS_DIR = os.path.dirname(os.path.realpath('__file__'))

"""
A typical usage should call the following methods in the given oder:
    
    1. remove_silence_batch()
    2. extract_features_batch()
    3. create_target_vectors_batch()

Once done, the features and target vectors are pickled into mfcc60_features_timit and target_vectors_timit.

This process may take a long time depending on the size of the speech dataset.
"""

class FeatureExtractor():
    def __init__(self,root_databse_direcoty_name):
        self.root_db = root_databse_direcoty_name
        self.target_vectors_file_name=SETTINGS_DIR+'/training_data/target_vectors'
        self.feature_vectors_file_name=SETTINGS_DIR+'/training_data/mfcc60_features'

   
    
    def set_process_priority(self,set_high_priority=True):
        # this to run Spyder on high priority
        # Spyder must be executed with admin privilage. On terminal type:
        # sudo spyder
        """ Set the priority of the process to below-normal."""
        import sys
        try:
            sys.getwindowsversion()
        except AttributeError:
            isWindows = False
        else:
            isWindows = True
    
        if isWindows:
            # Based on:
            #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
            #   http://code.activestate.com/recipes/496767/
            import win32api,win32process,win32con
    
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            if (set_high_priority):
                win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
            else:
                win32process.SetPriorityClass(handle, win32process.NORMAL_PRIORITY_CLASS)
        else:
            import psutil
            print("Process PID:", os.getpid())
            p= psutil.Process(os.getpid())
            if (set_high_priority):
                p.nice(-1)
            else:
                p.nice(0)
            print (p.nice())
        return
    
    def remove_silence(self,file_name):
        #subprocess.call(["afplay", file_name])
        sound = AudioSegment.from_wav(file_name)
        a=split_on_silence(sound,min_silence_len=19,keep_silence=15,silence_thresh=sound.dBFS-3)
        newsound=AudioSegment.empty()
        for seg in a:
            newsound=newsound+seg
        newsound.export(file_name,format="wav")
        #subprocess.call(["afplay", file_name])
        return
    
    def remove_silence_batch(self):
    # this method removes silence from all files in the root_databse_direcoty_name)
        for directory, s, files in os.walk(self.root_db):
            speaker_name=directory[len(self.root_db):]
            print("Removing silence for speaker "+speaker_name)
            for f in files:
                file_path=directory+"/"+f
                if ("wav" in file_path):
                    self.remove_silence(file_path)

    def feature_extraction(self,audio_file):
        (rate,signal_values) = wav.read(audio_file)
        # Exctarcing MFCCs. The arguments are:
        # numcep – the number of cepstrum to return, default 13
        # winlen – the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        # winstep – the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        # appendEnergy:if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
        
        mfccs = mfcc(signal_values,rate, numcep=20, winlen=0.020, winstep=0.005, appendEnergy=False)
        # Getting the deltas
        delta1 = delta(mfccs,1)
        delta2 = delta(delta1,1)
        deltas =np.append(delta1,delta2, axis=1)
        mfccs= np.append(mfccs, deltas, axis=1)
          
        # Flatten the fetures as one dimension array
        mfccs=mfccs.flatten()
        return mfccs
    
    def extract_features_batch(self,number_of_utterances_per_speaker=3):
        # this method extarcts features from all audio files in root_databse_direcoty_name)
        number_of_features=20 * 3
        # n is number of frames, i.e. speech segments
        n=1
        #mfccs = pd.DataFrame()
        mfccs_list=[]
        speaker_no=0
        
        prog = pyprog.ProgressBar("Extracting Features ", " Done",len(next(os.walk(self.root_db))[1]))
         # Show the initial status
        prog.update()
        
        # RUNNING THIS MAY TAKE A LONG TIME. LOAD PICKLE INSTEAD
        for directory, s, files in os.walk(self.root_db):
            speaker_name=directory[len(self.root_db):]
            processed_utterances_per_speaker=0
            prog.set_stat(speaker_no)
            prog.update()
            speaker_no=speaker_no+1
            for f in files:
                file_path=directory+"/"+f
                if ("wav" in file_path):
                    if (processed_utterances_per_speaker==number_of_utterances_per_speaker):
                        break
                    features=self.feature_extraction(file_path)
                    #print ("Processing",file_path)
                    # Each audio file is converted to several samples by selecting 20 frames each contain 780 features 
                    # for each sample from the extracted features
                    number_of_samples = len (features) // (n*number_of_features)
                    
                    #print (speaker_name,"number of samples:", number_of_samples)
                    t = pd.DataFrame ( data = [features[0:n*number_of_features]], index = [speaker_name] )
                    #mfccs = pd.concat ( [mfccs,t])
                    mfccs_list.append(t)
                    for i in range(1,number_of_samples-1):
                        l = number_of_features*i*n
                        h=number_of_features * (i+1)*n
                        t = pd.DataFrame ( data = [features[l:h]], index = [speaker_name] )
                        mfccs_list.append(t)
                        #mfccs = pd.concat ( [mfccs,t])
                    processed_utterances_per_speaker = processed_utterances_per_speaker+1

        self.mfccs=pd.concat(mfccs_list,axis=0)
        del mfccs_list
        prog.end()
        # Remove possible Nan
        #mfccs=mfccs.fillna(0)
        
        number_of_samples = len (self.mfccs)
        number_of_features = len (self.mfccs.keys())
        number_of_speakers = len(self.mfccs.index.unique())
        print ("\nAll feaures are extracted:")
        print ("No. Sound Segment Samples Generated:",number_of_samples, "No. Speakers Generated:",number_of_speakers, "No. MFCCs Generated:", number_of_features)

        print ("\nSaving the extracted features in",self.feature_vectors_file_name)
        # Saving the mfcc features into pickle
        # Use the following code in case mfccss are more than 4GB
        pic=PickleLargeFile()
        pic.pickle_dump(self.mfccs,self.feature_vectors_file_name)

        print ("\nDone.")

    def create_target_vectors_batch(self):
        
        # Loading the mfcc features into a Pandas DataFrame from a pickle
        pic=PickleLargeFile()
        self.mfccs=pic.pickle_load(self.feature_vectors_file_name)
       
        number_of_samples = len (self.mfccs)
        number_of_features = len (self.mfccs.keys())
        number_of_speakers = len(self.mfccs.index.unique())
        print ("\nNo. Speech Segment Samples Loaded:",number_of_samples, "\nNo. Speakers Loaded:",number_of_speakers, "\nNo. MFCCs Loaded per speech segment:", number_of_features)
        # Preparing a target vector for each speaker
        list_of_speakers = self.mfccs.index.unique()
        i=0
        targets_per_speaker_list=[]
        for speaker in list_of_speakers:
            vector = np.zeros(number_of_speakers)
            vector[i]=1
            t = pd.DataFrame( data = [vector], index =[speaker])
            targets_per_speaker_list.append(t)
            i+=1
        self.targets_per_speaker = pd.concat(targets_per_speaker_list,axis=0)  
        del targets_per_speaker_list  
        # Creating the target vectors
        prog = pyprog.ProgressBar("Creating Target Vector ", " Done",number_of_samples)
        # Show the initial status
        prog.update()
        targets_list=[]
        counter=1
        for speaker in self.mfccs.index:
            d = np.array( self.targets_per_speaker.loc[speaker])
            t = pd.DataFrame(data = [d], index = [speaker])
            targets_list.append(t)
            counter=counter+1
            prog.set_stat(counter)
            prog.update()

        self.targets = pd.concat(targets_list,axis=0) 
         # Saving the target vectors into pickle       
        pic.pickle_dump(self.targets,self.target_vectors_file_name)
       
        del targets_list
        print ("\nAll Target vectors are created.")
        print ("\nSaving the target vectors in", self.target_vectors_file_name+"...")      
        print ("\nDone")
        prog.end()
        return


# This part of the code is not executed outside the class
if __name__ == '__main__':
    
    p=FeatureExtractor("./TIMIT/")
    #p.batch_remove_silence()
    #p.extract_features_batch()
    #p.create_target_vectors_batch()
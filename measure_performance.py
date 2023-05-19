import os
from emal_speaker_identification import EMALSpeakerIdentifier
from PickleLargeFile import PickleLargeFile
import keras
import pyprog
import numpy as np
import pandas as pd
SETTINGS_DIR = os.path.dirname(os.path.realpath('__file__'))
emal_si=EMALSpeakerIdentifier(SETTINGS_DIR+"/TIMIT")

# Loading the test data
pic=PickleLargeFile()
emal_si.X_train     = pic.pickle_load(SETTINGS_DIR+'/training_data/X_train')
emal_si.y_train = pic.pickle_load(SETTINGS_DIR+'/training_data/y_train')
emal_si.X_test     = pic.pickle_load(SETTINGS_DIR+'/training_data/X_test')
emal_si.y_test = pic.pickle_load(SETTINGS_DIR+'/training_data/y_test')


emal_si.load_dnns()

training_performance=pd.DataFrame(
        columns=['loss','accuracy','mse','dnn index']
        )

testing_performance=pd.DataFrame(
        columns=['loss','accuracy','mse','dnn index']
        )
for i in range(emal_si.number_of_speakers):
   print ("DNN",i)
   # Training evaluation
   dnn=emal_si.networks[i] 
   dnn.compile(loss='binary_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=0.0001),
                          metrics=['accuracy','mean_squared_error'])
   a=dnn.evaluate(emal_si.X_train,emal_si.y_train[:,i])
   a.append(i)
   training_performance .loc[ len (training_performance)] =a
   # Testing evaluation
   b=dnn.evaluate(emal_si.X_test,emal_si.y_test[:,i])
   b.append(i)
   testing_performance .loc[ len (testing_performance)] =b


training_performance.to_csv(SETTINGS_DIR+'/training_performance.csv')
testing_performance.to_csv(SETTINGS_DIR+'/testing_performance.csv')

training_performance.describe()
testing_performance.describe()

# predict the first DNN
prog = pyprog.ProgressBar("Prediction ", " Done",emal_si.number_of_speakers)
prog.update()
y_pred=pd.DataFrame( data= emal_si.networks[0].predict( emal_si.X_test) )
# Predicting the rest of the DNNs
for i in range(1,emal_si.number_of_speakers):
    dnn=emal_si.networks[i] 
    y_pred[i]=dnn.predict(emal_si.X_test)
    prog.set_stat(i)
    prog.update()
prog.end()
y_pred.to_csv(SETTINGS_DIR+'/predictions.csv')


#correct=0
#for i in range( len(y_pred)):
#    row_pred = y_pred.iloc[i]
#    row_true = emal_si.y_test[i]
#    if np.argmax(row_pred) == np.argmax(row_true):
#        correct+=1
#
#print (correct / len(y_pred) *100)
#
#unperforming_dnns=[]
#for i in range ( len(testing_performance)):
#    loss = testing_performance.iloc[i]['loss']
#    accuracy=testing_performance.iloc[i]['accuracy']
#    if loss > 0.08 or accuracy < 0.97:
#        unperforming_dnns.append(i)
#
#for dnn_index in unperforming_dnns:
#    emal_si.train_larger_DNNs( dnn_start_index=dnn_index,dnn_end_index=dnn_index+1,ideal_loss=0.07, is_dnn_structure_changned=True)

        
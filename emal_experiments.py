import os
from emal_speaker_identification import EMALSpeakerIdentifier

SETTINGS_DIR = os.path.dirname(os.path.realpath('__file__'))

results=[]
training_time=[]

# You can loop the following code for corss validation
# Nevertheless, generate_random_training_testing_data() takes a long time

# Create X and y data
emal_si=EMALSpeakerIdentifier(SETTINGS_DIR+"/TIMIT")
m=emal_si.get_model()
m.summary()
#emal_si.generate_random_training_testing_data()
#emal_si.set_process_priority()  


# train the MVSL SI CNN
# Run each train on a different machine for parallel processing
ideal_loss=0.03
emal_si.train(dnn_start_index=0,dnn_end_index=20,
              ideal_loss=ideal_loss,
              is_dnn_structure_changned=True)




training_time.append(emal_si.training_time)
# test the DNN
res,predictions= emal_si.test()
print (res)





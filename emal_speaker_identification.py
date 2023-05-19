import numpy as np
import pandas as pd
import os
import pickle
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Convolution1D, MaxPooling1D
from keras.optimizers import RMSprop, Adam
from keras import losses
import matplotlib.pyplot as plt
from feature_extractor import FeatureExtractor
from keras.models import model_from_json
from PickleLargeFile import PickleLargeFile
import pyprog

SETTINGS_DIR = os.path.dirname(os.path.realpath('__file__'))


class EMALSpeakerIdentifier():
    def __init__(self, root_databse_direcoty_name):
        fe = FeatureExtractor(root_databse_direcoty_name)

        # Loading the mfcc features into a Pandas DataFrame from a pickle
        pic = PickleLargeFile()
        self.mfccs = pic.pickle_load(fe.feature_vectors_file_name)

        self.number_of_samples = len(self.mfccs)
        self.number_of_features = len(self.mfccs.keys())
        self.number_of_speakers = len(self.mfccs.index.unique())
        self.networks = []
        print("No. sound segment samples loaded:", self.number_of_samples, "No. Speakers :",
              self.number_of_speakers, "No. MFCCs per sound segment:", self.number_of_features)

        # Loading the target vectors into a Pandas DataFrame from a pickle
        self.targets = pic.pickle_load(fe.target_vectors_file_name)

    def set_process_priority(self, set_high_priority=True):
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
            import win32api
            import win32process
            import win32con

            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(
                win32con.PROCESS_ALL_ACCESS, True, pid)
            if (set_high_priority):
                win32process.SetPriorityClass(
                    handle, win32process.HIGH_PRIORITY_CLASS)
            else:
                win32process.SetPriorityClass(
                    handle, win32process.NORMAL_PRIORITY_CLASS)
        else:
            import psutil
            print("Process PID:", os.getpid())
            p = psutil.Process(os.getpid())
            if (set_high_priority):
                p.nice(-1)
            else:
                p.nice(0)
            print(p.nice())
        return

    def generate_random_training_testing_data(self):

        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []

        prog = pyprog.ProgressBar(
            "Train/Test Data Generation ", " Done", len(self.mfccs.index.unique()))
        # Show the initial status
        prog.update()
        counter = 0
        for speaker in self.mfccs.index.unique():
            counter = counter+1
            # getting all features and terget vecotrs
            no_of_samples = len(self.mfccs.loc[speaker])
            sp_features = self.mfccs.loc[speaker]
            sp_targets = self.targets.loc[speaker]
            no_traingin_samples = int((no_of_samples-1) * 0.8)
            for i in range(no_traingin_samples):
                rnd = random.randint(0, len(sp_features)-1)
                x = pd.DataFrame(
                    data=[sp_features.iloc[rnd].values], index=[speaker])
                X_train_list.append(x)
                # X_train=pd.concat([X_train,x])
                sp_features = pd.DataFrame(
                    np.delete(sp_features.values, [rnd], axis=0))
                y = pd.DataFrame(
                    data=[sp_targets.iloc[rnd].values], index=[speaker])
                y_train_list.append(y)
                # y_train=pd.concat([y_train,y])
                sp_targets = pd.DataFrame(
                    np.delete(sp_targets.values, [rnd], axis=0))
            #X_test = pd.concat([X_test,sp_features])
            X_test_list.append(sp_features)
            #y_test = pd.concat ([y_test,sp_targets])
            y_test_list.append(sp_targets)
            prog.set_stat(counter)
            prog.update()
            #print ("X and y data for speaker",speaker,"are generated.")

        X_train = pd.concat(X_train_list, axis=0)
        del X_train_list
        y_train = pd.concat(y_train_list, axis=0)
        del y_train_list
        X_test = pd.concat(X_test_list, axis=0)
        del X_test_list
        y_test = pd.concat(y_test_list, axis=0)
        del y_test_list
        # Converting them to numpy arrays
        self.X_train = X_train.values
        self.X_test = X_test.values
        self.y_train = y_train.values
        self.y_test = y_test.values
        del X_train, X_test, y_train, y_test
        # checking for correctness
        if (len(self.X_train) != len(self.y_train)) or (len(self.X_test) != (len(self.y_test))):
            print("ERROR!")
        else:
            print(len(self.X_train), "trainin samples and", len(
                self.X_test), " testing samples are generated.")
            # saving X and y data
            pic = PickleLargeFile()
            pic.pickle_dump(self.X_train, SETTINGS_DIR +
                            '/training_data/X_train')
            pic.pickle_dump(self.X_test, SETTINGS_DIR+'/training_data/X_test')
            pic.pickle_dump(self.y_train, SETTINGS_DIR +
                            '/training_data/y_train')
            pic.pickle_dump(self.y_test, SETTINGS_DIR+'/training_data/y_test')

    def set_data(self):
        pic = PickleLargeFile()
        self.X_train = pic.pickle_load(SETTINGS_DIR+'/training_data/X_train')
        #self.X_test = pic.pickle_load(SETTINGS_DIR+'/training_data/X_test')
        self.y_train = pic.pickle_load(SETTINGS_DIR+'/training_data/y_train')
        #self.y_test = pic.pickle_load(SETTINGS_DIR+'/training_data/y_test')
        
    def get_model(self):
            model = Sequential()

            model.add(Convolution2D(filters=32,
                                    kernel_size=(3, 1),
                                    input_shape=(
                                        self.number_of_features, 1, 1),
                                    activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 1)))

            model.add(Convolution2D(filters=64,
                                    kernel_size=(3, 1),
                                    activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 1)))

            model.add(Flatten())

            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))

            model.add(Dense(1, activation='sigmoid'))
            return model

    def train(self, batch_size=256, dnn_start_index=0, dnn_end_index=None,
              ideal_loss=0.1, is_dnn_structure_changned=False,
              learning_rate=0.001,  max_epoch=300):
        # Load X and y
        pic = PickleLargeFile()
        self.X_train = pic.pickle_load(SETTINGS_DIR+'/training_data/X_train')
        #self.X_test = pic.pickle_load(SETTINGS_DIR+'/training_data/X_test')
        self.y_train = pic.pickle_load(SETTINGS_DIR+'/training_data/y_train')
        #self.y_test = pic.pickle_load(SETTINGS_DIR+'/training_data/y_test')

        # Normalize data for CNN
#        from sklearn.preprocessing import StandardScaler
#        sc_X = StandardScaler()
#        self.X_train = sc_X.fit_transform(self.X_train)
#        from sklearn.externals import joblib
#        joblib.dump(sc_X, SETTINGS_DIR+'/encoders/mfcc_normalizer.pkl')
#        print("Training data is Standardized")

        # Reshape X_train
        self.X_train = self.X_train.reshape(
            (self.X_train.shape[0], self.X_train.shape[1], 1, 1))

        if (dnn_end_index == None):
            dnn_end_index = self.number_of_speakers
            print("dnn_end_index is set to", dnn_end_index)

        import time
        start_time = time.time()
        for i in range(dnn_start_index, dnn_end_index):
            print("===================================================================")
            dnn_file_name_structure = SETTINGS_DIR + \
                "/dnns/mvml_si_dnn_"+str(i)+".json"
            dnn_file_name_weights = SETTINGS_DIR + \
                "/dnns/mvml_si_dnn_"+str(i)+"_weight.h5"
            if (os.path.isfile(dnn_file_name_structure) and
                    (os.path.isfile(dnn_file_name_weights)) and (is_dnn_structure_changned == False)):
                # load the previosly trained DNN
                json_file = open(dnn_file_name_structure, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)
                model.compile(loss='binary_crossentropy',
                              optimizer=keras.optimizers.Adam(
                                  lr=learning_rate),
                              metrics=['accuracy', 'mean_squared_error'])
                # load weights into the model
                model.load_weights(dnn_file_name_weights)
                print("CNN", i, "is loaded.")
            else:
                model = self.get_model()
               
                model.compile(loss='binary_crossentropy',
                              optimizer=keras.optimizers.Adam(lr=0.0001),
                              metrics=['accuracy', 'mean_squared_error'])
                print("CNN", i, "is created")

            # Balance the class weights
            #from sklearn.utils import class_weight
            #class_weights = class_weight.compute_class_weight('balanced', np.unique(self.y_train[:,i]), self.y_train[:,i])
            class_weights = {0: 1., 1: float(self.number_of_speakers-1)}
            history = model.fit(self.X_train, self.y_train[:,i],
                                batch_size=batch_size,
                                epochs=1,
                                class_weight=class_weights,
                                verbose=2)
            ep = 2
            while (history.history['loss'][0] >= ideal_loss):
                print("CNN", i, "Epoch", ep)
                history = model.fit(self.X_train, self.y_train[:,i],
                                    batch_size=batch_size,
                                    epochs=1,
                                    class_weight=class_weights,
                                    verbose=2)

                # Save/overwrite the model
                model_json = model.to_json()
                with open(dnn_file_name_structure, "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights(dnn_file_name_weights)
                ep += 1
                # stop the traning if certain training accuracy is reached
#                if (history.history['loss'][0]<ideal_loss):
#                    break
                if (ep > max_epoch):
                    break

            del model

        self.training_time = time.time() - start_time
        print("Training time:", self.training_time, "seconds")

        return

    def load_dnns(self):
        prog = pyprog.ProgressBar(
            "Loading EMAL CNNs ", " Done", self.number_of_speakers)
        prog.update()
        networks = []
        for i in range(self.number_of_speakers):
            # load json and create model
            json_file = open(
                SETTINGS_DIR+"/dnns/mvml_si_dnn_"+str(i)+".json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(
                SETTINGS_DIR+"/dnns/mvml_si_dnn_"+str(i)+"_weight.h5")
            networks.append(loaded_model)
            prog.set_stat(i)
            prog.update()
        prog.end()
        self.networks = networks
        del networks
        return

    def test(self, verbose=0):
        pic = PickleLargeFile()
        self.X_test = pic.pickle_load(SETTINGS_DIR+'/training_data/X_test')
        self.y_test = pic.pickle_load(SETTINGS_DIR+'/training_data/y_test')

#        # Normalize data for CNN
#        from sklearn.preprocessing import StandardScaler
#        from sklearn.externals import joblib
#        sc_X = joblib.load(SETTINGS_DIR+'/encoders/mfcc_normalizer.pkl')
#        self.X_test = sc_X.transform(self.X_test)
#        print("Testing data is standarized")

        # Reshape X_test
        self.X_test = self.X_test.reshape(
            (self.X_test.shape[0], self.X_test.shape[1], 1, 1))

        if (len(self.networks) < self.number_of_speakers):
            self.load_dnns()
        print("Testing", len(self.X_test), "samples on", self.number_of_speakers,
              "speakers. This will take 4-8 hours on a typical PC.")
        prog = pyprog.ProgressBar(
            "Prediction ", " Done", self.number_of_speakers)
        # Show the initial status
        prog.update()
        self.y_pred = self.networks[0].predict(self.X_test)
        for i in range(1, self.number_of_speakers):
            self.y_pred = np.append(
                self.y_pred, self.networks[i].predict(self.X_test), axis=1)
            prog.set_stat(i+1)
            prog.update()
        prog.end()
        print("Applying Softmax function...")
        number_of_correct_classificitation = 0
        for pred, test in zip(self.y_pred, self.y_test):
            if (np.argmax(pred) == np.argmax(test)):
                number_of_correct_classificitation += 1

        number_of_incorrect_classifications = len(
            self.y_test)-number_of_correct_classificitation
        self.accuracy = number_of_correct_classificitation/len(self.y_pred)*100
        mse = np.sum(np.power(self.y_test - self.y_pred, 2)) / \
            (len(self.y_pred)*self.number_of_speakers)
        nrmse = np.sqrt(mse)
        if (verbose != 0):
            print("----------------------------------")
            print("Testing Results:")
            print("Correct classification:",
                  number_of_correct_classificitation, "segments")
            print("Incorrect classification:",
                  number_of_incorrect_classifications, "frames")
            print("Accuracy (%):", self.accuracy)
            print("MSE=", mse)
            print("NRMSE (%) = ", nrmse*100)
#
        test_results = {
            "Number of correct classifications": number_of_correct_classificitation,
            "Number of Incorrect Classifications": number_of_incorrect_classifications,
            "Accuracy (%)": self.accuracy,
            "Mean Squared Error": mse,
            "Normalized Root Mean Squared Error (%)": nrmse*100
        }
        return test_results, self.y_pred


# This part of the code is not executed outside the class
if __name__ == '__main__':
    print("Compiled")
    # m=EMALSpeakerIdentifier(SETTINGS_DIR+"/TIMIT/")
    # m.train(epochs=10)
    # m.test()

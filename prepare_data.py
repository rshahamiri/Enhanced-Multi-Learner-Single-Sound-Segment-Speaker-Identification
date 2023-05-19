from feature_extractor import FeatureExtractor
import os


SETTINGS_DIR = os.path.dirname(os.path.realpath('__file__'))
# Prepare the audio data and extract the features
# run this only once
fe=FeatureExtractor(SETTINGS_DIR+"/TIMIT")

# this is to run Spyder on high priority
# Spyder must be executed with admin privilage. On terminal type:
# sudo spyder
#fe.set_process_priority(set_high_priority=True)

fe.remove_silence_batch()


fe.extract_features_batch(number_of_utterances_per_speaker=10)

fe.create_target_vectors_batch()



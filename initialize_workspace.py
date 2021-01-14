# function that initializes our workspace
import os
import argparse
from glob import glob
class workspace_class:
    def __init__(self,workspace_folder):
        self.workspace_folder = workspace_folder
        self.input_folder_full = os.path.join(self.workspace_folder,'full_dataset')
        self.input_folder_50ms = os.path.join(self.workspace_folder, 'fixed_50ms_dataset')
        self.result_folder = self.create_result_folder()
        self.annotations_folder = os.path.join(self.workspace_folder, 'annos')
        self.mic_folder = os.path.join(self.workspace_folder, 'mic')
        self.mix_folder = os.path.join(self.workspace_folder, 'audio_mono_pickup')
        self.audio_hex_folder = os.path.join(self.workspace_folder, 'hex_cln')
        self.model_folder = os.path.join(self.workspace_folder, 'models')


    def create_result_folder(self):
        path = os.path.join(self.workspace_folder, 'results')
        self.result_folder = path
        if os.path.isdir(path):
            pass
        else:
            try:
                os.mkdir(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)
            for midi in range(40,74):
                path = os.path.join(self.workspace_folder, 'results','c'+str(midi))
                if os.path.isdir(path):
                    pass
                else:
                    try:
                        os.mkdir(path)
                    except OSError:
                        print ("Creation of the directory %s failed" % path)
                    else:
                        print ("Successfully created the directory %s " % path)
        return self.result_folder

def initialize_workspace(workspace_folder):
    workspace = workspace_class(workspace_folder)
    return workspace

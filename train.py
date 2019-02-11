from demo_image import process,load_m
from config_reader import config_reader
from model.cmu_model import get_testing_model
import os
import json
import cv2
import pickle
def get_training_data_(dir_path,params,model_params):
    if not os.path.exists("./output_images"):
        os.mkdir("output_images")

    files = os.listdir(dir_path)
    
    count = 0
    train_data = []
    for f in files:
        train_image = cv2.imread(os.path.join(dir_path,f))
        print(os.path.join(dir_path,f))
        h,w,_= train_image.shape
        canvas,subset,candidate = process(os.path.join(dir_path,f),params,model_params,series = True)
        cv2.imwrite("output_images/output"+str(count)+".jpg",canvas)
        temp_tup = (f,(subset,candidate))
        train_data.append(temp_tup)
        with open("value.pickle","wb") as f:
            pickle.dump(train_data,f)
        
        count+=1

    

def get_images(pickle_path):
    with open("images.pickle","rb") as f:
        test = pickle.load(f)
    count = 0
    for t in test:
        print(t.shape)
        cv2.imwrite("output_images/output"+str(count)+".jpg",t)
        count+=1
    


def train_data_measurement(train_Data_path):
    with open("value.pickle","rb") as f:
        test = pickle.load(f)

    for t in test:
        print(t[0],t[1].shape)


if __name__ == "__main__":
    # keras_weights_file = "model.h5"
    # model = get_testing_model()
    # model.load_weights(keras_weights_file)
    load_m(None)
    params, model_params = config_reader()
    get_training_data_("sample_images/",params,model_params)
    # get_images(None)
    # train_data_measurement(None)

from demo_image import process,load_m
from config_reader import config_reader
from model.cmu_model import get_testing_model
import os
import json
import cv2
import pickle
import glob
import numpy as np

"""
Normalize the image coordinates depending upon the image size
"""
def normalize_cord(image,X,Y):
  h,w,_  = image.shape
  x = [X[0]/w,X[1]/w]
  y = [Y[0]/h,Y[1]/h]
  return x,y


"""
Run Pose Estimation Model on all the images and return a pickle file containing the coordinates of the pose on persons on the images 

"""
def get_training_data_(dir_path,params,model_params,output_pickle_file,notfight_flag = False):
    if not os.path.exists("./output_images"):
        os.mkdir("output_images")

    files = glob.glob(dir_path+"/*.jpg")
    # print()
    count = 0
    train_data = []
    for f in files:
        print(f)
        train_image = cv2.imread(f)
        # print(os.path.join(dir_path,f))
        h,w,_= train_image.shape
        canvas,subset,candidate = process(f,params,model_params,series = True)
        if not notfight_flag:
            cv2.imwrite("output_images/output_"+f.split("/")[1],canvas)
        else:
            cv2.imwrite("output_images/output_"+f.split("/")[2],canvas)
        temp_tup = (f,(subset,candidate))
        train_data.append(temp_tup)
        with open(output_pickle_file,"wb") as f:
            pickle.dump(train_data,f)
        
        count+=1

    

# def get_images(pickle_path):
#     with open(,"rb") as f:
#         test = pickle.load(f)
#     count = 0
#     for t in test:
#         print(t.shape)
#         cv2.imwrite("output_images/output2"+str(count)+".jpg",t)
#         count+=1
    

limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]
           
           
"""
Using the cordinates, get the points per person and return data of human that are most visible in the image
"""
def train_data_measurement(train_data_pickle_path):
    with open(train_data_pickle_path,"rb") as f:
        test = pickle.load(f)
    import numpy as np
    image_person = []
    for t in test:
        print(t[0],t[1][1].shape,t[1][0].shape)
        temp = cv2.imread(t[0])
        subset = t[1][0]
        candidate = t[1][1]
        person_points = []
        for n in range(len(subset)):
            temp_person = []
            count =0 
            for i in range(17):
                index = subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    continue
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
#                 print(temp.shape[1],temp.shape[0])
#                 print(Y,X)
                X,Y = normalize_cord(temp,Y,X)
#                 print(X,Y)
                temp_person.append([X,Y])
                count+=1
            if count > 6:
              person_points.append(temp_person)
        if len(person_points) == 2:
          image_person.append((t[0],person_points))
    return image_person

"""
Convert Raw Point data to numpy arry of training data of size (136). Pose of two people in the image are appended to great one vector.
returns Array of training data 

"""
def postprocess_train_data(image_person):
    train_ = []
    for j in range(len(image_person)):
        result  = np.array([])
        test_image_vector = image_person[j][1]
        for i in range(len(test_image_vector)):
            value = np.expand_dims(np.ndarray.flatten(np.asarray(test_image_vector[i])),axis = 0)
            size = value.shape[1]
            zero_  = np.zeros((1,68-size))
            value = np.concatenate((value,zero_),axis = 1)
            if result.size == 0 :
                result = value
            else:
                result = np.concatenate((result,value),axis = 1)
        #   print(value,value.shape)
        train_.append(result)
    return np.asarray(train_)


if __name__ == "__main__":
    # keras_weights_file = "model.h5"
    # model = get_testing_model()
    # model.load_weights(keras_weights_file)
    load_m(None)
    params, model_params = config_reader()
    fight_path = "sample_images"
    notfight_path = "sample_images/notfight"
    fight_pickle = "fight.pickle"
    notfight_pickle = "notfight.pickle"
    get_training_data_(fight_path,params,model_params,fight_pickle)
    # get_images(None)
    image_person = train_data_measurement(fight_pickle)
    fight_train  = postprocess_train_data(image_person)
    get_training_data_(notfight_path,params,model_params,notfight_pickle,notfight_flag = True)
    image_person = train_data_measurement(notfight_pickle)
    notfight_train = postprocess_train_data(image_person)
    train_data = np.concatenate((fight_train,notfight_train),axis = 0)
    train_data = np.squeeze(train_data,axis = 1)
    fight_size = fight_train.shape[0]
    notfight_size = notfight_train.shape[0]
    print(fight_size,notfight_size)
    train_y = np.zeros((train_data.shape[0],1))
    train_y[0:fight_size] = 1.0
    print(train_data)
    print(train_y)
    with open("train_data.pickle","wb") as f:
        pickle.dump(train_data,f)
    with open("train_data_y.pickle","wb") as f:
        pickle.dump(train_y,f)


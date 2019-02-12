from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config_reader import config_reader
import pickle
import numpy as np
from demo_image import process,load_m
import cv2
import sys
import argparse
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

def normalize_cord(image,X,Y):
  h,w,_  = image.shape
  x = [X[0]/w,X[1]/w]
  y = [Y[0]/h,Y[1]/h]
  return x,y

"""
Create_model()
Return's model created 
"""
def create_model():
  model = Sequential()
  model.add(Dense(120, input_dim=(136), kernel_initializer='normal', activation='relu'))
  model.add(Dense(60,input_dim = (120),kernel_initializer='normal',activation='relu'))
  model.add(Dense(1,input_dim = (60), kernel_initializer='normal', activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
seed = 7
np.random.seed(seed)


"""
Load train Data
"""
def load_train_data(train_data_path,train_data_path_y):
    with open(train_data_path,"rb") as f:
        train_ = pickle.load(f)
    
    with open(train_data_path_y,"rb") as f:
        train_y = pickle.load(f)

    return train_,train_y


"""
Fit Model to the train data, pickles the model after training
"""
def fit_model():
    model = create_model()
    train_data ,train_y_data = load_train_data("train_data.pickle","train_data_y.pickle")
    model.fit(train_data,train_y_data,batch_size=1,epochs= 100,shuffle= True)

    with open("model.pickle","wb") as f:
        pickle.dump(model,f)
    return model

"""
Class ActionClassifier

Load's model and then using pose estimation model, estimates the pose of the people on the scene. 
Using the pose , create the vector of the image for query (vector is normalized pose points)
using the vector, querys the trained model to predict the result. 

"""

class ActionClassifier:
    model = None
    @staticmethod
    def load_model(model_path):
        with open(model_path,"rb") as f:
            ActionClassifier.model = pickle.load(f)
        load_m(None)
    
    def __init__(self,model_path = "model.pickle"):
        ActionClassifier.load_model(model_path)


    def getImageVector(self,image):
        train_ = []
        temp = np.copy(image)
        params,model_params = config_reader()
        canvas,subset,candidate = process(image,params,model_params,series = True,model_call = True)
        cv2.imwrite("Result.jpg",canvas)
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
        if len(person_points) <= 1:
            return -1
        else:

            for i in range(len(person_points)):
                result = np.array([])
                for j in range(i+1,len(person_points)):
                    result  = np.array([])
                    value = np.expand_dims(np.ndarray.flatten(np.asarray(person_points[i])),axis = 0)
                    size = value.shape[1]
                    zero_  = np.zeros((1,68-size))
                    value = np.concatenate((value,zero_),axis = 1)
                    if result.size == 0 :
                        result = value
                    else:
                        result = np.concatenate((result,value),axis = 1)
                    
                    value = np.expand_dims(np.ndarray.flatten(np.asarray(person_points[j])),axis = 0)
                    size = value.shape[1]
                    zero_  = np.zeros((1,68-size))
                    value = np.concatenate((value,zero_),axis = 1)
                    if result.size == 0 :
                        result = value
                    else:
                        result = np.concatenate((result,value),axis = 1)

                    # print(result.shape)
                    train_.append(result)
        # print(train_)
        # train_ = np.ndarray(train_)
        train_ = np.asarray(train_)
        train_ = np.squeeze(train_,axis= 1)
        return canvas,train_
                    


    def classify(self,image):
        canvas,train_ = self.getImageVector(image)
        fight_flag = 0
        notfight_flag = 0
        for i in range(len(train_)):
            temp_ = train_[i]
            temp_ = np.expand_dims(temp_,axis = 0)
            result = ActionClassifier.model.predict(temp_)
            print("result",result)
            if(result[0] >= 0.90):
                fight_flag+=1
            else:
                notfight_flag+=1
        
        if fight_flag >= 1:
            temp = np.copy(canvas)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(temp,'Fight',(20,20), font, 0.5,(255,255,255),2,cv2.LINE_AA)
            print("fight")
            return "fight",temp
        else:
            temp = np.copy(canvas)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(temp,'NotFight',(20,20), font, 0.5,(255,255,255),2,cv2.LINE_AA)
            print("notfight")
            return "notfight",temp


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cross_val', default = "false",help="CrossValidate")
    parser.add_argument('--fit', default = "false", help="Fit Train Data")
    args = parser.parse_args()


    if args.cross_val == "true":
        """

        Check the performance of the model using CrossValidation

        """
        estimators = []
        # estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=100, batch_size=4, verbose=0)))
        pipeline = Pipeline(estimators)
        train_data ,train_y_data = load_train_data("train_data.pickle","train_data_y.pickle")
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        results = cross_val_score(pipeline, train_data, train_y_data, cv=kfold)
        print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


    """
    Train model on the train_data and train_y_data
    """
    if args.fit == "true":
        fit_model()
    import glob
    """
    Test Model on the sampleimages/TestImages/*.jpg
    """
    ac = ActionClassifier()
    test_file = glob.glob("sample_images/TestImages/*.jpg")
    for test in test_file:
        test1 = cv2.imread(test)
        result,result_image = ac.classify(test1)
        file_name = test.split("/")[2]
        cv2.imwrite("result_"+file_name,result_image)

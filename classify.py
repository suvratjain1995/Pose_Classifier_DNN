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
def create_model():
  model = Sequential()
  model.add(Dense(120, input_dim=(136), kernel_initializer='normal', activation='relu'))
  model.add(Dense(60,input_dim = (120),kernel_initializer='normal',activation='relu'))
  model.add(Dense(1,input_dim = (60), kernel_initializer='normal', activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
seed = 7
np.random.seed(seed)

def load_train_data(train_data_path,train_data_path_y):
    with open(train_data_path,"rb") as f:
        train_ = pickle.load(f)
    
    with open(train_data_path_y,"rb") as f:
        train_y = pickle.load(f)

    return train_,train_y



def fit_model():
    model = create_model()
    train_data ,train_y_data = load_train_data("train_data.pickle","train_data_y.pickle")
    model.fit(train_data,train_y_data,batch_size=1,epochs= 100,shuffle= True)

    with open("model.pickle","wb") as f:
        pickle.dump(model,f)
    return model



class ActionClassifier:
    model = None
    @staticmethod
    def load_model(model_path):
        with open(model_path,"rb") as f:
            ActionClassifier.model = pickle.load(f)
        load_m(None)
    
    def __init__(self,model_path = "model.pickle"):
        ActionClassifier.load_model(model_path)


    def getImageVector(image):
        train_ = []
        params,model_params = config_reader()
        canvas,subset,candidate = process(image,params,model_params,series = True,model_call = True)
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
                    train_.append(result)
        train_ = np.asarray(train_)
        train_ = np.squeeze(train_,axis= 1)
        return train_
                    


    def classify(image):
        train_ = getImageVector(image)
        fight_flag = 0
        notfight_flag = 0
        for i in range(len(train_)):
            temp_ = train_[i]
            temp_ = np.expand_dims(temp_,axis = 0)
            result = ActionClassifier.model.predict(temp_)
            if(result[0] >= 1):
                fight_flag+=1
            else:
                notfight_flag+=1

        if fight_flag > notfight_flag:
            return "fight"
        else:
            return "notfight" 


# estimators = []
# # estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=100, batch_size=4, verbose=0)))
# pipeline = Pipeline(estimators)
# train_data ,train_y_data = load_train_data("train_data.pickle","train_data_y.pickle")
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
# results = cross_val_score(pipeline, train_data, train_y_data, cv=kfold)
# print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

fit_model()
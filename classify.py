from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
def create_smaller():
  model = Sequential()
  model.add(Dense(120, input_dim=(136), kernel_initializer='normal', activation='relu'))
  model.add(Dense(60,input_dim = (120),kernel_initializer='normal',activation='relu'))
  model.add(Dense(1,input_dim = (60), kernel_initializer='normal', activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def load_train_data(train_data_path):
    with open(train_data_path,"rb") as f:
        train_ = pickle.load(f)
    train_len = train.shape[0]
    
estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=4, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, train_data, train_y_data, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
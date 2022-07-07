#Import required libraries
from random import Random
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

from sklearn.ensemble import RandomForestRegressor


model_params_fname = "model_params.save"
model_fname = "model.save"
history_fname = "history.json"
MODEL_NAME = "Random_forest"

class Random_forest(): 
    
    def __init__(self, n_estimators = 100, max_features = 1, max_samples = 0.5, **kwargs) -> None:
        super(Random_forest, self).__init__(**kwargs)
        self.n_estimators = int(n_estimators)
        self.max_features = int(max_features)
        self.max_samples= np.float(max_samples)
        
        self.model = self.build_model()
        
        
        
    def build_model(self): 
        model = RandomForestRegressor(n_estimators= self.n_estimators, max_features= self.max_features, max_samples= self.max_samples, random_state=42, criterion='squared_error',bootstrap= True, 
        oob_score= True, n_jobs=-1, verbose=0)
        return model
    
    
    def fit(self, train_X, train_y):        
                 
    
        self.model.fit(
                X = train_X,
                y = train_y
            )
    
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        model_params = {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features
            
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))

        joblib.dump(self.model, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path): 
        model_params = joblib.load(os.path.join(model_path, model_params_fname))

        rf = joblib.load(os.path.join(model_path, model_fname))
        return rf


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = Random_forest.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def save_training_history(history, f_path): 
    hist_df = pd.DataFrame(history.history) 
    hist_json_file = os.path.join(f_path, history_fname)
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise
def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path,encoding='latin')
        logger.debug('data loaded from %s with shape %d',file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(x_train:np.ndarray,y_train:np.ndarray,params:dict)->RandomForestClassifier:
    try:
        if x_train.shape[0]!=y_train.shape[0]:
            raise ValueError('number of samples of x and y train must be same')
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        logger.debug('model training started with %d samples',x_train.shape[0])
        clf.fit(x_train,y_train)
        logger.debug('model training completed')
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise
def main():
    try:
        params=load_params('params.yaml')['model_training']
        train_data=load_data('./data/processed/train_tfidf.csv')
        x_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values
        clf=train_model(x_train=x_train,y_train=y_train,params=params)
        model_path = os.path.join("models", "model.pkl")

# Ensure the models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the model
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        print(f"Model saved at: {model_path}")
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")
        
if __name__=='__main__':
    main()
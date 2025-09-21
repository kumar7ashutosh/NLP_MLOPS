import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path:str):
    try:
        with open(file_path,'rb') as file:
            model=pickle.load(file)
            logger.debug('model loaded from %s',file_path)
            return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
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

def evaluate_model(clf,x_test:np.ndarray,y_test:np.ndarray)->dict:
    try:
        y_pred=clf.predict(x_test)
        y_predict_proba=clf.predict_proba(x_test)[:,1]
        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=precision_score(y_test,y_pred)
        auc_score=roc_auc_score(y_test,y_predict_proba)
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc_score
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise
def main():
    try:
        clf=load_model('./models/model.pkl')
        test_data=load_data('./data/processed/test_tfidf.csv')
        x_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values
        metrics=evaluate_model(clf,x_test=x_test,y_test=y_test)
        
        eval_path=os.path.join('reports','metrics.json')
        os.makedirs(os.path.dirname(eval_path),exist_ok=True)
        with open(eval_path,'w') as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()
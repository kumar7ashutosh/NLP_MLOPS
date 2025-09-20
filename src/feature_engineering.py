import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug('data loaded and filled nan values from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
    
def apply_tfidf(train_data,test_data,max_features:int)->tuple:
    try:
        vec=TfidfVectorizer(max_features=max_features)
        x_train=train_data['text'].values
        y_train=train_data['target'].values
        x_test=test_data['text'].values
        y_test=test_data['target'].values
        x_train_bow=vec.fit_transform(x_train)
        x_test_bow=vec.transform(x_test)
        train_df=pd.DataFrame(x_train_bow.toarray())
        train_df['label']=y_train
        test_df=pd.DataFrame(x_test_bow.toarray())
        test_df['label']=y_test
        return train_df,test_df
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise
    
def main():
    try:
        train_data=load_data('./data/interim/train_processed.csv')
        test_data=load_data('./data/interim/test_processed.csv')
        train_df,test_df=apply_tfidf(train_data=train_data,test_data=test_data,max_features=50)
        data_path=os.path.join('./data','processed')
        os.makedirs(data_path,exist_ok=True)
        train_df.to_csv(os.path.join(data_path,'train_tfidf.csv'),index=False)
        test_df.to_csv(os.path.join(data_path,'test_tfidf.csv'),index=False)
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()
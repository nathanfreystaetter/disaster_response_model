import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input:
        messages.csv: File path of messages data
        categories.csv: File path of categories data
    Output:
        df: Merged dataset from messages and categories
    '''
    messages = pd.read_csv(messages_filepath) #load messages.csv
    categories = pd.read_csv(categories_filepath) #load categories.csv
    df = messages.merge(categories, on='id',how='left') #merge messages and categories dataframes into df
    return df

def clean_data(df):
    '''
    Input:
        df: Merged dataset from messages and categories
    Output:
        df: Cleaned dataset
    ''' 
    #create new df with each categories as separate column
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe and use it to rename categories df columns
    row = categories.iloc[[0]] 
    category_colnames = row.apply(lambda x: x.str.slice(0,-2)).values.tolist().pop() 
    categories.columns = category_colnames

    #replace value of categories columns to a binary 1/0 and convert to numeric value
    for column in categories:
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column], errors='coerce')
    
    #remove original categories column for df and replace with categories by concatenating them
    df = df.drop('categories', 1)
    df = pd.concat([df, categories], axis=1, join_axes=[df.index])
    
    #dedup dataset, keep first
    df=df.drop_duplicates(keep='first')
    
    return df

def save_data(df, database_filename):
    '''
    Save df into sqlite db
    Input:
        df: cleaned dataset
        database_filename: database name
    Output: 
        A SQLite database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('factMessages', engine, index=False, if_exists='replace')  

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
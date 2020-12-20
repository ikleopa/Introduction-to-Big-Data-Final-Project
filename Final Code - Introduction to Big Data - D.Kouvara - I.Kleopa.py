##############################################################################
#####                        IMPORT PACKAGES                             #####
##############################################################################
import csv
import json
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
# import seaborn as sns

import mysql.connector
from mysql.connector import errorcode
from sqlalchemy import create_engine
from sqlalchemy.sql import text

# to remove warnings on console:
pd.options.mode.chained_assignment = None
import plotly_express as px
#  pip install plotly
# pip install plotly_express

# import seaborn as sns
# from pywaffle import Waffle
# pip install pywaffle

import pandas as pd
import numpy as np
import json
# Library for the Count min sketch algorithm
from probables import CountMinSketch
# Library for the Hyperloglog algorithm
import hyperloglog
# Libraries for the evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats
# Running time
import time
##############################################################################
global_path = ""  #Change value to the path of your desired directory it is  important to end the path with a '/'

def partA(): 
    if global_path == "":
        path = 'C:/Users/Iris/Desktop/Deree/Master/Fall Semester 2020/INTRODUCTION TO BIG DATA/Project/'    
    else :
        path = global_path    
    store_Results = path
    # Encoding
    enc = 'ISO-8859-15'
     
    ###############################################################################
    #                           PART A                              #
    ###############################################################################
    print('''##################################################################
                         PART A                              
    ##################################################################''')
    # Read the csv files and import them to DataFrames
    # Users
    f_Users = path + 'BX-Users.csv' 
    d_Users = pd.read_csv(f_Users, encoding = enc, sep = ';', delimiter = ';')
    df_Users= pd.DataFrame(d_Users)
    df_Users.columns = df_Users.columns.str.strip().str.lower().str.replace('-', '_')
     
     
     # Books
    f_Books = path + 'BX_Books_correct.csv'
    d_Books =  pd.read_csv(f_Books, sep = ';', encoding = enc)
    df_Books = pd.DataFrame(d_Books)
    df_Books.columns = df_Books.columns.str.strip().str.lower().str.replace('-', '_')
     
     
     # Book Ratings
    f_BRat = path + 'BX-Book-Ratings.csv'
    d_BRat = pd.read_csv(f_BRat, encoding = enc, sep = ';', delimiter = ';')
    df_BRat = pd.DataFrame(d_BRat)
    df_BRat.columns = df_BRat.columns.str.strip().str.lower().str.replace('-', '_')
    
     
    # Merge the Users table with parts of the Books Ratings table
    df = pd.merge(df_Users, df_BRat,  on = 'user_id')
    # Merge the previously merged DataFrame with the Books table
    df = pd.merge(df, df_Books, on = 'isbn')
     
    ds_books = df['book_title'].value_counts().reset_index()
    ds_books.columns = ['book', 'count']
    # Sort by book popularity in reverse order        
    ds_books = ds_books.sort_values('count').tail(20)
     
    ds_authors = df['book_author'].value_counts().reset_index()
    ds_authors.columns = ['book authors', 'count']
     
     # Sort by author popularity in reverse order        
    ds_authors = ds_authors.sort_values('count').tail(20)
     
    # CLEANING
    print('''##################################################################
                         Q1  Understanding the data                           
    ##################################################################''')
    
    
    #                           VALID ISBN FUNCTION                               #
    
    def isValidISBN(isbn):
        # check for length
        if len(isbn) != 10:
            return False
          
        # Computing weighted sum of first 9 digits
        _sum = 0
        for i in range(9):
            try:
                if 0 <= int(isbn[i]) <= 9:
                    _sum += int(isbn[i]) * (10 - i)
                else:
                    return False
            except ValueError:
                return False
              
        # Checking last digit
        try:
            if(isbn[9] != 'X' and not (0 <= int(isbn[9]) <= 9) ):
                return False
        except ValueError:
            return False
          
        # If last digit is 'X', add 10 to sum, else add its value.
        _sum += 10 if isbn[9] == 'X' else int(isbn[9])
          
        # Return true if weighted sum of digits is divisible by 11
        return (_sum % 11 == 0)
    
    #                             DATA DESCRIPTION                                #  
    # Describe the dataset -- Descriptive Statistics
    print("Descriptive Statistics for Users ")
    print(df_Users.describe())
    print("--------------------------------------------------------------")
    print("Descriptive Statistics for Books ")
    print(df_Books.describe())
    print("--------------------------------------------------------------")
    print("Descriptive Statistics for Books Ratings ")
    print(df_BRat.describe())
    print("--------------------------------------------------------------")
    # Memory Requirements - Information about size
    print("Memory Requirements for Dataframe of Users ")
    df_Users.info(memory_usage='deep')
    print("--------------------------------------------------------------")
    print("Memory Requirements for Dataframe of Books")
    df_Books.info(memory_usage='deep')
    print("--------------------------------------------------------------")
    print("Memory Requirements for Dataframe of Book Ratings")
    df_BRat.info(memory_usage='deep')
    print("--------------------------------------------------------------")
    
    #                                  DATA CLEANING                              #
    # Investigating Users table
    # Checking for unique users
    uniq_users = df_Users.user_id.nunique()
    all_users = df_Users.user_id.count()
    print(f'No. of unique user_id entries: {uniq_users} | Total user_id entries: {all_users}')
    print("--------------------------------------------------------------")
    
    # Investigating Books table
    # Dropping image-url columns
    df_Books.drop(columns = ['image_url_s', 'image_url_m', 'image_url_l'], inplace = True)
    # Checking for unique book entries
    uniq_books = df_Books.isbn.nunique()
    all_books = df_Books.isbn.count()
    print(f'No. of unique books: {uniq_books} | All book entries: {all_books}')
    print("--------------------------------------------------------------")
     
    # Investigating Book Ratings table
    # Removing the entries that have a zero rating
    print(f'Size of book_ratings before removing zero ratings: {len(df_BRat)}')
    print("--------------------------------------------------------------")
    df_BRat = df_BRat[df_BRat.book_rating != 0]
    print(f'Size of book_ratings after removing zero ratings: {len(df_BRat)}')        
    print("--------------------------------------------------------------")
    
    # Merge the Users table with parts of the Books Ratings table
    df = pd.merge(df_Users, df_BRat,  on = 'user_id')
    # Merge the previously merged DataFrame with the Books table
    df = pd.merge(df, df_Books, on = 'isbn')
     
    ds_books = df['book_title'].value_counts().reset_index()
    ds_books.columns = ['book', 'count']
    # Sort by book popularity in reverse order        
    ds_books = ds_books.sort_values('count').tail(20)
     
    ds_authors = df['book_author'].value_counts().reset_index()
    ds_authors.columns = ['book authors', 'count']
     
    # Sort by author popularity in reverse order        
    ds_authors = ds_authors.sort_values('count').tail(20)
    
    
    # Display the most popular Books
    ds_books.plot(x='book', y='count', kind='bar')
    plt.show()
    
    # Display the most popular Book Authors
    ds_authors.plot(x='book authors', y='count', kind='bar')  # cumulative = True
    plt.show()        
    # Age ranges by reading activity
    print("Unique Ages before preprocessing")
    print(sorted(df_Users.age.unique()))
    # Choosing the age bins based on the unique age values
    df['age_bins'] = pd.cut(x=df['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 245])
    df_age = df.groupby(['age_bins']).count()
    ds_age = df_age['book_title']
     
    # Display age ranges by reading activity
    ds_age.plot(x='age_bins', y='book_title', kind='bar')
    plt.show()
     
     
    print("--------------------------------------------------------------")
     # Setting ages less than 5 and older than 100 to NaN to try keep them realistic
    df_Users.loc[(df_Users.age < 5) | (df_Users.age > 100), 'age'] = np.nan
    print("Unique Ages after preprocessing")
    print(sorted(df_Users.age.unique()))
    # First create a histogram of the Age field
    df_age = df_Users.age.hist(bins=10, figsize=(12, 5))
    df_age.set_xlabel('Age')
    df_age.set_ylabel('counts')
    df_age.set_xticks(range(0, 110, 10))
    plt.show()
     
    # Same plot without any binning
    u = df_Users.age.value_counts().sort_index()
    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 15})  # Set larger plot font size
    plt.bar(u.index, u.values)
    plt.xlabel('Age')
    plt.ylabel('counts')
    plt.show()
    
    # Checking the number of missing Age's in the dataset
    age_null = df_Users.age.isnull().sum()  # Sums up the 1's returned by the isnull() mask
    all_users = df_Users.user_id.count()  # Counts the number of cells in the series - excludes NaNs!
    print(f'There are {age_null} empty age values in our set of {all_users} users (or {(age_null / all_users) *100:.2f} % ).')
    # Replacing these values with zeros for the SQL processing part
    df_Users['age'] = df_Users['age'].fillna(0)
    
    
    #                                 Outlier detection                           #
    # Z-scores for outlier detection
       
    # Books Outliers
    # Calculate the standard deviation (std)
    std_Books = round(ds_books['count'].std())
    # Z-scores less than -3.0 or greater than +3.0 indicate outlier values
    user_Outl_Book = ds_books[(ds_books['count']-ds_books['count'].mean() > 3*std_Books) |
        (ds_books['count']-ds_books['count'].mean() < -3*std_Books)]
    print("--------------------------------------------------------------")
    print('The books outliers in terms of how many times they have been read are \n', user_Outl_Book)
    # # Dropping books outliers based on z scores
    # df_Books = df_Books.loc[~(df_Books.book_title.isin(user_Outl_Book.book))]
    print("--------------------------------------------------------------")
    # Checking for outliers based on logic
    # Create df of books with publication years in the future
    books_from_the_future = df_Books[df_Books.year_of_publication>2020]
    future_books_mini = books_from_the_future[['book_title', 'year_of_publication']]
    print(f'Future books:\n{future_books_mini}')
    print("--------------------------------------------------------------")
       
    print(f'Length of books dataset before removal: {len(df_Books)}')
    print("--------------------------------------------------------------")
    # Removing future books
    df_Books = df_Books.loc[~(df_Books.isbn.isin(books_from_the_future.isbn))]
    print(f'Length of books dataset after removal: {len(df_Books)}')
    print("--------------------------------------------------------------")
       
    # Authors Outliers
    std_Authors = round(ds_authors['count'].std())
    # Z-scores less than -3.0 or greater than +3.0 indicate outlier values
    user_Outl_Author = ds_authors[(ds_authors['count']-ds_authors['count'].mean() > 3*std_Authors) |
            (ds_authors['count']-ds_authors['count'].mean() < -3*std_Authors)]
    print('The authors outliers in terms of how many times they have been read are \n', user_Outl_Author)
    print("--------------------------------------------------------------")
       
    # Users Outliers in terms of how many books they have read
    data_users = [ds_books['count'], df['user_id']]
    headers = ['Books count', 'user_id']
    ds_users = pd.concat(data_users, axis=1, keys=headers)
    std_Users = round(ds_users['Books count'].std())
    
    # Z-scores less than -3.0 or greater than +3.0 indicate outlier values
    user_Outl_User = ds_users[(ds_users['Books count']-ds_users['Books count'].mean() > 3*std_Users) |
            (ds_users['Books count']-ds_users['Books count'].mean() < -3*std_Users)]
    print('The books outliers for users are \n', user_Outl_User)
    print("--------------------------------------------------------------")
    
    # Users Outliers
    filter_users = df_BRat['user_id'].value_counts()
    ds_user_ratings = filter_users.rename_axis('user_id').to_frame('count')
    # Calculate the standard deviation (std)
    std_Users = round(ds_user_ratings['count'].std())
    
    # Z-scores greater than +1.0 indicate outlier values
    user_Outl = ds_user_ratings[(ds_user_ratings['count']-ds_user_ratings['count'].mean() > std_Users) ]
    user_Outl.reset_index(inplace = True)
    print('The users outliers in terms of how many books they have read are \n', user_Outl)
    print(f'Length of book ratings dataset before removal: {len(df_BRat)}')
    print("--------------------------------------------------------------")
    
    # Removing users outliers
    df_BRat = df_BRat.loc[(df_BRat.user_id.isin(user_Outl.user_id))]
    print(f'Length of book ratings dataset after removal: {len(df_BRat)}')
    print("--------------------------------------------------------------")
    
    # Removing invalid ISBN in book ratings
    print(f'Length of book ratings dataset before removal: {len(df_BRat)}')
    print("--------------------------------------------------------------")
    valid = df_BRat['isbn'].apply(isValidISBN).to_frame()
    indexes = valid[valid['isbn'] == False].index
    df_BRat.drop(indexes, inplace=True)
    df_BRat.index = np.arange(0, len(df_BRat))
    #df_BRat = df_BRat.loc[~(df_BRat.user_id.isin(user_Outl.user_id))]
    print(f'Length of book ratings dataset after removal: {len(df_BRat)}')
    print("--------------------------------------------------------------")
    
    # Removing invalid ISBN in books
    print(f'Length of books dataset before removal: {len(df_Books)}')
    print("--------------------------------------------------------------")
    valid_2 = df_Books['isbn'].apply(isValidISBN).to_frame()
    indexes = valid_2[valid_2['isbn'] == False].index
    df_Books.drop(indexes, inplace=True)
    df_Books.index = np.arange(0, len(df_Books))
    #df_BRat2 = df_BRat.loc[~(df_BRat.user_id.isin(user_Outl.user_id))]
    print(f'Length of books dataset after removal: {len(df_Books)}')
    print("--------------------------------------------------------------")
        
    # Code for Figure for country distribution that was shown in the presentation
        # df_users_country = df_BRat_u['country'].value_counts(normalize=True).reset_index()
        # df_users_country.columns = ['country', 'share']
        # df_users_country.loc[df_users_country['share'] < 0.01] = 'Other', df_users_country[df_users_country['share'] < 0.01][
        #     'share'].sum()
        # df_users_country.drop_duplicates(inplace=True)       
        
        # waffle_data = df_users_country.set_index('country').T.to_dict('records')[0]
        
        # fig = plt.figure(
        #     FigureClass=Waffle,
        #     figsize=(18, 4),
        #     rows=5,
        #     columns=15,
        #     values=waffle_data,
        #     labels=["%s (%.2f%%)" % (k, v * 100) for k, v in waffle_data.items()],
        #     legend={
        #         'loc': 'upper left',
        #         'bbox_to_anchor': (1, 1)
        #     },
        #     colors=sns.color_palette("RdBu_r", len(waffle_data)).as_hex(),
        #     icons='user',
        #     font_size=30,
        #     title={
        #         'label': 'Share of book ratings by country',
        #         'loc': 'left',
        #         'fontdict': {
        #             'fontsize': 14
        #         }
        #     }
        # )
        
        # plt.show()
          
   
    print('''##################################################################
                         Q2  Recommender System                           
    ##################################################################''')
   
    #  #                                  SQL                              #
    # # Find Similarities: 
    #     # 1. Create books schema
    
    # #Create database schema books
    # db = mysql.connector.connect(host = 'localhost',
    #                             user = 'root',
    #                             password = 'root',
    #                             allow_local_infile = True)
    # mycursor = db.cursor()
    
    # mycursor.execute('DROP DATABASE IF EXISTS books')
    # mycursor.execute('CREATE DATABASE books')
    # mycursor.execute('USE books')
    
    
    # # Create tables
    # # Create users
    # mycursor.execute("""CREATE TABLE bx_users (user_id INT NOT NULL primary key,  
    #                                       location VARCHAR(250),
    #                                       age INT NOT NULL)""")
    
    # # Create books table
    # mycursor.execute("""CREATE TABLE bx_books (isbn VARCHAR(13) NOT NULL, 
    #                                     book_title VARCHAR(""" + str(df_Books.book_title.apply(lambda x : len(x)).max()*1.1) + """) DEFAULT NULL, 
    #                                     book_author VARCHAR(255), 
    #                                     year_of_publication INT UNSIGNED, 
    #                                     publisher VARCHAR(255),
    #                                     primary key (ISBN))""")  
    
    
    # # Create book ratings table
    # mycursor.execute("""CREATE TABLE bx_book_ratings (user_id  INT NOT NULL,
    #                                                     isbn  VARCHAR(13) NOT NULL,
    #                                                     book_rating INT NOT NULL, 
    #                                                     primary key (User_ID, ISBN))""") 
    
    # mycursor.close()
    # db.close()
    
    # engine = create_engine('mysql+pymysql://root:root@localhost/books?charset=utf8', encoding = 'utf8', echo = True)
    # dbConnection = engine.connect()
    
    # try:
    #     statement = text("""SET NAMES UTF8""")
    #     dbConnection.execute(statement)
    #     frame_Users = df_Users.to_sql('bx_users', dbConnection, if_exists = 'append', index = False);
    #     frame_Books = df_Books.to_sql('bx_books', dbConnection, if_exists = 'append', index = False);
    #     frame_BRat = df_BRat.to_sql('bx_book_ratings', dbConnection, if_exists = 'append', index = False);
        
    # except ValueError as vx: 
    #     print(vx) 
    # except Exception as ex:   
    #     print(ex) 
    # else: 
    #     print('Data loaded successfully.');    
    # finally: 
    #     dbConnection.close() 
        
    #                        Similarities                             #
       
    #Transform to csv the dataset of the book ratings from each user
    user_item_rating = df_BRat
    user_item_rating.to_csv(path + 'User_Item_Rating.csv')
    
    # For 10000 first: because of Memory Error and because it took us approximately 1 day to run the project
    user_item_rating = user_item_rating[:10000]
    df_user_book_pair_th = pd.pivot_table(user_item_rating, index = 'user_id', columns = 'isbn', values = 'book_rating', fill_value=0)
    df_user_book_pair_th.to_csv(path + 'df_user_book_pair51.csv')
    df_user_book_pair_th = path + 'df_user_book_pair51.csv'
    df_user_book_pair_th = pd.read_csv(df_user_book_pair_th, encoding = enc, delimiter = ',')
    
    # the following commented lines were because we could not run due to memory error
    # for the whole dataset the command for the pivot table and that is why we ran it
    # on a better computer with a better RAM and we just read it from a csv on our computers
    # when we ran it to produce our results !----------------------!
    
    users = df_user_book_pair_th.user_id  
    df_user_book_pair_th = df_user_book_pair_th.set_index('user_id')
    
    # Create an empty DataFrame
    users_similarity = pd.DataFrame(data = None, index = None, columns = None)
    
    i = 0 
    j = 0 
    arr_A = []      
    arr_B = [] 
    
    while i <= (len(users) - 1):
        j = 0
        while j < len(users):
            u_i = users[i]
            u_j = users[j]
            if u_i != u_j:
                arr_A = df_user_book_pair_th.loc[u_i].values
                arr_B = df_user_book_pair_th.loc[u_j].values
            
                cosine_sim = np.double(np.dot(arr_A, arr_B)/(la.norm(arr_A) * la.norm(arr_B))) 
                users_similarity = users_similarity.append( [[u_i, u_j, cosine_sim]], ignore_index = True)
            j = j + 1
        i = i + 1
    users_similarity.rename(columns = {0: 'UserID_i'}, inplace = True) 
    users_similarity.rename(columns = {1: 'UserID_j'}, inplace = True) 
    users_similarity.rename(columns = {2: 'cosine_sim'}, inplace = True) 
    
    # df_user_book_pair_th=pd.read_csv('C:/Users/Iris/Desktop/Deree/Master/Fall Semester 2020/INTRODUCTION TO BIG DATA/Project/df_user_book_pair4.csv')
    
    users_similarity.to_csv(path + 'users_similarity51_limit.csv')
    # users_similarity = path + 'users_similarity4.csv'
    # users_similarity = pd.read_csv(users_similarity, encoding = enc, delimiter = ',')
    
    
    #                   K- nearest neighbors                        #
    
    sims=pd.read_csv('C:/Users/Iris/Desktop/Deree/Master/Fall Semester 2020/INTRODUCTION TO BIG DATA/Project/users_similarity51_limit.csv')
    
    sims=sims.drop('Unnamed: 0',axis=1)
    users = sims['UserID_i'].unique()
    
    # json_file={} this dataframe will hold for each user all the K nearest neighbors in a list
    # so for user 0 the first row would be a list of [X,X,X,X,X] where each X is one near neighbor 
    # json_file=pd.DataFrame( data=None, index=None, columns = None)
    KNN = 5
      
    json_file1={}
    # Here is where we fill the neighbors for each user. We go through the similarities matrix
    # and we sort descending and append to the json_file the KNN
    for u in users:    
        json_file1[u] = sims[sims['UserID_i']==u].sort_values(by='cosine_sim',ascending=False)[0:KNN]['UserID_j'].values.tolist()
    df_knn = pd.DataFrame.from_dict(json_file1, orient = 'index')
    df_knn.reset_index(inplace = True)
    df_knn.rename(columns={'index':'mainUser'}, inplace=True)
    df_knn.rename(columns={0:'knn1', 1:'knn2', 2:'knn3', 3:'knn4', 4:'knn5'}, inplace=True)
    # Save the k-nearest neighbours (knn)  
    df_knn.to_json(store_Results + 'neighbors-k-books.json')
    

    # # Save the k-nearest neighbours (knn) per user to database
    # db = mysql.connector.connect(host = 'localhost',
    #                         user = 'root',
    #                         password = 'root',
    #                         allow_local_infile = True)
    # mycursor = db.cursor()
    # mycursor.execute('USE books')
         
    # # Create tables
    # # Create users
    # mycursor.execute("""CREATE TABLE user_neighbors (mainuser INT NOT NULL, 
    #                   knn1 INT NOT NULL, knn2 INT NOT NULL, 
    #                   knn3 INT NOT NULL, knn4 INT NOT NULL,
    #                   knn5 INT NOT NULL, primary key (mainuser))""")
     
    # mycursor.close()
    # db.close()
     
    # engine = create_engine("mysql+pymysql://root:root@localhost/books?charset=utf8", encoding = 'utf8', echo = True)
    # dbConnection = engine.connect()
     
    # try:
    #     statement = text("""SET NAMES UTF8""")
    #     dbConnection.execute(statement)
    #     frame_user_neigh  = df_knn.to_sql('user_neighbors', dbConnection, if_exists = 'append', index = False);
    
    # except ValueError as vx: 
    #     print(vx) 
    # except Exception as ex:   
    #     print(ex) 
    # else: 
    #     print('Data loaded successfully.');    
    # finally: 
    #     dbConnection.close() 
    
    
    #          Recommendation / Evaluation / Improving                   #
    
    def findSharedBooks(neighbors):
        recommendation = []
        shared = np.empty(0)
        if KNN>1:        
            for i in range(0,KNN):
                for j in range (i+1,KNN):
                    shared = np.append(shared, np.intersect1d(neighbors[:][i],neighbors[:][j]))
            return shared
        elif KNN==1:
            return neighbors[:][0]
    
    # findSharedBooks(rec)
        
    def findMostFreqBooks(liBooks):
        unique, counts = np.unique(liBooks, return_counts=True)
        dictBooks = dict(zip(unique, counts))
        sorted(dictBooks, key=dictBooks.get, reverse=True)[:2]
        return dictBooks
    
    recommendations = pd.DataFrame()
    # for loop in the json file where the neighbors for each user are stored to produce recommendations
        # for each user we go through the list of the KNN neighbors.
            # For each of the KNN neighbors we go to the pivot table to keep only the books that he rated
            # >5 and put them all in a list called rec.
        # After we fill the rec list we pass it to the findSharesBooks that will find 
        # the most commonly shared read books among the neighbors and produce the recommendations
        # and store them in the recommendations dataframe for the specific user.  
    json_file=pd.DataFrame( data=None, index=None, columns = None)
    for u in users:     
        json_file[u] = sims[sims['UserID_i']==u].sort_values(by='cosine_sim',ascending=False)[0:KNN]['UserID_j'].values.tolist()
    p=0
    for i in json_file:
        print(json_file[i])
        p+=1
        if p>5:
            break
    
    j=json_file.transpose()
    
    for index,row in j.iterrows():
        rec=[]
        for k in j.loc[index]:
            n=df_user_book_pair_th.loc[k]>5
            neighbor_rated=n.where(n==True).dropna().index 
            rec.append(neighbor_rated.values)
        recommendations = recommendations.append([[index, findSharedBooks(rec)]])
           
    
    recommendations.columns = ['users', 'recommended_books']
    recommendations.reset_index(inplace=True)
    recommendations=recommendations.drop(columns='index')
    
    # to select first each row:
        # recommendations['recommended_book'].iloc[0]
        # recommendations['recommended_book'].iloc[1]
    
    # df_BRat.set_index('user_id')['isbn'].to_dict()
    
    # returns a list of the most rated books in the dataframe
    top_ten=df_BRat['isbn'].value_counts()[:10]
    # the first book is: 0316666343 and appeared 145
    
    # ratings for one of the most common book
    first_common_book= df_BRat.loc[df_BRat['isbn']=='0316666343'] 
    # the mean of ratings 
    df_BRat["book_rating"].mean()
    
    
    #returns a list of the most rated books in the dataframe
    most_common_books = df_BRat['isbn'].value_counts()[:10]
    
    best_books = df_BRat.loc[df_BRat['isbn'].isin(most_common_books.index)].groupby('isbn')['book_rating'].mean().sort_values(ascending=False)
        
    
    numOfRecBooks = 5
    for i in range(len(recommendations)):
        if len(recommendations['recommended_books'].iloc[i])<numOfRecBooks:        
            for j in range(numOfRecBooks-len(recommendations['recommended_books'].iloc[i])):
                recommendations['recommended_books'][i]=np.append(recommendations['recommended_books'][i],best_books.index[j]) 
    
    
    TP = 0
    FP = 0
    
    
     # We use as test set the books the user has already read
    for user in recommendations['users'][:]: # here we loop through all the users
       for listOfRecBooks in recommendations[recommendations['users']==user]['recommended_books']: # here we loop through the recommended books for each user
           for book in listOfRecBooks:
               if  df_user_book_pair_th[book][user]>5:
                   TP += 1
               else:
                   FP += 1
                   
    precision   = TP/(TP+FP)
    print('---- Precision = ------',precision)       
            

##############################################################################
#####                               PART B                            #####
##############################################################################
def partB():
    
    print('''##################################################################
                         PART B                              
    ##################################################################''')
                         
    time_start = time.perf_counter()
    
    # Construct functions for mae, mape, rmse and kendall's tau
    def mae_function(df_1, df_2):
        return df_1.iloc[:].sub(df_2.iloc[:]).abs().mean()
    
    def mape_function(df_1, df_2):
        return (df_1.iloc[:].sub(df_2.iloc[:]) / df_1.iloc[:]).abs().mean()
    
    def rmse_function(df_1, df_2):
        mse = df_1.iloc[:].sub(df_2.iloc[:]).pow(2).mean()
        return np.sqrt(mse)
    
    def kendalltau(df_1, df_2):
        return stats.kendalltau(df_1.iloc[:],df_2.iloc[:])
    if global_path =="":     
        path = 'C:/Users/Iris/Desktop/deleted from project/partB/'
    else:
        path = global_path
    # path for storing the resulted csv
    store_Results = path+'twitter_world_cup_1m/'
    # path of the tweets.json files
    json_folder_path = path + 'twitter_world_cup_1m/'
    # Encoding
    enc = 'UTF-8'
    
    json_files = []         # create an empty list to load all the json files
    for i in range (0,46): 
        json_files.append(json_folder_path +'tweets.json.'+ str(i))
    json_data = json_files
    # json_data = json_data[0:1]
    
    # Set the Count min sketch parameters (width and depth)
    cms = CountMinSketch(confidence = 0.999, error_rate = 0.0003)
    # cms = CountMinSketch(width = 10000, depth = 19)
    
    # Set the Hyperloglog parameter (error)
    error_hll = 0.01
    users_hll = hyperloglog.HyperLogLog(error_hll)
    hashtags_hll = hyperloglog.HyperLogLog(error_hll)
     
    # Initialize limit  
    heavy_hitters = 100  
    
    # Empty dictionaries for counting the heavy hitters
    # for users
    users_frq = {}
    users_frq_cms = {}
    max_hitters_users = {}
    # for hashtags
    hashtags_frq = {}
    hashtags_frq_cms = {}
    max_hitters_hashtags = {}
    
    # Empty lists for counting unique elements
    unique = []
    unique_hll = []
    # unique_users = []
    # unique_hashtags = []
    
    # Create empty DataFrames for the simple and approximated algorithms for count unique users and hashtags
    df_unique = pd.DataFrame(data = {'unique_users': [], 'unique_hashtags':[]}, index = None, columns = ['unique_users','unique_hashtags'])
    df_unique.index.name = 'Window'
    
    df_unique_hll = pd.DataFrame(data = {'unique_users': [], 'unique_hashtags':[]}, index = None, columns = ['unique_users','unique_hashtags'])
    df_unique_hll.index.name = 'Window'
    
    window = 0  # window of the data batches (every 1000 datapoints) 
    
    for file in json_data:  # loop to access all json files
        tweets = []
        for line in open(file, 'r', encoding = 'utf-8'):
            tweets.append(json.loads(line))
        
        # Set the parameters (start, end) for the streaming
        start = 0
        end = 1000
        n = 0        # row of the tweets
        
        while n < len(tweets):
            window_N = 'Window_' + str(window)
            for i in range(start, end):
                # Access the user_id from tweets
                user_id = tweets[i]['user']['id']
                # user frequency counter
                if str(user_id) in users_frq:
                    users_frq[str(user_id)] += 1
                else:
                    users_frq[str(user_id)] = 1
                
                # count min-sketch algorithm for users (approximation)
                user_id_cms = str(user_id)
                cms.add(user_id_cms)
                users_frq_cms[user_id] = cms.check(user_id_cms)
                               
                # HyperLogLog algorithm for users
                users_hll.add(str(user_id))
                    
                hash_list = list(tweets[i]['entities']['hashtags'])
                for k in range(len(hash_list)):
                    # Access each hashtag from tweets
                    hashtag = hash_list[k]['text']
                    # hashtag frequency counter
                    if hashtag in hashtags_frq:
                        hashtags_frq[hashtag] += 1
                    else:
                        hashtags_frq[hashtag] = 1
                        
                    # count min-sketch algorithm for hashtags (approximation)
                    hash_cms = str(hashtag)
                    cms.add(hash_cms)
                    hashtags_frq_cms[hashtag] = cms.check(hash_cms)
                    
                    # HyperLogLog for hashtags
                    hashtags_hll.add(str(hashtag))
                
                if i == len(tweets) - 1:
                    break
            
            # sorted dictionary for users
            users_frq = dict(sorted(users_frq.items(),key = lambda item: item[1], reverse = True))
            # heavy hitters for users (the first 100)
            max_hitters_users = dict(list(users_frq.items())[0: heavy_hitters])
            print(max_hitters_users)
            
            # sorted dictionary for hashtags
            hashtags_frq = dict(sorted(hashtags_frq.items(),key = lambda item: item[1], reverse = True))
            # heavy hitters for hashtags (the first 100)
            max_hitters_hashtags = dict(list(hashtags_frq.items())[0: heavy_hitters])
            print(max_hitters_hashtags)
            
            # Q2.1
            # Fill the DataFrame of the unique users and hashtags every 1000 datapoints (simple algorithm)
            df_unique.loc[window_N] = [len(users_frq), len(hashtags_frq)] 
            # Q2.2
            # Fill the DataFrame of the unique users and hashtags every 1000 datapoints (hyperloglog algorithm)
            df_unique_hll.loc[window_N] = [len(users_hll), len(hashtags_hll)]
    
            start += 1000
            end += 1000
            n += 1000
            window += 1
    
    # Q2.1
    # The count of unique users and hashtags (simple algorithm)
    unique = [len(users_frq), len(hashtags_frq)]
    
    # Q2.2a
    # The count of unique users and hashtags (hyperloglog algorithm)
    unique_hll = [len(users_hll), len(hashtags_hll)]
    
    # Dataframes for the users and hashtags' frequency
    df_users_all = pd.Series(users_frq).to_frame('users frequency')
    df_users_all = df_users_all.rename_axis('Users').reset_index()
    df_hashtags_all = pd.Series(hashtags_frq).to_frame('hashtags frequency')
    df_hashtags_all = df_hashtags_all.rename_axis('Hashtags').reset_index()
    
    # Q1.1
    # Using items() + list slicing  
    # Get first N items in dictionary for the heavy hitters
    # for users 
    users_frq = dict(list(users_frq.items())[0: heavy_hitters])
    df_users = pd.Series(users_frq).to_frame('users frequency')
    df_users = df_users.rename_axis('Users').reset_index()
    
    # Memory requirements for the users' frequency
    print('Bytes for users frequency by the simple approach: ', df_users.__sizeof__())
    
    # Q1.2
    users_frq_cms = dict(sorted(users_frq_cms.items(),key = lambda item: item[1], reverse = True))
    users_frq_cms = dict(list(users_frq_cms.items())[0: heavy_hitters])
    df_users_cms = pd.Series(users_frq_cms).to_frame('users frequency')
    df_users_cms = df_users_cms.rename_axis('Users').reset_index()        
    
    # Memory requirements for the users' frequency
    print('Bytes for users frequency by Count-min sketch approach: ', df_users_cms.__sizeof__())
    
    # Q1.1        
    # for hashtags
    hashtags_frq = dict(list(hashtags_frq.items())[0: heavy_hitters])
    df_hashtags = pd.Series(hashtags_frq).to_frame('hashtags frequency')
    df_hashtags = df_hashtags.rename_axis('Hashtags').reset_index()
    
    # Memory requirements for the hashtags' frequency
    print('Bytes for hashtags frequency by the simple approach: ', df_hashtags.__sizeof__())
    
    # Q1.2 
    hashtags_frq_cms = dict(sorted(hashtags_frq_cms.items(),key = lambda item: item[1], reverse = True))
    hashtags_frq_cms = dict(list(hashtags_frq_cms.items())[0: heavy_hitters])
    df_hashtags_cms = pd.Series(hashtags_frq_cms).to_frame('hashtags frequency')
    df_hashtags_cms = df_hashtags_cms.rename_axis('Hashtags').reset_index()  
    
    # Memory requirements for the hashtags' frequency
    print('Bytes for hashtags frequency by Count-min sketch approach: ', df_hashtags_cms.__sizeof__())
    
    
    # Save DataFrames to csv
    df_users.to_csv(store_Results + 'heavy_hitters_users.csv')
    df_hashtags.to_csv(store_Results + 'heavy_hitters_hashtags.csv')
    
    df_users_cms.to_csv(store_Results + 'heavy_hitters_users_cms.csv')
    df_hashtags_cms.to_csv(store_Results + 'heavy_hitters_hashtags_cms.csv')
    
    # Compare method accuracy - Evaluation of Heavy hitters (Q1) 
    # for users
    # RMSE
    mse_users = mean_squared_error(df_users['users frequency'], df_users_cms['users frequency'])
    rmse_users = np.sqrt(mse_users)
    print('Compare method accuracy for users, RMSE: ', rmse_users)
    # MAE
    mae_users = mean_absolute_error(df_users['users frequency'], df_users_cms['users frequency'])
    print('Compare method accuracy for users, MAE: ', mae_users)
    # Alternatively: calling the functions: rmse_function() and mae_function()
    # RMSE
    print('Compare method accuracy for users, RMSE: ', rmse_function(df_users['users frequency'], df_users_cms['users frequency']))
    # MAE
    print('Compare method accuracy for users, MAE: ', mae_function(df_users['users frequency'], df_users_cms['users frequency']))       
    # MAPE
    print('Compare method accuracy for users, MAPE: ', mape_function(df_users['users frequency'], df_users_cms['users frequency']))    
    
    
    # for hashtags
    # RMSE
    mse_hashtags = mean_squared_error(df_hashtags['hashtags frequency'], df_hashtags_cms['hashtags frequency'])
    rmse_hashtags = np.sqrt(mse_hashtags)
    print('Compare method accuracy for hashtags, RMSE: ', rmse_hashtags)
    # MAE
    mae_hashtags = mean_absolute_error(df_hashtags['hashtags frequency'], df_hashtags_cms['hashtags frequency'])
    print('Compare method accuracy for hashtags, MAE: ', mae_hashtags)
    # Alternatively: calling the functions: rmse_function() and mae_function()
    # RMSE
    print('Compare method accuracy for hashtags, RMSE: ', rmse_function(df_hashtags['hashtags frequency'], df_hashtags_cms['hashtags frequency']))
    # MAE
    print('Compare method accuracy for hashtags, MAE: ', mae_function(df_hashtags['hashtags frequency'], df_hashtags_cms['hashtags frequency']))    
    # MAPE
    print('Compare method accuracy for hashtags, MAPE: ', mape_function(df_hashtags['hashtags frequency'], df_hashtags_cms['hashtags frequency']))    
    
    # Q2.1
    # Simple approach to count the number of unique users and hashtags
    # Unique users
    print('The count of unique users', unique[0])
    df_users_unique = df_users_all.copy()
    df_users_unique.drop(columns = ['Users'], inplace = True)
    # Memory requirements for the unique users
    print('Bytes for unique users by the simple approach', df_users_unique.__sizeof__())
    
    # Unique Hashtags
    print('The count of unique hashtags', unique[1])
    df_hashtags_unique = df_hashtags_all.copy()
    df_hashtags_unique.drop(columns = ['Hashtags'], inplace = True)
    # Memory requirements for the unique hashtags
    print('Bytes for unique hashtags by the simple approach: ', df_hashtags_unique.__sizeof__())
    
    # Memory requirements for HyperLogLog algorithm on unique users and hashtags
    print('Bytes for unique users by HyperLogLog algorithm: ', users_hll.__sizeof__())
    print('Bytes for unique hashtags by HyperLogLog algorithm: ', hashtags_hll.__sizeof__()) 
       
    
    # Compare method accuracy - Evaluation of Counting Unique (Q2)
    # Unique Users
    # RMSE
    mse_unique_users = mean_squared_error(df_unique['unique_users'], df_unique_hll['unique_users'])
    rmse_unique_users = np.sqrt(mse_unique_users)
    print('RMSE between Actual and Hyperloglog for unique users: ', rmse_unique_users)
    # MAE
    mae_unique_users = mean_absolute_error(df_unique['unique_users'], df_unique_hll['unique_users'])
    print('MAE between Actual and Hyperloglog for unique users: ', mae_unique_users)
    # Alternatively: calling the functions: rmse_function() and mae_function()
    # RMSE
    print('RMSE between Actual and Hyperloglog for unique users: ', rmse_function(df_unique['unique_users'], df_unique_hll['unique_users']))
    # MAE
    print('MAE between Actual and Hyperloglog for unique users: ', mae_function(df_unique['unique_users'], df_unique_hll['unique_users']))
    # MAPE
    print('MAPE between Actual and Hyperloglog for unique users: ', mape_function(df_unique['unique_users'], df_unique_hll['unique_users']))    
    
    # Unique Hashtags
    # RMSE
    mse_unique_hashtags = mean_squared_error(df_unique['unique_hashtags'], df_unique_hll['unique_hashtags'])
    rmse_unique_hashtags = np.sqrt(mse_unique_hashtags)
    print('RMSE between Actual and Hyperloglog for unique hashtags: ', rmse_unique_hashtags)
    # MAE
    mae_unique_hashtags = mean_absolute_error(df_unique['unique_hashtags'], df_unique_hll['unique_hashtags'])
    print('MAE between Actual and Hyperloglog for unique hashtags: ', mae_unique_hashtags)
    # Alternatively: calling the functions: rmse_function() and mae_function()
    # RMSE
    print('RMSE between Actual and Hyperloglog for unique hashtags: ', rmse_function(df_unique['unique_hashtags'], df_unique_hll['unique_hashtags']))
    # MAE  
    print('MAE between Actual and Hyperloglog for unique hashtags: ', mae_function(df_unique['unique_hashtags'], df_unique_hll['unique_hashtags']))
    # MAPE
    print('MAPE between Actual and Hyperloglog for unique hashtags: ', mape_function(df_unique['unique_hashtags'], df_unique_hll['unique_hashtags']))    
    
    # Kendall's tau
    tau_users = kendalltau(df_unique['unique_users'], df_unique_hll['unique_users'])
    print(tau_users)
    tau_hashtags = kendalltau(df_unique['unique_hashtags'], df_unique_hll['unique_hashtags'])
    print(tau_hashtags)
    
    # Dataframes for the count unique users and hashtags
    df_unique.to_csv(store_Results + 'unique_count.csv')
    df_unique_hll.to_csv(store_Results + 'unique_count_hll.csv')
    
    # Running time     
    time_elapsed = (time.perf_counter() - time_start)
    print('Total running time: ', time_elapsed)


def menu():
    selection = 0
    while selection != -1:
        print("Press 1 to run Part A: ")
        print("Press 2 to run Part B: ")             
        print(" Press -1 to exit program.")        

        selection = int(input("Enter Part of the project: "))
        exercises1(selection)


def exercises1 (part):
    if part == -1:
        return
    if part == 1:
        partA()
    if part == 2:
        partB()
    

menu()
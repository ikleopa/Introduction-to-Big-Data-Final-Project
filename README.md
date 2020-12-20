# Introduction-to-Big-Data-Final-Project

This is the final project for the course of Introduction to Big Data

 

Instructions for running the project's code:
The files mentioned below will be send to your email, as GitHub has a size limit in the uploaded files

 

For Part A:
- Book Datasets: BX-Users.csv, BX_Books_correct.csv, BX-Book-Ratings.csv
- df_user_book_pair4.csv, df_user_book_pair51.csv, users_similarity4.csv, users_similarity51_limit.csv, neighbors-k-books.json

 

For Part B:
- heavy_hitters_users.csv, heavy_hitters_users_cms.csv, heavy_hitters_hashtags.csv, heavy_hitters_hashtags_cms.csv, unique_count_hll.csv, unique_count.csv 

 

STEP 1.

 

INSTALL THE NECESSARY LIBRARIES

 

import csv

 

import json

 

import pandas as pd

 

import numpy as np

 

import numpy.linalg as la

 

import matplotlib.pyplot as plt

 

import mysql.connector

 

from mysql.connector import errorcode

 

from sqlalchemy import create_engine

 

from sqlalchemy.sql import text

 

pd.options.mode.chained_assignment = None

 

import plotly_express as px

 

from probables import CountMinSketch

 

import hyperloglog

 

from sklearn.metrics import mean_squared_error

 

from sklearn.metrics import mean_absolute_error

 

import scipy.stats as stats

 

import time

 

STEP 2.

 

a. Please change the path directory to your own in lines 41, 670: global_path = ""

 

b. To store the results make sure to have the 'twitter_world_cup_1m/' saved to your path (line 675: store_Results = path+'twitter_world_cup_1m/', line 677: json_folder_path = path + 'twitter_world_cup_1m/')

 

STEP 3.

 

For your convenience, we have created a menu to run Parts A and B:

 

Press 1 to run Part A:

 

Press 2 to run Part B:        

 

Press -1 to exit program

 

STEP 4.

 

a. In case you want to execute the SQL commands, please uncomment the corresponding lines: 373-430 (for the relational database entitled: ‘books’) and 510-543 (for the neighbors-k-books.json)

 

b. Note that user and password in lines 379-380, 512-513 for msql.connector and in lines 413, 528 for create_engine should be changed (we have used root by default)

 

STEP 5. 

 

a. For efficiency purposes, we have decided to use only a sample of the pivot table named: df_user_book_pair51.csv

 

In case you want to run the original version (full size), please replace df_user_book_pair51.csv with df_user_book_pair4.csv in line 442

 

b. A sample is also used for the users similarity named: users_similarity51_limit.csv

 

In case you want to run the original version (full size), please replace users_similarity51_limit.csv with users_similarity4.csv in line 487

 

Note that in the same line the path directory should also be changed

 

STEP 6.

 

The running time for Part B is around 16 min for a 12 GΒ RAM. 

 

In case you want to obtain the results faster, please change the Count-min sketch parameters accordingly (less precision, less running time) in line 688

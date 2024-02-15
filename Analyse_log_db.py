#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Adam Tejkl
#
# Created:     16.08.2021
# Copyright:   (c) Adam Tejkl 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os
from matplotlib.image import imread
import tensorflow as tf
from tensorflow import keras
import json
import sqlite3

def treat_format(num, places, placeholder):
    symbol = str(num)
    for i in range(places):
        if len(symbol) < places:
            symbol = placeholder + symbol
    return(symbol)

## specify model number
model_number = "001"
##

main_folder = "E:/DS_ML/model_" + model_number

logs_folder = os.path.join(main_folder, "mosaics")

classification_model = keras.models.load_model(main_folder)

log_list = ['20181115_02', '20181115_03', '20181123_02', '20181123_03', '20181124_01', '20181124_02', '20181127_01', '20181127_02', '20181127_03', '20181129_01', '20181129_02', '20181129_03', '20181203_01', '20181203_02', '20181203_03', '20181203_04',
            '20181206_01', '20181206_02', '20181206_03', '20181211_01', '20181211_02', '20181211_03', '20181211_04', '20181211_05', '20181211_06', '20181214_01', '20181214_02', '20181214_03', '20181219_01', '20181219_02', '20181219_03', '20190108_01', '20190108_02', '20190108_03', '20200708_01', '20200708_02',
            '20200716_01', '20200716_03', '20200721_01', '20200721_02', '20200721_03', '20200727_01', '20200727_02', '20200727_03', '20200729_01', '20200729_02', '20200729_03', '20200803_01', '20200803_03', '20200810_01', '20200810_03', '20200824_01', '20200824_02', '20200827_03', '20201102_01', '20201102_02',
            '20201102_03', '20201105_01', '20201105_02', '20201105_03', '20201112_01', '20201112_02', '20201112_03', '20201112_04', '20201112_05', '20201119_01', '20201119_02', '20201119_03', '20201119_04', '20201119_05', '20201126_01', '20201126_02', '20201126_03', '20201126_04', '20201203_06']

log_list = ['20181112_01', '20181112_02', '20181112_03', ]
done_01 = []

##shift_list = ['0', '1', '2', '3']
shift_list = [ '1', '2', '3']

for shift in shift_list:
    for log in log_list:
        i = 0

        db_name = 'List_ds_' + log + '_' + shift + '_log.db'
        db_path = os.path.join(logs_folder, db_name)

        table_name = 'segments'
        columns = ['NoRill', 'Rill', 'Sheet']

        ## get data from db
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        ## Select rows where state is equal to 0 and Done is equal to 0
##        where_clause = f'State = 0 AND Done = 0' + ''.join([f' AND {column} = 0.0' for column in  columns])
##        query = f"SELECT Name, Path FROM {table_name} WHERE {where_clause}"
##        cursor.execute(query)

        ## Select rows where state is equal to 0 and overwrite previous classification
        cursor.execute(f"SELECT Path, Name FROM {table_name} WHERE State = 0")

        rows = cursor.fetchall()
        total = len(rows)
        print(total)

        for item in rows:

            mosaic_path = item[0]
##            mosaic_path = os.path.join(main_folder, 'others_4', item[0])
            try:
                img = tf.keras.preprocessing.image.load_img(mosaic_path, color_mode='rgb')
            except:
                mosaic_path = mosaic_path.replace('Rimov', 'CELSA')
                img = tf.keras.preprocessing.image.load_img(mosaic_path, color_mode='rgb')

            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = classification_model.predict(img_array)

            ## Update segments probability value with the new value
            rounded_predictions = []
            print_predictions = []
            for prediction in predictions[0]:
                num = int(prediction * 100)
                rounded_predictions.append(num)
                symbol = treat_format(num, 3, ' ')
                print_predictions.append(symbol)

            percentage = round((i/total)*100,2)
            print_clause = f'Image {item[1]} ' + ''.join([f'{prediction} {column} ' for prediction, column in zip(print_predictions, columns)]) + f'{percentage} %'
            print(print_clause)

            set_clause = ','.join([f' {column} = ?' for column in columns])
            set_clause = set_clause + ' ,State = 1'
            sql = f"UPDATE segments SET {set_clause} WHERE Name = ?"
##            print(sql)

            rounded_predictions.append(item[1])
            cursor.execute(sql, rounded_predictions)

            conn.commit()

            i += 1

        cursor.close()
        conn.close()

        ## set results to calibration segments
        state_list = [1, 2, 3, 4, 6, 7, 8, 9, 10]
        state_dict = {1:-1, 2:-2, 3:-3, 4:-4, 6:-5, 7:-6, 8:-7, 9:-8, 10:-9}

        for item in state_list:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
##            print(item)
            state = state_dict[item]
##            print(state)
            set_clause = ','.join([f'{column} = {state}' for column in columns])
            where_clause = f' State = {item} ' + ''.join([f' AND {column} = 0' for column in columns])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
            cursor.execute(query)
##            print(query)

            conn.commit()
            cursor.close()
            conn.close()

        print("Analysis of ", log, " done")


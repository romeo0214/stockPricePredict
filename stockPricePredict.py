#AI Recurrent Neural Network, Long Short Term Memory
#closing stock price predict

#import lib
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, model_from_yaml
from tensorflow.keras.layers import Dense, LSTM
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastquant import get_pse_data
import os
from datetime import  datetime, timedelta

class stockPricePredict():
    def __init__(self, stockCode, StartDate,EndDate):
        # get the stock price
        df = get_pse_data(stockCode, StartDate, EndDate)
        #print(df.tail())
        
        '''#visualize
        plt.figure(figsize=(16,8))
        plt.title("Closing Price History")
        plt.plot(df['close'])
        plt.xlabel("Date", fontsize=18)
        plt.xlabel("Closing Price", fontsize=18)
        #plt.show()
        '''
        #dataframe with only closing
        data =df.filter(['close'])
        #convert to numpy array
        dataset = data.values

        #get the number of rows to train the model
        training_Data_len = math.ceil(len(dataset) * .8)

        #scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)


        #create the training data set
        #create the scaled training data set
        train_data = scaled_data[0:training_Data_len, :]

        #split the training data x: training y: target
        x_train=[]
        y_train=[]

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
            '''if i<=60:
                print(x_train)
                print(y_train)
                print()'''

        #convert x_train and y_train to numpy arraws
        x_train, y_train = np.array(x_train), np.array(y_train)

        #reshape the data
        x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


        #check if model exists skip training
        if os.path.exists("model.yaml"):
            # load YAML and create model
            yaml_file = open("model.yaml", 'r')
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            model = model_from_yaml(loaded_model_yaml)
            # load weights into new model
            model.load_weights("model.h5")

        else:

            #build the LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            #compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            #train the model
            model.fit(x_train, y_train, batch_size=1, epochs=1)

            # serialize model to YAML
            model_yaml = model.to_yaml()
            with open("model.yaml", "w") as yaml_file:
                yaml_file.write(model_yaml)
            # serialize weights to HDF5
            model.save_weights("model.h5")


        #create the testing data set
        #create a new array containing scaled values
        test_data = scaled_data[training_Data_len-60:, :]

        #Create the data sets x_test and y_test
        x_test = []
        y_test = dataset[training_Data_len:, :]

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        #Convert the data into numpy arraw
        x_test = np.array(x_test)
        
        #reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        #get the models predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        #print(predictions)
        #Get the root mean squared error
        rmse = np.sqrt(np.mean(predictions - y_test)**2)
        
        #plot the data
        train = data[:training_Data_len]
        valid = data[training_Data_len:]
        valid['predictions'] = predictions

        #visualize the model
        plt.figure(figsize=(16,8))
        plt.title('Model for {stockCode}'.format(stockCode=stockCode))
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price', fontsize=18)
        plt.plot(train['close'])
        plt.plot(valid[['close', 'predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()


        #show the valid and predicted prices
        #print(valid)



        #predict today's Closing price
        today = datetime.today()
        DD = timedelta(days=300)
        delta = today - DD
        d_today = datetime.today().strftime('%Y-%m-%d')
        d_delta = delta.strftime('%Y-%m-%d')

        #get the quote

        quote = get_pse_data(stockCode, d_delta, d_today)
        #create a new dataframe
        new_df = quote.filter(['close'])

        #get last 60 days data
        last_60_days = new_df[-60:].values
        #scale the data
        last_60_days_scaled = scaler.transform(last_60_days)

        #Create an empty list
        X_test = []
        #append the past 60 days
        X_test.append(last_60_days_scaled)
        #Convert X_test to numpy arrau
        X_test = np.array(X_test)
        #reshape
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        #get the predicted scaled price
        pred_price = model.predict(X_test)
        #undo the scaling
        pred_price = scaler.inverse_transform(pred_price)
        print('Prediction Closing Price for today:', pred_price)


if __name__ == "__main__":
    stockCode="JFC"
    StartDate="2019-01-01"
    EndDate="2020-06-10"
    stockPricePredict(stockCode, StartDate, EndDate)
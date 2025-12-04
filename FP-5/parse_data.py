import pandas as pd

df = pd.read_csv('BeijingPM20130101_20151231.csv')

df.describe()


df = df[['hour', 'PM_US Post', 'DEWP', 'HUMI', 'PRES', 'TEMP','precipitation']]

df.describe()

df = df.rename( columns={'PM_US Post':'PM_US_Post', 'hour':'Hour', 'DEWP':'Dew_Point_Temperature', 'HUMI':'Humidity', 'PRES':'Pressure', 'TEMP':'Temperature', 'precipitation':'Precipitation'} )

df.describe()
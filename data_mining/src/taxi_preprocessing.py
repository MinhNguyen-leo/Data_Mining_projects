import pandas as pd
import numpy as np

def load_data(file_path, nrows=100000):
    df = pd.read_csv(file_path, nrows=nrows)
    return df

#Chia du lieu thanh cac chunk va lay mau ngau nhien tu tung chunk de giam bo nho
def load_sampled_data(file_path, sample_frac=0.05):
    chunks = pd.read_csv(file_path, chunksize=100000)
    df_list = []

    for chunk in chunks:
        sampled_chunk = chunk.sample(frac=sample_frac, random_state=42)
        df_list.append(sampled_chunk)

    df = pd.concat(df_list)
    return df

def clean_invalid_data(df):
    # Remove tọa độ = 0
    df = df[(df['pickup_longitude'] != 0) & (df['pickup_latitude'] != 0)]
    df = df[(df['dropoff_longitude'] != 0) & (df['dropoff_latitude'] != 0)]

    # Remove trip_distance <= 0
    df = df[df['trip_distance'] > 0]

    # Remove fare <= 0
    df = df[df['fare_amount'] > 0]
    df = df[df['total_amount'] > 0]

    # Remove passenger_count = 0
    df = df[df['passenger_count'] > 0]

    return df

def process_datetime(df):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # tao trip duration tu dropoff - pickup
    df['trip_duration'] = (
        df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    ).dt.total_seconds()

    df = df[df['trip_duration'] > 0]

    return df

def remove_outliers_duration(df):
    # 1. Domain filter
    df = df[
        (df['trip_duration'] > 60) &        # > 1 phút
        (df['trip_duration'] < 21600) &      # < 6 giờ
        (df['trip_distance'] > 0.1) &
        (df['fare_amount'] > 2.5)
    ]

    # 2. Quantile (cắt đuôi trên)
    df = df[
        (df['trip_duration'] < df['trip_duration'].quantile(0.99)) &
        (df['trip_distance'] < df['trip_distance'].quantile(0.99))
    ]

    return df

# Chia khung giờ trong ngày thành 4 nhóm: morning, afternoon, evening, night    
def create_time_features(df):
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['pickup_month'] = df['tpep_pickup_datetime'].dt.month

    conditions = [
        (df['pickup_hour'] >= 5) & (df['pickup_hour'] < 12),
        (df['pickup_hour'] >= 12) & (df['pickup_hour'] < 17),
        (df['pickup_hour'] >= 17) & (df['pickup_hour'] < 22),
    ]

    choices = ['morning', 'afternoon', 'evening']

    df['time_of_day'] = np.select(conditions, choices, default='night')

    df['is_peak_hour'] = ((df['pickup_hour'].between(7,9)) | 
                         (df['pickup_hour'].between(16,19))).astype(int)

    return df

    
def filter_rate_code(df):
    df = df[df['RatecodeID'] == 1]
    return df

# Tạo biến tip_amount chỉ tính tiền tip khi thanh toán bằng thẻ tín dụng (payment_type = 1), ngược lại = 0
def process_tip(df):
    df['tip_amount'] = np.where(df['payment_type'] == 1, df['tip_amount'], 0)

    df['is_credit_card'] = (df['payment_type'] == 1).astype(int)

    df['total_income'] = df['fare_amount'] + df['tip_amount']

    return df

#tao feature speed de loai bo cac chuyen di co van toc khong hop ly (vd: van toc > 100 mph)
def add_speed_feature(df):
    df['speed'] = df['trip_distance'] / (df['trip_duration'] / 3600)
    
    df = df[(df['speed'] > 1) & (df['speed'] < 100)]
    return df


def save_processed(df, output_path):
    df.to_csv(output_path, index=False)
    
def preprocess_taxi_df(df):
    df = clean_invalid_data(df)
    df = process_datetime(df)
    df = add_speed_feature(df)
    df = create_time_features(df)
    df = filter_rate_code(df)
    df = process_tip(df)           
    df = remove_outliers_duration(df)

    return df
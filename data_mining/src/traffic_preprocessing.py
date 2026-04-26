import pandas as pd
import numpy as np

# ========================
# 1. LOAD DATA
# ========================
def load_data(file_path):
    return pd.read_csv(file_path)


# ========================
# 2. DATETIME PROCESSING
# ========================
def process_datetime(df):
    df['date_time'] = pd.to_datetime(df['date_time'])

    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month

    # weekend feature
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # peak hour
    df['is_peak_hour'] = (
        df['hour'].between(7, 9) | df['hour'].between(16, 19)
    ).astype(int)

    return df


# ========================
# 3. ENCODE HOLIDAY
# ========================
def encode_holiday(df):
    df['is_holiday'] = np.where(df['holiday'] == "None", 0, 1)
    return df


# ========================
# 4. ENCODE WEATHER
# ========================
def encode_weather(df):
    df = pd.get_dummies(df, columns=['weather_main'], drop_first=True)
    return df


# ========================
# 5. RESAMPLE TIME SERIES
# ========================
def resample_time(df):
    df = df.set_index('date_time').sort_index()

    # chỉ resample numeric
    df = df.resample('h').mean(numeric_only=True)

    # nội suy dữ liệu thiếu
    df = df.interpolate(method='linear')

    df = df.reset_index()

    return df


# ========================
# 6. TEMPERATURE CONVERSION
# ========================
def convert_temperature(df):
    df['temp_c'] = df['temp'] - 273.15
    return df


# ========================
# 7. REMOVE OUTLIERS
# ========================
def remove_outliers(df):
    rain_threshold = df['rain_1h'].quantile(0.99)
    snow_threshold = df['snow_1h'].quantile(0.99)

    df = df[
        (df['rain_1h'] <= rain_threshold) &
        (df['snow_1h'] <= snow_threshold)
    ]

    return df


# ========================
# 8. FINAL PIPELINE
# ========================
def preprocess_traffic(df):
    # Step 1: datetime
    df = process_datetime(df)

    # Step 2: encode categorical BEFORE resample
    df = encode_holiday(df)
    df = encode_weather(df)

    # Step 3: resample time series
    df = resample_time(df)

    # Step 4: convert units
    df = convert_temperature(df)

    # Step 5: remove outliers
    df = remove_outliers(df)

    return df


# ========================
# 9. SAVE FILE
# ========================
def save_processed(df, output_path):
    df.to_csv(output_path, index=False)
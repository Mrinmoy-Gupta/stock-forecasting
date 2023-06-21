
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import mlflow
import nest_asyncio
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
from datetime import datetime, timedelta

app = FastAPI()

class model_input(BaseModel):
    smoothing_trend : float
    smoothing_seasonal : float
    smoothing_level : float

def data_process(start, end, df):
    duplicate_df = pd.date_range(start,end, freq='T')
    df_temp = pd.DataFrame({ 't': duplicate_df, 'd': None }) 
    common = df_temp.merge(df, on=["t"])
    row_not_in_originaldf = df_temp[~df_temp.t.isin(common.t)]
    row_not_in_originaldf.reset_index(drop=True, inplace=True)
    row_not_in_originaldf['c'] = row_not_in_originaldf['d']
    new_row_not_in_originaldf=row_not_in_originaldf.drop('d',axis=1)
    final_df=pd.concat([df,new_row_not_in_originaldf],ignore_index=True)
    final_df = final_df.sort_values(by="t",ignore_index=True)
    final_df=final_df.drop_duplicates('t',keep='first')
    new_final_df=final_df.fillna(method='ffill')
    new_final_df.drop(new_final_df.index[-1], inplace=True)
    return new_final_df


@app.post('/predict')
def predict(Inputs : model_input):
   
    train_data = pd.read_csv('')
    print(train_data)
    train_data['t'] = pd.to_datetime(train_data['t'])
    start_date=train_data.iloc[0].tolist()
    end_date=train_data.iloc[-1].tolist()
    print(start_date)
    print(end_date)
    train_data = data_process(start_date[0], end_date[0], train_data)
    train_data.set_index('t', inplace=True)
    fitted_model = ExponentialSmoothing(train_data['c'],trend='MUL',seasonal='MUL', freq='T',
                                            seasonal_periods=1440).fit(
                                            smoothing_trend = Inputs.smoothing_trend, 
                                            smoothing_seasonal = Inputs.smoothing_seasonal,
                                            smoothing_level =  Inputs.smoothing_level,
                                            optimized=False)
    forecast = fitted_model.forecast(1).values[0]
    now = datetime.today() + pd.Timedelta(hours = 3.5)
    now = now.strftime("%Y-%m-%d %H:%M") + ":00"
    print(now)
    pred_df = pd.DataFrame({'t':now, 'predictions':forecast}, index=['t'])
    response = pred_df.to_json(orient='split')
    
    return response

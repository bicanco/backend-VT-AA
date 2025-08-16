from fastapi import FastAPI, UploadFile, Query, Form
from umap import UMAP
from typing import Annotated, Optional
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager
import os
import re


class Settings(BaseSettings):
    folder: Optional[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    try:
        os.remove('temp.csv')
    except:
        return


settings = Settings()

app = FastAPI(lifespan=lifespan)
origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
)

@app.post("/projection")
async def plot_projection(
    files: UploadFile,
    selectedClass: str,
    features: Annotated[list[str], Query()],
    key: str,
    normalise: bool = False
):
    df = pd.read_csv(files.file)
    ds = df['Class'] == selectedClass
    data = df.loc[ds][features].dropna()
    data = data.fillna('')
    if normalise:
        data = (data-data.min())/(data.max()-data.min())
    result = UMAP(n_components=2).fit_transform(data).astype('double')
    plot = []
    keys = df.loc[ds][key]
    condition = 'colour' in df.columns
    if condition:
        aux = df.loc[ds]['colour'].fillna('')
    else:
        aux = df.loc[ds].fillna('')
    for index in range(len(result)):
        col = aux.iloc[index]
        point = {
            'x': result[index][0],
            'y': result[index][1],
            'text': keys.iloc[index],
            'colour': col if condition and col else None
        }
        plot.append(point)
    return plot

def check_type(column: str, df: pd.DataFrame):
    return {'feature': column, 'isNumeric': is_numeric_dtype(df[column])}

@app.post('/open')
async def read_features(files: UploadFile):
    df = pd.read_csv(files.file)
    columns = df.columns.to_list()
    return {
        'features': list(map(lambda x: check_type(x, df), columns)),
        'classes': df.Class.unique().tolist(),
    }

@app.post('/comment')
async def add_comment(files: UploadFile, selectedClass: str,
    lines: Annotated[str, Form()]):

    df = pd.read_csv(files.file)
    if 'comment' not in df.columns:
        df['comment'] = ''

    for line, comment in json.loads(lines).items():
        df.loc[(df['filename'] == line) & (df['Class'] == selectedClass), 'comment'] = comment

    df.to_csv('temp.csv', index=False)
    return FileResponse('temp.csv')

@app.get('/wav')
async def get_wav(selectedClass: str, filename: str):
    if settings.folder == None or selectedClass == None or filename == None:
        return
    selectedClass = re.sub('[^A-Za-z0-9_]+', '', selectedClass)
    filename = re.sub('[^A-Za-z0-9_.]+', '', filename)
    filename = re.sub('\\.{2,}', '.', filename)
    filename = settings.folder+'/'+selectedClass+'_audios/'+filename
    if os.path.isfile(filename):
        return FileResponse(filename)
    else:
        return


def check_numeric(column: str, df: pd.DataFrame, classes: list[str]):
    if is_numeric_dtype(df[column]):
        col = {'feature': column}
        if len(classes) > 1:
            for item in classes:
                partial = df.loc[df['Class'] == item, column]
                col[item] = {'max': partial.max(),'min': partial.min()}
        else:
            col[classes[0]] = {'max': df[column].max(),'min': df[column].min()}
        return col
    else:
        return None

def check_string(column: str, df: pd.DataFrame, classes: list[str]):
    if is_string_dtype(df[column]):
        col = {'feature': column, 'values': df[column].unique().tolist()}
        if len(classes) > 1:
            for item in classes:
                partial = df.loc[df['Class'] == item, column].unique()
                col[item] = partial.tolist()
        else:
            values = list(filter(lambda x: x, df[column].unique().tolist()))
            col[classes[0]] = values
        return col
    else:
        return None

@app.post('/parallel')
async def plot_parallel(files: UploadFile):
    df = pd.read_csv(files.file)
    classes = df['Class'].unique()

    df = df.fillna('')
    df = df.drop(columns=['filename', 'colour'])
    columns = df.columns.to_list()
    return {
        'numericFeatures': list(filter(lambda x: x,
            map(lambda x: check_numeric(x, df, classes), columns))),
        'nonNumericFeatures': list(filter(lambda x: x,
            map(lambda x: check_string(x, df, classes), columns))),
        'data': df.to_dict('records')
    }

def setColors(df: pd.DataFrame, key: str, value: str, colour: str):
    df[df[key] == value] = colour

@app.post('/colours')
async def export_colours(files: UploadFile, colours: Annotated[str, Form()], key: Annotated[str, Form()]):
    df = pd.read_csv(files.file)
    colours = json.loads(colours)
    if isinstance(colours, list):
        for item in colours:
            (value, colour) = item
            df.loc[df[key] == value, 'colour'] = colour
    else:
        return

    df.to_csv('temp.csv', index=False)
    return FileResponse('temp.csv')

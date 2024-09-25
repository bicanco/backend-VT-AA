from fastapi import FastAPI, UploadFile, Query, Form
from umap import UMAP
from typing import Annotated
import pandas as pd
from pandas.api.types import is_numeric_dtype
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
)

@app.post("/plot")
async def read_root(
    files: UploadFile,
    species: str,
    features: Annotated[list[str], Query()],
    key: str
):
    df = pd.read_csv(files.file)
    ds = df['Class'] == species
    data = df.loc[ds][features]
    result = UMAP(n_components=2).fit_transform(data).astype('double')
    plot = []
    keys = df.loc[ds][key]
    for index in range(len(result)):
        plot.append({
            'x': result[index][0],
            'y': result[index][1],
            'text': keys.iloc[index]
        })
    return plot

def check_type(column: str, df: pd.DataFrame):
    return {'feature': column, 'type': is_numeric_dtype(df[column])}

@app.post('/open')
async def read_features(files: UploadFile):
    df = pd.read_csv(files.file)
    columns = df.columns.to_list()
    return {
        'features': list(map(lambda x: check_type(x, df), columns)),
        'species': df.Class.unique().tolist()
    }

@app.post('/comment')
async def add_comment(files: UploadFile, species: str,
    lines: Annotated[str, Form()]):

    df = pd.read_csv(files.file)
    if 'comment' not in df.columns:
        df['comment'] = ''

    for line, comment in json.loads(lines).items():
        df.loc[(df['filename'] == line) & (df['Class'] == species), 'comment'] = comment

    df.to_csv('temp.csv', index=False)
    return FileResponse('temp.csv')

from fastapi import FastAPI, UploadFile, Query
from umap.umap_ import UMAP
from typing import Annotated
import pandas as pd
from pandas.api.types import is_numeric_dtype
from fastapi.responses import FileResponse

app = FastAPI()

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
async def add_comment(files: UploadFile, comment: str, species: str,
    lines: Annotated[list[str], Query()]):
    df = pd.read_csv(files.file)
    df['comment'] = ''

    for line in lines:
        df.loc[(df['filename'] == line) & (df['Class'] == species), 'comment'] = comment

    df.to_csv('test.csv')
    return FileResponse('test.csv')

import pandas as pd
import jax
import jax.numpy as np
import os
import pickle as pickle
import numpy as onp
import bisect
import json
import itertools
from itertools import chain, combinations


from funcs_LNC import *

def readMeta(folder, file):
    meta = pd.read_csv(folder+file, sep="#", nrows=19, header=None)
    meta = [el.replace(" ","").split("=",2) for el in meta[1].tolist()]
    meta  = {el[0]:el[1] for el in meta}
    meta["latents"] = int(meta["latents"])
    meta["confounder"] = int(meta["confounder"])
    meta["proxy"] = int(meta["proxy"])
    meta["noisesd"] = float(meta["noisesd"])
    meta["independent"] = meta["independent"]=="TRUE"
    meta["noiseproxy"] = int(meta["noiseproxy"])
    meta["ncl"] = int(meta["ncl"])
    meta["size"] = int(meta["size"])
    meta["ix"] = [int(el) for el in meta["ix"].replace("c(","").replace(")","").replace(" ","").split(",")]
    meta["iy"] = [int(el) for el in meta["iy"].replace("c(","").replace(")","").replace(" ","").split(",")]
    meta["ic"] = int(meta["ic"])
    meta["causal_coeff"] = float(meta["causal_coeff"])
    meta["coefx"] = [float(el) for el in meta["coefx"].replace("c(","").replace(")","").replace(" ","").split(",")]
    meta["coefy"] = [float(el) for el in meta["coefy"].replace("c(","").replace(")","").replace(" ","").split(",")]
    return meta

def readFile(folder, file):
    meta = readMeta(folder, file)
    num_proxies = meta["proxy"]
    num_latents = meta["latents"]
    
    data = pd.read_csv(folder+file, skiprows=19)
    zcols = ["Z."+str(i) for i in range(1,num_latents+1)]
    ucols = ["U."+str(i) for i in range(1,num_proxies+1)]
    x= np.array(data["X"])[:,None]
    x_std = np.std(x)
    x = stdrze(x)
    y= np.array(data["Y"])[:,None]
    y_std = np.std(y)
    y = stdrze(y)
    Z= np.array(onp.apply_along_axis(stdrze, 0, data[zcols]))
    U= np.array(onp.apply_along_axis(stdrze,0, data[ucols]))
    idx_x = np.array(meta["ix"]) #np.array([7, 1, 2])
    idx_y = np.array(meta["iy"]) #np.array([0, 6, 5])
    idx_c = np.array([meta["ic"]]) ##np.array([9])
    idxs = (idx_x, idx_y, idx_c)
    beta_real = np.array(meta["causal_coeff"])#/y_std*x_std)
    stds = (x_std, y_std)
    return x, y, Z, U, meta, idxs, beta_real, stds

def getDataSetTab(repos, pars):
    files = os.listdir(repos)
    files = list(set(files).difference(set(["log.txt"])))
    parsList = [pars[k] for k in pars.keys()]
    combos = [list(it) for it in itertools.product(*parsList)]
    combos = [[combos[j][i] for j in range(len(combos))] for i in range(len(combos[0]))]
    combos = {k: v for k, v in zip(pars.keys(), combos)}
    
    aux = {"fileNames": files}
    aux = pd.DataFrame.from_dict(aux)
    datasetTab2 = {"fileNames": [f for f in files for i in range(len(combos["lambda"]))]}

    for par in pars.keys():
        datasetTab2[par] = [p for f in files for p in combos[par]]

    datasetTab2 = pd.DataFrame.from_dict(datasetTab2)
    datasetTab2 = datasetTab2.merge(aux, on="fileNames", sort=True)
    datasetTab2["job"] = onp.arange(1, datasetTab2.shape[0]+1)
    return datasetTab2

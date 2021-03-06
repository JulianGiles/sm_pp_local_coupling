#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:54:12 2018

@author: julian.giles
"""

# Este script calcula los indices de heterogeneidad para comparar los eventos 
# de precipitación por la tarde con las condiciones precedentes de humedad del suelo
# Version 2 : improved efficiency
# Version 3 : fixed proper control non-event definition (no morning prec) 
# Version 4 : Completely redone code. Calculation is fragmented in more stages for easier bug solving.
# Version 5 : New approach for calculating faster non event metrics
# Version 6 : Fixed bug, v5 was not taking into account the season selection and was calculating all year.
#             Now all year is calculated and saved, season selection is for delta calculations and posterior plotting
# Version 7 : Updated code to use xarray and cartopy
# Version 8 : Implemented box size election
# Version 9 : Implemented dynamic regime classification (welty et al 2018, 2020) using VIMFC
# Version 10: Implemented degrading of resolution for delta calculation (from original grid to NxN pixels grid) (esto influye a partir del calculo de los deltas)
# Version 11: Added combination of datasets (GLEAM+CMORPH+ERA5). Added delta_period for selection of sub-period before delta calculation. 
#             Added alternative Ys calculation. Added SM anomaly parameters.
# Version 12: Modified code for new timestep selection (new juli_functions option). 
#             Now timesteps represent values in the current hour (t - t+1) instead of previous hour (t-1 - t)
#             Added new SM anomaly calculation option (with respect to the climatology for that day of the year)
#             Added new SM anomaly calculation option (with respect to the seasonal expectation: average 21-day rolling mean for that day of the year, without considering event year)
# Version 13: Missing feature added: now events with morning P>pre_mor_max in adjacent tiles are also filtered out. 
#             Improvements to speed with wrapping of xr.apply_ufunc and new event detection method
# Version 14: Missing feature added: now non-events with morning P>pre_mor_max in adjacent tiles are also filtered out. 
# Version 15: minor bug fixed: SM anomalies calculated only from morning values (instead of daily averages). Forced SM from UNC to be fixed to the mean annual wave.
# Version 16: bug fixed: improved way to select timezone bands (some hours from the day after the event were being left out)
# Version 17: added condition to test whether the SM increases after the afternoon P event (useful for mixed SM and P datasets)



# Queda por retocar:
    # falta agregar un if que si se cargan los datos de una corrida anterior se pase directamente a los graficos (o al calculo de deltas)
    # si me voy a otra region que no sea Sudamerica, cuidado con la seleccion de bandas horarias (actualmente a las horas de manana/tarde se les resta el huso horario, que podria ser negativo en otra region)
    # CUIDADO CON LOS MULTIPLICADORES DE LAS UNIDADES DE PRECIPITACION (por ej ERA5 está en m/h no en mm/h)
    # Guardar con pickle no es forward compatible, cambiar (ys_event_types)
    
    ##########################
    #
    #   CUIDADO AL CARGAR DATOS YA CALCULADOS QUE LOS DELTAS NO ESTAN IDENTIFICADOS SEGUN ESTACION, FALTA MEJORAR ESTO
    #
    ##########################

# ------------- PAQUETES --------------

# Con esto que está primero se soluciona el eror del device grafico. Al final no grafica nada pero se puede salvar el plot
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs	
import cartopy.feature 	
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
from numpy import ma 
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib.colors as mcolors
import os
from matplotlib import gridspec
import scipy.stats
from scipy import signal
import gc
import juli_functions
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
#import comet as cm
from skimage.feature import peak_local_max
import itertools
import math
import timeit
from scipy import stats
import collections

import warnings
#ignore by message
warnings.filterwarnings("ignore", message="Default reduction dimension will be changed to the grouped dimension")
warnings.filterwarnings("ignore", message="More than 20 figures have been opened")
warnings.filterwarnings("ignore", message="Setting the 'color' property will override the edgecolor or facecolor properties.")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- PARÁMETROS ----------
#lista de variables que quiero ver (si quiero EF ponerlo al final de la lista, dsps de lhf y shf)
var_list = ['pre','lhf','shf','lwr','swr','t2m','slp','u900','v900','uv900conv','u10m', 'v10m', 'uv10mconv', 'qu2d', 'qv2d', 'vimfc2d',
            'EF', 'EF_v2', 'q900', 'zi', 'evapot', 'swa', 'w900', 'w500', 'q2m', 'cloudbot', 'lowcc', 'totcc', 'qu900', 'qv900', 'quv900conv',
            'precwtr'] 
var_list = ['pre', 'sm1', 'vimfc2d', 'evapot', 'orog', 'lsmask'] # la orografia tiene que ir ultima

chunksize = 1000  # tamaño de los chunks para dask
start_date = '1983-01-01'
end_date = '2012-12-31'

models = {
          'RCA4': "RCA4 "+start_date[0:4]+"-"+end_date[0:4],
          'RCA4CLIM': "RCA4CLIM "+start_date[0:4]+"-"+end_date[0:4],
#          'LMDZ': "LMDZ "+start_date[0:4]+"-"+end_date[0:4],
#          'TRMM-3B42': "TRMM-3B42 V7 1998-2012",
#          'CMORPH': "CMORPH V1.0 1998-2012",
#          'JRA-55': "JRA-55 "+start_date[0:4]+"-"+end_date[0:4],
          'ERA5': "ERA5 "+start_date[0:4]+"-"+end_date[0:4],
          'GLEAM': "GLEAM "+start_date[0:4]+"-"+end_date[0:4],
          'GLDASNOAH': "GLDASNOAH "+start_date[0:4]+"-"+end_date[0:4],
          'ESACCI': "ESA-CCI "+start_date[0:4]+"-"+end_date[0:4],
          }
latlims=(-57,13.5) # custom lat limits for loading the data (default is -50.3, 13.5)
load_raw = True # cargar todos los datos originales crudos?
reload = False #reload previously calculated climatologies?
trih = False # force datasets into 3h steps (poner en True para los mapas trihorarios)
seasons = ['']    # estaciones que quiero calcular (en este script se calcula en ppio para todo y despues se elige una muestra para los delta)
months = 'JFMAMJJASOND' # string de meses
seas_warm = [1,2,3,10,11,12] # En que meses quiero la estacion a considerar (incluidos)
seas_cold = [4,5,6,7,8,9]
homepath = '/home/julian.giles/datos/'
data_path = '/datosfreyja/d1/GDATA/'
temp_path = '/home/julian.giles/datos/temp/heterog_sm_pp/run5_v17/'
images_path = '/home/julian.giles/datos/CTL/Images/heterogeneity_sm_pre_taylor_guillod/run5_v17/'
font_size = 20 # font size for all elements
proj = ccrs.PlateCarree(central_longitude=0.0)  # Proyeccion cyl eq

mult_hd = juli_functions.unit_multipliers()

interpolated= 'no' # yes or no all datasets interpolated to the RCA grid
lonfix = {'yes': {'LMDZ':0, 'TRMM-3B42':0},
          'no': {'LMDZ':-360, 'TRMM-3B42':-360}}
latfix = {'yes':1, 'no':-1}

units_labels=juli_functions.units_labels()

regions = juli_functions.regions()

arrow_scale={'vimfc2d':4000,
             'quv900conv':1,
             'uv900conv':100,
             'uv850conv':100,
             'uv10mconv':100}

arrow_scale_anoms={'vimfc2d':1000,
             'quv900conv':0.3,
             'uv900conv':50,
             'uv850conv':50,
             'uv10mconv':25}

arrow_spacing = 6

# diccionario con los nombres de los archivos:
files = juli_functions.files_dict()

# para las etiquetas de los subplots
numbering = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)']

# PRINT SELECTED OPTIONS
print('Script started. Computing datasets '+str([mod for mod in models.keys()])+'\n variables: '+str(var_list)+'\n from '+start_date+' to '+end_date+'\n Reloading previous calculations? '+str(reload)+'\n Forcing 3h data conversion? '+str(trih))

# ----------- PASAR RCA4 A TRIHORARIO -------------
#aux_counter = 0
#for model in ['RCA4', 'RCA4CLIM']:
#    for var in ['pre','t2m', 'evapot']: 
#        if not ('3h' in files[model][var]):
#            aux = xr.open_dataset(homepath+files[model][var], chunks={'time':1000})        
#            var_temp = dict(aux.data_vars)
#            if var == 'pre' or var== 'evapot':
#                aux_2 = aux.sel(time=slice(start_date, end_date)).resample(time='3H', closed= 'right', label= 'right').sum()  #[[str(var_name) for var_name in set(var_temp.keys())][0]][1:,:,:,:]\
#                aux_2.to_netcdf(homepath+files[model][var][:-3]+'_3hsum.nc')
#            if var == 't2m':
#                aux_2 = aux.sel(time=slice(start_date, end_date)).resample(time='3H', closed= 'right', label= 'right').last(skipna=False)  #[[str(var_name) for var_name in set(var_temp.keys())][0]][1:,:,:,:]\
#                aux_2.to_netcdf(homepath+files[model][var][:-3]+'_3h.nc')
#            del(aux); del(aux_2)
#            aux_counter = aux_counter +1
#
#if aux_counter != 0:
#    #exit the program early
#    from sys import exit
#    exit('3h file computed, reset script with new file')

# ---------- TOMA DE DATOS -----------
lon = dict()
lat = dict()
lonproj = dict()
latproj = dict()
if load_raw:
    data_xr, data, lon, lat, lonproj, latproj = juli_functions.load_datasets(models, var_list, start_date, end_date, data_path, homepath,
                                                                             files, chunksize, seas_warm, seas_cold, latfix, lonfix,
                                                                             forward_timestep=True,
                                                                             interpolated='no', latlims=latlims, triRCA= trih)
    
# ------------- CAMBIAR FUENTE ------------------
matplotlib.rcParams.update({'font.size':font_size}) #Change font size for all elements


#%% ---------- PARÁMETROS ----------
pre_multipliers = 1#/1000  # para acomodar las unidades de precipitación. Ej: ERA5 viene en m/h, entonces hay q dividir los umbrales por mil para q sean las mismas unidades
if 'ERA5' in models: pre_multipliers = 1/1000
if 'GLDASNOAH' in models: pre_multipliers = 1/(3*60*60)
if 'CMORPH' in models: pre_multipliers = 1

ys_calculation_type = 'min' # 'min' for comparison against min P point, 'mean' for comparison against sorrounding mean 
delta_period = ('1983-01-01', '2012-12-31') # (start_date, end_date) Período sobre el cual calcular los deltas (puedo tener todo el proceso de eventos hecho para un periodo mas largo y dsps elegir sub periodo para los delta finales)
load_deltas = False # load previously calculated deltas?
seas = set([10,11,12,1,2,3]) # set([12,1,2])# set([4,5,6,7,8,9]) # set([1,2,3,10,11,12]) # En que meses quiero la estacion a considerar (incluidos)
seas_name = 'ONDJFM' # Para los títulos

# test for increase in SM after afternoon P event?
test_increase_sm = False

# rangos horarios, recordar que ahora los timesteps indican el inicio del intervalo (forward_timestep=True)
rango_sm = (6,11) # Rango de horas para SM (en la mañana)
rango_pre_mor = (6,11) # Rango de horas para pre (en la mañana) Tener en cuenta que es el acumulado
rango_pre_aft = (12,23) # Rango de horas para pre (en la tarde)
if 'RCA4' in models or 'RCA4CLIM' in models: rango_sm = (6,8); rango_pre_mor = (6,8); rango_pre_aft = (9,20)

pre_mor_max = 1*pre_multipliers # Máxima prec en la mañana en mm
pre_aft_min = 4*pre_multipliers # Mínima prec en la tarde en mm

delta_orog_lim = 180 # Máximo cambio en orografía admitido dentro de la caja 3x3 en metros (para 0.25 seria 180, para 0.5 seria 360)
box_size = (3,3) # Dimensiones (horizontal,vertical) de la caja de los eventos (en puntos de grilla) 
if 'RCA4' in models: delta_orog_lim = 360
 
box_size2 = (int((box_size[0]-1)/2), int((box_size[1]-1)/2)) # para uso en el calculo
bootslength = 1000 # Cantidad de valores del bootstrapping
min_events = 25 # minimo de eventos que tiene que haber para plotear el resultado (solo aplica a los graficos)

degrade = True # degradar la reticula para agrupar eventos?
degrade_n = 6 # numero de puntos de reticula a agrupar (degrade_n x degrade_n)
if 'RCA4' in models: degrade_n = 3

if degrade_n >0: degrade =True

n_rollmean = 31 # nro de dias de referencia para tomar la anomalia (si es centrada tiene que ser impar)
sm_roll_mean_center = True # si tomar la anomalia de SM respecto a la rolling mean centrada (True) o para atras (False), para quitar los bias estacionales

seas_expect_smanom = False # la anomalia de SM es respecto a la seasonal expectation (Petrova, tomado de Taylor): roll mean centrada de 21 dias y promediada en los años menos en el año del evento
if seas_expect_smanom: sm_roll_mean_center = True

dayofyear_smanom = False # si hacer la anomalia de SM respecto a la climatología para ese dia del año. Si False, entonces hacer la rolling mean
previous_day_sm = False # calcula usando la SM promedio del dia previo en lugar de por la mañana (usa datos diarios de SM)


#%% --------- CONSTRUYO EL DATASET MEZCLA DE GLEAM+CMORPH+ERA5 ----------
if 'GLEAM' in models.keys():
    data['GLEAM+CMORPH+ERA5'] = dict()
    for var in var_list:
        data['GLEAM+CMORPH+ERA5'][var] = dict()
        
    data['GLEAM+CMORPH+ERA5']['pre'][''] = data['CMORPH']['pre']['']
    data['GLEAM+CMORPH+ERA5']['sm1'][''] = data['GLEAM']['sm1'][''][:,:,1:-1].transpose('time', 'lat', 'lon')
    data['GLEAM+CMORPH+ERA5']['evapot'][''] = data['GLEAM']['evapot'][''][:,:,1:-1].transpose('time', 'lat', 'lon')
    
    for var in ['vimfc2d', 'orog', 'lsmask']:
        data['GLEAM+CMORPH+ERA5'][var][''] = data['ERA5'][var][''][:,1:-2,:-1].rename({'longitude':'lon', 'latitude':'lat'})
        
    lat['GLEAM+CMORPH+ERA5'] = data['GLEAM+CMORPH+ERA5'][var]['']['lat']
    lon['GLEAM+CMORPH+ERA5'] = data['GLEAM+CMORPH+ERA5'][var]['']['lon']
    lonproj['GLEAM+CMORPH+ERA5'], latproj['GLEAM+CMORPH+ERA5'] = np.meshgrid(lon['GLEAM+CMORPH+ERA5'], lat['GLEAM+CMORPH+ERA5'])
    
    data['GLEAM+CMORPH+ERA5']['pre'][''].coords['lon'] = data['GLEAM+CMORPH+ERA5']['pre']['']['lon']-0.125
    data['GLEAM+CMORPH+ERA5']['pre'][''].coords['lat'] = data['GLEAM+CMORPH+ERA5']['pre']['']['lat']-0.125
    
    data['GLEAM+CMORPH+ERA5']['sm1'][''].coords['lon'] = data['GLEAM+CMORPH+ERA5']['sm1']['']['lon']-0.125
    data['GLEAM+CMORPH+ERA5']['sm1'][''].coords['lat'] = data['GLEAM+CMORPH+ERA5']['sm1']['']['lat']-0.125
    
    data['GLEAM+CMORPH+ERA5']['evapot'][''].coords['lon'] = data['GLEAM+CMORPH+ERA5']['evapot']['']['lon']-0.125
    data['GLEAM+CMORPH+ERA5']['evapot'][''].coords['lat'] = data['GLEAM+CMORPH+ERA5']['evapot']['']['lat']-0.125
    
    models = {'GLEAM+CMORPH+ERA5': "GLEAM+CMORPH+ERA5 "+start_date[0:4]+"-"+end_date[0:4]}


if 'GLDASNOAH' in models.keys():
    if 'CMORPH' in models.keys():
        # creo diccionario
        data['GLDASNOAH+CMORPH+ERA5'] = dict()
        for var in var_list:
            data['GLDASNOAH+CMORPH+ERA5'][var] = dict()
    
        # Muevo las lon/lat para que coincidan
        data['CMORPH']['pre'][''].coords['lon'] = data['CMORPH']['pre']['']['lon']-0.125
        data['CMORPH']['pre'][''].coords['lat'] = data['CMORPH']['pre']['']['lat']-0.125
        
        data['GLDASNOAH']['sm1'][''].coords['lat'] = data['GLDASNOAH']['sm1']['']['lat']-0.125
    
        # asigno variables 
        data['GLDASNOAH+CMORPH+ERA5']['pre'][''] = data['CMORPH']['pre']['']
        data['GLDASNOAH+CMORPH+ERA5']['sm1'][''] = data['GLDASNOAH']['sm1'][''].where(data['CMORPH']['pre'][''][0]+1) # el where es porque la reticula de CMORPH es la mas chica
        #data['GLDASNOAH+CMORPH+ERA5']['evapot'][''] = data['GLDASNOAH']['evapot']['']
        
        for var in ['vimfc2d', 'orog', 'lsmask']:
            data['GLDASNOAH+CMORPH+ERA5'][var][''] = data['ERA5'][var][''].rename({'longitude':'lon', 'latitude':'lat'}).where(data['CMORPH']['pre'][''][0]+1)
            
        lat['GLDASNOAH+CMORPH+ERA5'] = data['GLDASNOAH+CMORPH+ERA5'][var]['']['lat']
        lon['GLDASNOAH+CMORPH+ERA5'] = data['GLDASNOAH+CMORPH+ERA5'][var]['']['lon']
        lonproj['GLDASNOAH+CMORPH+ERA5'], latproj['GLDASNOAH+CMORPH+ERA5'] = np.meshgrid(lon['GLDASNOAH+CMORPH+ERA5'], lat['GLDASNOAH+CMORPH+ERA5'])
            
        models = {'GLDASNOAH+CMORPH+ERA5': "GLDASNOAH+CMORPH+ERA5 "+start_date[0:4]+"-"+end_date[0:4]}
    
    else:
        # creo diccionario
        data['GLDASNOAH+ERA5'] = dict()
        for var in var_list:
            data['GLDASNOAH+ERA5'][var] = dict()
    
        # Muevo las lon/lat para que coincidan
        
        data['GLDASNOAH']['sm1'][''].coords['lat'] = data['GLDASNOAH']['sm1']['']['lat']-0.125
        data['GLDASNOAH']['pre'][''].coords['lat'] = data['GLDASNOAH']['pre']['']['lat']-0.125
    
        # asigno variables 
        data['GLDASNOAH+ERA5']['pre'][''] = data['GLDASNOAH']['pre']['']
        data['GLDASNOAH+ERA5']['sm1'][''] = data['GLDASNOAH']['sm1']['']
        #data['GLDASNOAH+CMORPH+ERA5']['evapot'][''] = data['GLDASNOAH']['evapot']['']
        
        for var in ['vimfc2d', 'orog', 'lsmask']:
            data['GLDASNOAH+ERA5'][var][''] = data['ERA5'][var][''].rename({'longitude':'lon', 'latitude':'lat'}).where(data['GLDASNOAH']['pre'][''][0]+1) # el where es porque la reticula de CMORPH es la mas chica
            
        lat['GLDASNOAH+ERA5'] = data['GLDASNOAH+ERA5'][var]['']['lat']
        lon['GLDASNOAH+ERA5'] = data['GLDASNOAH+ERA5'][var]['']['lon']
        lonproj['GLDASNOAH+ERA5'], latproj['GLDASNOAH+ERA5'] = np.meshgrid(lon['GLDASNOAH+ERA5'], lat['GLDASNOAH+ERA5'])
            
        models = {'GLDASNOAH+ERA5': "GLDASNOAH+ERA5 "+start_date[0:4]+"-"+end_date[0:4]}
        

#%% --------- FILTRO LOS PUNTOS CON OROGRAFIA EMPINADA --------
print('#########  Filtrando puntos de orografía empinada y enmascarando oceanos ##############')
mask = dict()

for model in models.keys():

    print('.... '+model+' ......')
    init_time = timeit.time.time()           
    
    lat_name = [coord for coord in set(data[model]['orog'][''].coords.keys()) if "lat" in coord][0]
    lon_name = [coord for coord in set(data[model]['orog'][''].coords.keys()) if "lon" in coord][0]

    orog_shifted = xr.concat([data[model]['orog'][''][0].shift({lat_name:ii, lon_name:jj})*mult_hd['orog'][model][0] for ii,jj in list(itertools.product(range(-box_size2[0],box_size2[0]+1), repeat=2))], dim='shifted').compute()
    orog_shifted_max = orog_shifted.max(dim='shifted')
    orog_shifted_min = orog_shifted.min(dim='shifted')
    
    mask[model] = ( ((orog_shifted_max - orog_shifted_min)<delta_orog_lim).values * (data[model]['lsmask'][''][0] >0) ).compute()
    
    print(str(round((timeit.time.time()-init_time)/60,2))+' min')

#orog_ocean_mask = np.rollaxis(np.dstack([mask]*sm_RCA_CTL_daily_rm.shape[0]), axis=2)

#%% ------- ###### Cargar datos ya calculados #########


data_mor = dict() #datos diarios de mañana
data_aft = dict() #datos diarios de tarde
data_day = dict() #datos diarios de vimfc2d
ys_e = dict()
yt_e = dict()
yh_e = dict()
ys_event_types = dict()
ys_c = dict()
yt_c = dict()
yh_c = dict()
delta_e_ys = dict()
delta_e_yt = dict()
delta_e_yh = dict()

delta_ys = dict()
delta_yt = dict()
delta_yh = dict()

delta_e_ys_dynreg = dict()
delta_e_yt_dynreg = dict()
delta_e_yh_dynreg = dict()

delta_ys_dynreg = dict()
delta_yt_dynreg = dict()
delta_yh_dynreg = dict()

if degrade:
    delta_e_ys_dg = dict()
    delta_e_yt_dg = dict()
    delta_e_yh_dg = dict()
    
    delta_ys_dg = dict()
    delta_yt_dg = dict()
    delta_yh_dg = dict()

    delta_e_ys_dg_dynreg = dict()
    delta_e_yt_dg_dynreg = dict()
    delta_e_yh_dg_dynreg = dict()
    
    delta_ys_dg_dynreg = dict()
    delta_yt_dg_dynreg = dict()
    delta_yh_dg_dynreg = dict()
    
for model in models.keys():
    if reload:
        print('cargando datos ya calculados')
        try:
            timename='time'
            if model=='JRA-55': timename='initial_time0_hours'


            data_mor[model] = dict() #datos diarios de mañana
            data_aft[model] = dict() #datos diarios de tarde
            data_day[model] = dict()
            data_mor[model]['pre'] = xr.open_dataarray(temp_path+'/pre_mor_'+model+'.nc')
            data_aft[model]['pre'] = xr.open_dataarray(temp_path+'/pre_aft_'+model+'.nc')
            data_mor[model]['sm1'] = xr.open_dataarray(temp_path+'/sm1_mor_'+model+'.nc')
            data_day[model]['vimfc2d'] = xr.open_dataarray(temp_path+'/vimfc2d_day_'+model+'.nc')
            
            ys_e[model] = xr.open_dataarray(temp_path+'/ys_e_'+model+'.nc', chunks={timename:-1})
            yt_e[model] = xr.open_dataarray(temp_path+'/yt_e_'+model+'.nc', chunks={timename:-1})
            yh_e[model] = xr.open_dataarray(temp_path+'/yh_e_'+model+'.nc', chunks={timename:-1})
            ys_event_types[model] = np.load(temp_path+'/ys_event_types_'+model+'.nc.npy', allow_pickle=True)
            
            ys_c[model] = xr.open_dataarray(temp_path+'/ys_c_'+model+'.nc', chunks={timename:-1})
            yt_c[model] = xr.open_dataarray(temp_path+'/yt_c_'+model+'.nc', chunks={timename:-1})
            yh_c[model] = xr.open_dataarray(temp_path+'/yh_c_'+model+'.nc', chunks={timename:-1})
            
            if load_deltas:
            
                if degrade:
                    delta_e_ys_dg[model] = xr.open_dataarray(temp_path+'/delta_e_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
                    delta_e_yt_dg[model] = xr.open_dataarray(temp_path+'/delta_e_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
                    delta_e_yh_dg[model] = xr.open_dataarray(temp_path+'/delta_e_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
                    delta_ys_dg[model] = xr.open_dataarray(temp_path+'/delta_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
                    delta_yt_dg[model] = xr.open_dataarray(temp_path+'/delta_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
                    delta_yh_dg[model] = xr.open_dataarray(temp_path+'/delta_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
                
                    delta_e_ys_dg_dynreg[model] = dict()
                    delta_e_yt_dg_dynreg[model] = dict()
                    delta_e_yh_dg_dynreg[model] = dict()
                    delta_ys_dg_dynreg[model] = dict()
                    delta_yt_dg_dynreg[model] = dict()
                    delta_yh_dg_dynreg[model] = dict()
                
                    for dr in ['low', 'mid', 'high']:
                        delta_e_ys_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
                        delta_e_yt_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
                        delta_e_yh_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
                        delta_ys_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
                        delta_yt_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
                        delta_yh_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
                
                
                else:
                    delta_e_ys[model] = xr.open_dataarray(temp_path+'/delta_e_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
                    delta_e_yt[model] = xr.open_dataarray(temp_path+'/delta_e_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
                    delta_e_yh[model] = xr.open_dataarray(temp_path+'/delta_e_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
                    delta_ys[model] = xr.open_dataarray(temp_path+'/delta_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
                    delta_yt[model] = xr.open_dataarray(temp_path+'/delta_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
                    delta_yh[model] = xr.open_dataarray(temp_path+'/delta_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        
        
                    delta_e_ys_dynreg[model] = dict()
                    delta_e_yt_dynreg[model] = dict()
                    delta_e_yh_dynreg[model] = dict()
                    delta_ys_dynreg[model] = dict()
                    delta_yt_dynreg[model] = dict()
                    delta_yh_dynreg[model] = dict()
        
                    for dr in ['low', 'mid', 'high']:
        
                        delta_e_ys_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
                        delta_e_yt_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
                        delta_e_yh_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
                        delta_ys_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
                        delta_yt_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
                        delta_yh_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
            
        except:
            print('algo salio mal cargando los datos ya calculados')


#%% ------- CALCULOS -----------

data_mor = dict() #datos diarios de mañana
data_aft = dict() #datos diarios de tarde
sm_daily = dict()
sm_daily_rm = dict()
sm_daily_doy = dict()
data_day = dict() #datos diarios para vimfc2d

# defino funciones para seleccionar mañana y tarde
def pre_morning(hour):
    return (hour >= rango_pre_mor[0]) & (hour <= rango_pre_mor[1])

def pre_afternoon(hour):
    return (hour >= rango_pre_aft[0]) & (hour <= rango_pre_aft[1])

def sm_morning(hour):
    return (hour >= rango_sm[0]) & (hour <= rango_sm[1])


# ESTO SOLO PARA CASOS DONDE LOS DATASET DE P Y SM SON DE DISTINTA FUENTE Y QUIERO CHEQUEAR QUE COINCIDE LA P CON UN AUMENTO DE SM
def sm_afternoon(hour):
    return (hour >= rango_pre_aft[1]-3) & (hour <= rango_pre_aft[1])

sm_daily_aft = dict()
sm_increase = dict()


print('########## CALCULANDO ##############')
for model in models.keys():
    print('..... '+model+' ....')
    
    data_mor[model] = dict()
    data_aft[model] = dict()
    data_day[model] = dict()

    
    timename='time'
    if model=='JRA-55': timename='initial_time0_hours'

    lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]
    
    # LIMITES DE LAS TIMEZONES #########################################
    timezones = [5,4,3,2] # A MEJORAR: SE PODRIA AGREGAR EL NEGATIVO DENTRO DE ESTA VARIABLE (son franjas -3, -4, etc)
    zonelimits = np.array([float(lon[model][0]), -67.5-0.24, -52.5-0.24, -37.5-0.24, float(lon[model][-1])]) 
            
    pre_mor = dict() #diccionario temporal para guardar calculos por franjas horarias
    pre_aft = dict()
    sm_mor = dict()
    vimfc2d_day = dict()


    sm_aft = dict()

    
    # creo datos diarios sumando/promediando las horas de interés para cada caso
    print('calculando por franjas horarias')
    init_time = timeit.time.time()           

    for nn,zone in enumerate(timezones):
                
        print(str(nn+1)+'/'+str(len(timezones)))

        # muevo el tiempo de vimfc2d a LST , hago las cuentas y los vuelvo a sus tiempos originales
        data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] - zone*3600000000000
        data[model]['sm1'][''].coords[timename] = data[model]['sm1'][''][timename] - zone*3600000000000
        
        
        pre_mor[zone] = data[model]['pre'][''].loc[{timename:slice(start_date, end_date)}].loc[{timename:pre_morning(data[model]['pre'][''].loc[{timename:slice(start_date, end_date)}][timename+'.hour']), lon_name:slice(zonelimits[nn],zonelimits[nn+1])}].resample({timename:'D'}).sum().compute()
        
        pre_aft[zone] = data[model]['pre'][''].loc[{timename:slice(start_date, end_date)}].loc[{timename:pre_afternoon(data[model]['pre'][''].loc[{timename:slice(start_date, end_date)}][timename+'.hour']), lon_name:slice(zonelimits[nn],zonelimits[nn+1])}].resample({timename:'D'}).sum().compute()
        
        if 'GLEAM' not in model:            
            sm_mor[zone] = data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}].loc[{timename:sm_morning(data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}][timename+'.hour']), lon_name:slice(zonelimits[nn],zonelimits[nn+1])}].resample({timename:'D'}).mean().compute()
            
            if test_increase_sm:  
                sm_aft[zone] = data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}].loc[{timename:sm_afternoon(data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}][timename+'.hour']), lon_name:slice(zonelimits[nn],zonelimits[nn+1])}].resample({timename:'D'}).mean().compute()

        data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] + zone*3600000000000
        data[model]['sm1'][''].coords[timename] = data[model]['sm1'][''][timename] + zone*3600000000000


        # muevo el tiempo de vimfc2d a LST , hago la cuenta y lo vuelvo a su tiempo original
        data[model]['vimfc2d'][''].coords[timename] = data[model]['vimfc2d'][''][timename] - zone*3600000000000

        vimfc2d_day[zone] = data[model]['vimfc2d'][''].loc[{timename: slice(start_date,end_date), lon_name:slice(zonelimits[nn],zonelimits[nn+1])}].resample({timename:'D'}).mean().compute()*mult_hd['vimfc2d'][model][0]
        
        data[model]['vimfc2d'][''].coords[timename] = data[model]['vimfc2d'][''][timename] + zone*3600000000000
        
    print(str(round((timeit.time.time()-init_time)/60,2))+' min')
        
    # calculo la rolling mean de SM 
    print('calculando rolling mean diario de SM')
    init_time = timeit.time.time()           

    if 'GLEAM' in model:
        sm_daily[model] = data[model]['sm1']['']
        if dayofyear_smanom:
            sm_daily_doy[model] = sm_daily[model].groupby(timename+'.dayofyear').mean(dim=timename, skipna=True).compute()         
        else:
            sm_daily_rm[model] = sm_daily[model].rolling({timename:n_rollmean}, center=sm_roll_mean_center, min_periods=15).mean().compute()
    else:
        # muevo el tiempo de sm1 a LST (aprox)
        # data[model]['sm1'][''].coords[timename] = data[model]['sm1'][''][timename] - 4*3600000000000
        # sm_daily[model] = data[model]['sm1'][''].resample({timename:'D'}).mean().compute()
        # data[model]['sm1'][''].coords[timename] = data[model]['sm1'][''][timename] + 4*3600000000000
                
        sm_daily[model] = xr.concat([sm_mor[i] for i in timezones], dim=lon_name)

        if test_increase_sm:  
            sm_daily_aft[model] = xr.concat([sm_aft[i] for i in timezones], dim=lon_name)
            if len(sm_daily_aft[model]) < len(sm_daily[model]): sm_daily_aft[model] = xr.concat([sm_daily_aft[model], sm_daily[model][-(len(sm_daily[model]) - len(sm_daily_aft[model])):]], dim='time')
            sm_increase[model] = sm_daily_aft[model] > sm_daily[model]


        if dayofyear_smanom:
            sm_daily_doy[model] = sm_daily[model].groupby(timename+'.dayofyear').mean(dim=timename, skipna=True) 
        elif seas_expect_smanom:
            aux = sm_daily[model].rolling({timename:n_rollmean}, center=sm_roll_mean_center, min_periods=15).mean()         
            sm_daily_rm[model] = aux.groupby(timename+'.dayofyear').mean(dim=timename, skipna=True) - (aux/((int(end_date[0:4])-int(start_date[0:4]))+1)).groupby(timename+'.dayofyear')
        else:
            sm_daily_rm[model] = sm_daily[model].rolling({timename:n_rollmean}, center=sm_roll_mean_center, min_periods=15).mean()

    print(str(round((timeit.time.time()-init_time)/60,2))+' min')
        
    #junto los resultados por timezones en un solo array
    print('concatenando y guardando arrays')
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    data_mor[model]['pre'] = xr.concat([pre_mor[i] for i in timezones], dim=lon_name)
    data_mor[model]['pre'].to_netcdf(temp_path+'/pre_mor_'+model+'.nc')
    data_mor[model]['pre'] = xr.open_dataarray(temp_path+'/pre_mor_'+model+'.nc')

    data_aft[model]['pre'] = xr.concat([pre_aft[i] for i in timezones], dim=lon_name)
    data_aft[model]['pre'].to_netcdf(temp_path+'/pre_aft_'+model+'.nc')
    data_aft[model]['pre'] = xr.open_dataarray(temp_path+'/pre_aft_'+model+'.nc')

    if 'GLEAM' in model or previous_day_sm:
        if dayofyear_smanom:
            data_mor[model]['sm1'] = (sm_daily[model].groupby(timename+'.dayofyear')-sm_daily_doy[model]).roll(time=1, roll_coords=False)
        else:
            data_mor[model]['sm1'] = (sm_daily[model] - sm_daily_rm[model]).roll(time=1, roll_coords=False)
        
        data_mor[model]['sm1'].to_netcdf(temp_path+'/sm1_mor_'+model+'.nc')
        data_mor[model]['sm1'] = xr.open_dataarray(temp_path+'/sm1_mor_'+model+'.nc')
                    
    else:
        if dayofyear_smanom:
            #data_mor[model]['sm1'] = xr.concat([sm_mor[i] for i in timezones], dim=lon_name).groupby(timename+'.dayofyear') - sm_daily_doy[model]
            data_mor[model]['sm1'] = (sm_daily[model].groupby(timename+'.dayofyear') - sm_daily_doy[model]).round(7) #round is added to removes imprecise residuals in CLIM
            
        else:
            # data_mor[model]['sm1'] = xr.concat([sm_mor[i] for i in timezones], dim=lon_name) - sm_daily_rm[model]
            data_mor[model]['sm1'] = (sm_daily[model] - sm_daily_rm[model]).round(7) #round is added to removes imprecise residuals in CLIM

        data_mor[model]['sm1'].to_netcdf(temp_path+'/sm1_mor_'+model+'.nc')
        data_mor[model]['sm1'] = xr.open_dataarray(temp_path+'/sm1_mor_'+model+'.nc')

    data_day[model]['vimfc2d'] = xr.concat([vimfc2d_day[i] for i in timezones], dim=lon_name)
    data_day[model]['vimfc2d'].to_netcdf(temp_path+'/vimfc2d_day_'+model+'.nc')
    data_day[model]['vimfc2d'] = xr.open_dataarray(temp_path+'/vimfc2d_day_'+model+'.nc')


    if test_increase_sm:  
        sm_increase[model].to_netcdf(temp_path+'/sm_increase_'+model+'.nc')
        sm_increase[model] = xr.open_dataarray(temp_path+'/sm_increase_'+model+'.nc')


#%% ---------- IDENTIFICO LOS EVENTOS Y CALCULO LAS METRICAS ------------------
ys_e = dict()
yt_e = dict()
yh_e = dict()
ys_event_types = dict()
pre_cond_event = dict()


print('Identifying event days')            
for model in models.keys():
    print('..... '+model+' ....')
    print('Processing conditions') 
    init_time = timeit.time.time()           
    
    timename='time'
    if model=='JRA-55': timename='initial_time0_hours'

    lat_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lat" in coord][0]
    lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]
    
    # condicion de que el máximo esté en el pixel central

    iteration = list(itertools.product(range(-box_size2[0],box_size2[0]+1), repeat=2))
    iteration.remove((0,0))
    pre_cond1 = math.prod([data_aft[model]['pre'].shift({lat_name:ii, lon_name:jj})<data_aft[model]['pre'] for ii,jj in iteration])
    

    # condicion de no precip por la mañana en ningun punto de la caja (<pre_mor_max)
    pre_cond11 = math.prod([data_mor[model]['pre'].shift({lat_name:ii, lon_name:jj})<pre_mor_max for ii,jj in list(itertools.product(range(-box_size2[0],box_size2[0]+1), repeat=2))])

    # condicion de no precip por la mañana y precip por la tarde mayor a cierto umbral
    pre_cond2 = (data_mor[model]['pre'] < pre_mor_max)*(data_aft[model]['pre'] > pre_aft_min)

    # junto las condiciones 
    pre_cond_event[model] = (pre_cond1 * pre_cond11 * pre_cond2)

    if test_increase_sm:  
        pre_cond_event[model] = pre_cond1 * pre_cond11 * pre_cond2 * sm_increase[model]

    print(str(round((timeit.time.time()-init_time)/60,2))+' min')


    # Calculo Ys_e: métrica de preferencia espacial e Yh_e: métrica de heterogeneidad
    # (está contemplado que si hay dos mínimos de prec en un dia evento, la resta Ys_e es entre el 
    # punto del maximo (central) y la media de los valores en las posiciones de los minimos)
    
    print('Computing Ys_e & Yh_e')
    init_time = timeit.time.time()
    
    if ys_calculation_type == 'min':
        def calculo_ys_yh(sm_array, pre_cond_event, pre_aft):
            ys_e = np.empty(sm_array.shape); ys_e.fill(np.nan)
            yh_e = np.empty(sm_array.shape); yh_e.fill(np.nan)
            ys_event_types = np.empty(sm_array.shape, dtype=object)
            
            for i,j in zip(np.where(mask[model]==True)[0], np.where(mask[model]==True)[1]):
                if pre_cond_event[i,j]:
                    
                    min_pos = np.where(pre_aft[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)] == pre_aft[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)].min())
                    
                    ys_e[i, j] = float(sm_array[i,j] - np.mean([sm_array[i-box_size2[0]+ii, j-box_size2[1]+jj] for ii,jj in list(zip(min_pos[0], min_pos[1]))]))
                    ys_event_types[i,j] = (set(zip(min_pos[0],min_pos[1])))
                    
                    yh_e[i,j] = float(np.std(sm_array[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)]))
            
            return ys_e, yh_e, ys_event_types 
    
    
    elif ys_calculation_type == 'mean':
        def calculo_ys_yh(sm_array, pre_cond_event, pre_aft):
            ys_e = np.empty(sm_array.shape); ys_e.fill(np.nan)
            yh_e = np.empty(sm_array.shape); yh_e.fill(np.nan)
            ys_event_types = np.empty(sm_array.shape, dtype=object)
            
            for i,j in zip(np.where(mask[model]==True)[0], np.where(mask[model]==True)[1]):
                if pre_cond_event[i,j]:
                                        
                    ys_e[i, j] = float(sm_array[i,j] - np.mean(sm_array[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)][np.where(sm_array[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)]!=sm_array[i, j])]))
                    ys_event_types[i,j] = np.nan
                    
                    yh_e[i,j] = float(np.std(sm_array[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)]))
            
            return ys_e, yh_e, ys_event_types 
          
    else:
        print('Calculation type for Ys not recognized, aborting...')
        from sys import exit
        exit()

    def wrap():
        ys_e[model], yh_e[model], aux_events = xr.apply_ufunc(calculo_ys_yh, data_mor[model]['sm1'], pre_cond_event[model], data_aft[model]['pre'],
                                                              input_core_dims=[[lat_name, lon_name], [lat_name, lon_name], [lat_name, lon_name]],
                                                              output_core_dims=[[lat_name, lon_name], [lat_name, lon_name], [lat_name, lon_name]],
                                                              vectorize=True, dask='parallelized', 
                                                              dask_gufunc_kwargs={'allow_rechunk':True})

        ds_out = ys_e[model].to_dataset(name='ys_e')
        ds_out['yh_e'] = yh_e[model]
        ds_out['aux_events'] = aux_events
        
        return ds_out

    ds_out = wrap().compute()
    ys_e[model] = ds_out['ys_e']
    yh_e[model] = ds_out['yh_e']
    aux_events = ds_out['aux_events']

    
    def reduce_set_events(point):
        try:
            return set.union(*np.asarray(point[point != np.array(None)]))
        except:
            return {}
    
    ys_event_types[model] = xr.apply_ufunc(reduce_set_events, aux_events, input_core_dims=[[timename]], output_core_dims=[[]], vectorize=True, dask='parallelized', 
                                           dask_gufunc_kwargs={'allow_rechunk':True})



    print(str(round((timeit.time.time()-init_time)/60,2))+' min')

    
    # Calculo Yt_e: métrica de preferencia temporal
    print('Computing Yt_e')
    init_time = timeit.time.time()

    yt_e_condition_mask = (pre_cond_event[model]==1)*mask[model]
    yt_e[model] = data_mor[model]['sm1'].where(yt_e_condition_mask)
    
    print(str(round((timeit.time.time()-init_time)/60,2))+' min')

    # Guardo los resultados:
    print('Saving arrays in temp')
    ys_e[model].to_netcdf(temp_path+'/ys_e_'+model+'.nc')
    yt_e[model].to_netcdf(temp_path+'/yt_e_'+model+'.nc')
    yh_e[model].to_netcdf(temp_path+'/yh_e_'+model+'.nc')
    np.save(temp_path+'/ys_event_types_'+model+'.nc', ys_event_types[model])
    
    print('reloading results')
    ys_e[model] = xr.open_dataarray(temp_path+'/ys_e_'+model+'.nc', chunks={timename:-1})
    yt_e[model] = xr.open_dataarray(temp_path+'/yt_e_'+model+'.nc', chunks={timename:-1})
    yh_e[model] = xr.open_dataarray(temp_path+'/yh_e_'+model+'.nc', chunks={timename:-1})
    ys_event_types[model] = np.load(temp_path+'/ys_event_types_'+model+'.nc.npy', allow_pickle=True) # el pickle no es forward compatible, cambiar
    

#%% ---------- IDENTIFICO LOS NO EVENTOS Y CALCULO LAS METRICAS ------------------
# Creo los arrays vacios para guardar los datos
event_type = [(0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0), (0,0)] # tipo de evento segun la posicion del minimo

ys_c = dict()
yt_c = dict()
yh_c = dict()


print('Identifying non event days')            
for model in models.keys():
    print('..... '+model+' ....')

    timename='time'
    if model=='JRA-55': timename='initial_time0_hours'

    lat_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lat" in coord][0]
    lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]

    # condición de no evento: no tiene que haber evento ni precipitacion por la mañana
    pre_cond3 = data_mor[model]['pre'] < pre_mor_max
    pre_cond_nonevent = (~(pre_cond_event[model]==True))*pre_cond3

    # Calculo Ys_e: métrica de preferencia espacial e Yh_e: métrica de heterogeneidad
    # (está contemplado que si hay dos mínimos de prec en un dia evento, la resta Ys_e es entre el 
    # punto del maximo (central) y la media de los valores en las posiciones de los minimos)
    print('Computing Ys_c, Yh_c & Yt_c')
    
    if ys_calculation_type == 'min':   
        def calculo_no_events(data_mor, pre_cond_nonevent):
        
            event_type = [(0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0), (0,0)] # tipo de evento segun la posicion del minimo
            
            ys_c = np.empty((len(event_type),
                             pre_cond_nonevent.shape[0],
                             pre_cond_nonevent.shape[1])); ys_c.fill(np.nan)
            yt_c = np.empty(pre_cond_nonevent.shape); yt_c.fill(np.nan)
            yh_c = np.empty(pre_cond_nonevent.shape); yh_c.fill(np.nan)
    
            for i,j in zip(np.where(mask[model]==True)[0], np.where(mask[model]==True)[1]):
                ev_number=0
               
                
                if pre_cond_nonevent[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)].all() and i>box_size2[0]-1 and j>box_size2[1]-1 and i<mask[model].shape[0]-(box_size2[0]-1) and j<mask[model].shape[1]-(box_size2[1]-1):
                    for ev_type in ys_event_types[model][i,j]:
                        ys_c[ev_number,i,j] = np.asarray(data_mor[i,j] - data_mor[i-box_size2[0]+ev_type[0], j-box_size2[1]+ev_type[1]])
                        
                        ev_number = ev_number+1        
    
                    yh_c[i,j] = np.asarray(np.std(data_mor[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)]) )
                    yt_c[i,j] = np.asarray(data_mor[i,j])
        
            return ys_c, yh_c, yt_c
        
    elif ys_calculation_type == 'mean':
        def calculo_no_events(data_mor, pre_cond_nonevent):
        
            
            ys_c = np.empty((1,
                             pre_cond_nonevent.shape[0],
                             pre_cond_nonevent.shape[1])); ys_c.fill(np.nan)
            yt_c = np.empty(pre_cond_nonevent.shape); yt_c.fill(np.nan)
            yh_c = np.empty(pre_cond_nonevent.shape); yh_c.fill(np.nan)
    
            for i,j in zip(np.where(mask[model]==True)[0], np.where(mask[model]==True)[1]):
               
                
                if pre_cond_nonevent[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)].all() and i>box_size2[0]-1 and j>box_size2[1]-1 and i<mask[model].shape[0]-(box_size2[0]-1) and j<mask[model].shape[1]-(box_size2[1]-1):
                        
                    ys_c[0,i,j] = np.asarray(data_mor[i,j] - np.mean(data_mor[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)][np.where(data_mor[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)]!=data_mor[i, j])]))
    
                    yh_c[i,j] = np.asarray(np.std(data_mor[(i-box_size2[0]):(i+box_size2[0]+1), (j-box_size2[1]):(j+box_size2[1]+1)]) )
                    yt_c[i,j] = np.asarray(data_mor[i,j])
        
            return ys_c, yh_c, yt_c
    
    else:
        print('Calculation type for Ys not recognized, aborting...')
        from sys import exit
        exit()

    def wrap():
        ys_c[model], yh_c[model], yt_c[model] = xr.apply_ufunc(calculo_no_events, data_mor[model]['sm1'], pre_cond_nonevent,
                                                              input_core_dims=[[lat_name, lon_name], [lat_name, lon_name]],
                                                              output_core_dims=[['evtypes',lat_name, lon_name], [lat_name, lon_name], [lat_name, lon_name]],
                                                              vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk':True})
        
        ds_out = ys_c[model].to_dataset(name='ys_c')
        ds_out['yh_c'] = yh_c[model]
        ds_out['yt_c'] = yt_c[model]
        
        return ds_out

    init_time = timeit.time.time()
    ds_out = wrap().compute()
    ys_c[model] = ds_out['ys_c']
    yt_c[model] = ds_out['yt_c']
    yh_c[model] = ds_out['yh_c']
    print(str(round((timeit.time.time()-init_time)/60,2))+' min')


    # Guardo los resultados:
    print('Saving arrays in temp')
    ys_c[model].to_netcdf(temp_path+'/ys_c_'+model+'.nc')
    yt_c[model].to_netcdf(temp_path+'/yt_c_'+model+'.nc')
    yh_c[model].to_netcdf(temp_path+'/yh_c_'+model+'.nc')
    
    print('reloading results')
    ys_c[model] = xr.open_dataarray(temp_path+'/ys_c_'+model+'.nc', chunks={timename:-1})
    yt_c[model] = xr.open_dataarray(temp_path+'/yt_c_'+model+'.nc', chunks={timename:-1})
    yh_c[model] = xr.open_dataarray(temp_path+'/yh_c_'+model+'.nc', chunks={timename:-1})




#%% -------------------- RECORTO LA ESTACIÓN DESEADA  --------

ys_e_cut = dict()
yt_e_cut = dict()
yh_e_cut = dict()
ys_c_cut = dict()
yt_c_cut = dict()
yh_c_cut = dict()
data_day_cut = dict()

for model in models.keys():

    timename='time'
    if model=='JRA-55': timename='initial_time0_hours'

    daily_time_months = np.asarray(ys_e[model][timename+'.month'])
    
    if seas_name == 'Yearly':
        ys_e_cut[model] = ys_e[model].loc[{timename:slice(delta_period[0], delta_period[1])}]
        yt_e_cut[model] = yt_e[model].loc[{timename:slice(delta_period[0], delta_period[1])}]
        yh_e_cut[model] = yh_e[model].loc[{timename:slice(delta_period[0], delta_period[1])}]
        ys_c_cut[model] = ys_c[model].loc[{timename:slice(delta_period[0], delta_period[1])}]
        yt_c_cut[model] = yt_c[model].loc[{timename:slice(delta_period[0], delta_period[1])}]
        yh_c_cut[model] = yh_c[model].loc[{timename:slice(delta_period[0], delta_period[1])}]
        
        data_day_cut[model] = dict()
        data_day_cut[model]['vimfc2d'] = data_day[model]['vimfc2d'].loc[{timename:slice(delta_period[0], delta_period[1])}]
        
    else:
        def season_sel(month):
            return np.asarray([m in seas for m in month])
        
        ys_e_cut[model] = ys_e[model][season_sel(daily_time_months)].loc[{timename:slice(delta_period[0], delta_period[1])}]
        yt_e_cut[model] = yt_e[model][season_sel(daily_time_months)].loc[{timename:slice(delta_period[0], delta_period[1])}]
        yh_e_cut[model] = yh_e[model][season_sel(daily_time_months)].loc[{timename:slice(delta_period[0], delta_period[1])}]
        ys_c_cut[model] = ys_c[model][season_sel(daily_time_months)].loc[{timename:slice(delta_period[0], delta_period[1])}]
        yt_c_cut[model] = yt_c[model][season_sel(daily_time_months)].loc[{timename:slice(delta_period[0], delta_period[1])}]
        yh_c_cut[model] = yh_c[model][season_sel(daily_time_months)].loc[{timename:slice(delta_period[0], delta_period[1])}]
        
        
        daily_time_months2 = np.asarray(data_day[model]['vimfc2d'][timename+'.month'])
        data_day_cut[model] = dict()
        data_day_cut[model]['vimfc2d'] = data_day[model]['vimfc2d'][season_sel(daily_time_months2)].loc[{timename:slice(delta_period[0], delta_period[1])}]
        

#%% -------------------- CALCULO LOS DELTAS  --------
print('Calculando delta_e y delta_c')

print('... sin separar los regimenes dinamicos ...')
delta_e_ys = dict()
delta_e_yt = dict()
delta_e_yh = dict()

delta_ys = dict()
delta_yt = dict()
delta_yh = dict()

if degrade:
    ys_e_cut_dg = dict()
    yt_e_cut_dg = dict()
    yh_e_cut_dg = dict()
    ys_c_cut_dg = dict()
    yt_c_cut_dg = dict()
    yh_c_cut_dg = dict()

    delta_e_ys_dg = dict()
    delta_e_yt_dg = dict()
    delta_e_yh_dg = dict()
    
    delta_ys_dg = dict()
    delta_yt_dg = dict()
    delta_yh_dg = dict()

for model in models.keys():
    
    print('..... '+model+' ....')

    timename='time'
    if model=='JRA-55': timename='initial_time0_hours'

    lat_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lat" in coord][0]
    lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]

    
    def bootstrap_resample(X, n=None):
        """ Bootstrap resample an array_like
        Parameters
        ----------
        X : array_like
          data to resample
        n : int, optional
          length of resampled array, equal to len(X) if n==None
        Results
        -------
        returns X_resamples
        """
        if n == None:
            n = len(X)
            
        resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
        X_resample = X[resample_i]
        return X_resample    


    ###############################
    def boots(ys_e, yt_e, yh_e, ys_c, yt_c, yh_c):
        
        # Chequeo que sea un valid_gridpoint (al menos un valor no nan)
        if (~np.isnan(ys_e)).any():
        
            # Hago un slice para quitar los nans
            ys_e_2 = ys_e[~np.isnan(ys_e)]
            yt_e_2 = yt_e[~np.isnan(yt_e)]
            yh_e_2 = yh_e[~np.isnan(yh_e)]
            
            #ys_c_2 = ys_c.stack(z=(timename,'evtypes'))[~np.isnan(ys_c.stack(z=(timename,'evtypes')))] #con el vectorizado del apply_ufunc deja de ser xarray
            ys_c_2 = ys_c.flatten()[~np.isnan(ys_c.flatten())]
            yt_c_2 = yt_c[~np.isnan(yt_c)]
            yh_c_2 = yh_c[~np.isnan(yh_c)]
            
            delta_e_ys = np.nanmean(ys_e_2) - np.nanmean(ys_c_2)
            delta_e_yt = np.nanmean(yt_e_2) - np.nanmean(yt_c_2)
            delta_e_yh = np.nanmean(yh_e_2) - np.nanmean(yh_c_2)
            
            delta_ys = np.empty((bootslength))
            delta_yt = np.empty((bootslength))
            delta_yh = np.empty((bootslength))
            
            for k in np.arange(0, bootslength, 1):
                bootstrap = bootstrap_resample(np.concatenate((ys_e_2,ys_c_2)), n=len(ys_e_2))
                delta_ys[k] = (bootstrap.mean() - np.mean(ys_c_2))
        
                bootstrap = bootstrap_resample(np.concatenate((yt_e_2,yt_c_2)), n=len(yt_e_2))
                delta_yt[k] = (bootstrap.mean() - np.mean(yt_c_2))
        
                bootstrap = bootstrap_resample(np.concatenate((yh_e_2,yh_c_2)), n=len(yh_e_2))
                delta_yh[k] = (bootstrap.mean() - np.mean(yh_c_2))
                
            return delta_e_ys, delta_e_yt, delta_e_yh, delta_ys, delta_yt, delta_yh
        
        else:
            return np.nan, np.nan, np.nan, np.full((bootslength), np.nan), np.full((bootslength), np.nan), np.full((bootslength), np.nan)


    if degrade:
        res_lat = [None if len(ys_e_cut[model][lat_name])%degrade_n==0 else len(ys_e_cut[model][lat_name])%degrade_n*-1][0]
        res_lon = [None if len(ys_e_cut[model][lon_name])%degrade_n==0 else len(ys_e_cut[model][lon_name])%degrade_n*-1][0]
        
        ys_e_cut_dg[model] = xr.concat([ys_e_cut[model][:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
        yt_e_cut_dg[model] = xr.concat([yt_e_cut[model][:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
        yh_e_cut_dg[model] = xr.concat([yh_e_cut[model][:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
        ys_c_cut_dg[model] = xr.concat([ys_c_cut[model][:,:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
        yt_c_cut_dg[model] = xr.concat([yt_c_cut[model][:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
        yh_c_cut_dg[model] = xr.concat([yh_c_cut[model][:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
        
        def wrap():
            delta_e_ys_dg[model], delta_e_yt_dg[model], delta_e_yh_dg[model], delta_ys_dg[model], delta_yt_dg[model], delta_yh_dg[model] = xr.apply_ufunc(boots, ys_e_cut_dg[model], yt_e_cut_dg[model], yh_e_cut_dg[model], ys_c_cut_dg[model], yt_c_cut_dg[model], yh_c_cut_dg[model],
                                                                                                                                    input_core_dims=[[timename], [timename], [timename], ['evtypes', timename], [timename], [timename]],
                                                                                                                                    output_core_dims=[[], [], [], ['boots'], ['boots'], ['boots']], vectorize=True, dask='parallelized', 
                                                                                                                                    dask_gufunc_kwargs={'allow_rechunk':True, 'output_sizes':{'boots':bootslength}})
            ds_out = delta_e_ys_dg[model].to_dataset(name='delta_e_ys_dg')
            ds_out['delta_e_yt_dg'] = delta_e_yt_dg[model]
            ds_out['delta_e_yh_dg'] = delta_e_yh_dg[model]
            ds_out['delta_ys_dg'] = delta_ys_dg[model]
            ds_out['delta_yt_dg'] = delta_yt_dg[model]
            ds_out['delta_yh_dg'] = delta_yh_dg[model]
            
            return ds_out
        
        init_time = timeit.time.time()
        delta_out = wrap().compute()
        delta_e_ys_dg[model] = delta_out['delta_e_ys_dg']
        delta_e_yt_dg[model] = delta_out['delta_e_yt_dg']
        delta_e_yh_dg[model] = delta_out['delta_e_yh_dg']
        delta_ys_dg[model] = delta_out['delta_ys_dg']
        delta_yt_dg[model] = delta_out['delta_yt_dg']
        delta_yh_dg[model] = delta_out['delta_yh_dg']
        print(str(round((timeit.time.time()-init_time)/60,2))+' min')
        
        
        # mas lento
        # delta_e_ys_dg[model], delta_e_yt_dg[model], delta_e_yh_dg[model], delta_ys_dg[model], delta_yt_dg[model], delta_yh_dg[model] = xr.apply_ufunc(boots, ys_e_cut_dg[model], yt_e_cut_dg[model], yh_e_cut_dg[model], ys_c_cut_dg[model], yt_c_cut_dg[model], yh_c_cut_dg[model],
        #                                                                                                                         input_core_dims=[[timename], [timename], [timename], ['evtypes', timename], [timename], [timename]],
        #                                                                                                                         output_core_dims=[[], [], [], ['boots'], ['boots'], ['boots']], vectorize=True, dask='parallelized', 
        #                                                                                                                         dask_gufunc_kwargs={'allow_rechunk':True, 'output_sizes':{'boots':bootslength}})
    
        print('Saving arrays in temp')
        delta_e_ys_dg[model].to_netcdf(temp_path+'/delta_e_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
        delta_e_yt_dg[model].to_netcdf(temp_path+'/delta_e_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
        delta_e_yh_dg[model].to_netcdf(temp_path+'/delta_e_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
        delta_ys_dg[model].to_netcdf(temp_path+'/delta_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
        delta_yt_dg[model].to_netcdf(temp_path+'/delta_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
        delta_yh_dg[model].to_netcdf(temp_path+'/delta_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
        
        print('reloading arrays')
    
        delta_e_ys_dg[model] = xr.open_dataarray(temp_path+'/delta_e_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
        delta_e_yt_dg[model] = xr.open_dataarray(temp_path+'/delta_e_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
        delta_e_yh_dg[model] = xr.open_dataarray(temp_path+'/delta_e_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
        delta_ys_dg[model] = xr.open_dataarray(temp_path+'/delta_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
        delta_yt_dg[model] = xr.open_dataarray(temp_path+'/delta_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')
        delta_yh_dg[model] = xr.open_dataarray(temp_path+'/delta_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'.nc')

    else:
        # delta_e_ys[model], delta_e_yt[model], delta_e_yh[model], delta_ys[model], delta_yt[model], delta_yh[model] = xr.apply_ufunc(boots, ys_e_cut[model], yt_e_cut[model], yh_e_cut[model], ys_c_cut[model], yt_c_cut[model], yh_c_cut[model],
        #                                                                                                                         input_core_dims=[[timename], [timename], [timename], ['evtypes', timename], [timename], [timename]],
        #                                                                                                                         output_core_dims=[[], [], [], ['boots'], ['boots'], ['boots']], vectorize=True, dask='parallelized',
        #                                                                                                                         dask_gufunc_kwargs={'allow_rechunk':True, 'output_sizes':{'boots':bootslength}})

        def wrap():
            delta_e_ys[model], delta_e_yt[model], delta_e_yh[model], delta_ys[model], delta_yt[model], delta_yh[model] = xr.apply_ufunc(boots, ys_e_cut[model], yt_e_cut[model], yh_e_cut[model], ys_c_cut[model], yt_c_cut[model], yh_c_cut[model],
                                                                                                                                    input_core_dims=[[timename], [timename], [timename], ['evtypes', timename], [timename], [timename]],
                                                                                                                                    output_core_dims=[[], [], [], ['boots'], ['boots'], ['boots']], vectorize=True, dask='parallelized', 
                                                                                                                                    dask_gufunc_kwargs={'allow_rechunk':True, 'output_sizes':{'boots':bootslength}})
            ds_out = delta_e_ys[model].to_dataset(name='delta_e_ys')
            ds_out['delta_e_yt'] = delta_e_yt[model]
            ds_out['delta_e_yh'] = delta_e_yh[model]
            ds_out['delta_ys'] = delta_ys[model]
            ds_out['delta_yt'] = delta_yt[model]
            ds_out['delta_yh'] = delta_yh[model]
            
            return ds_out
        
        init_time = timeit.time.time()
        delta_out = wrap().compute()
        delta_e_ys[model] = delta_out['delta_e_ys']
        delta_e_yt[model] = delta_out['delta_e_yt']
        delta_e_yh[model] = delta_out['delta_e_yh']
        delta_ys[model] = delta_out['delta_ys']
        delta_yt[model] = delta_out['delta_yt']
        delta_yh[model] = delta_out['delta_yh']
        print(str(round((timeit.time.time()-init_time)/60,2))+' min')
    
        print('Saving arrays in temp')
        delta_e_ys[model].to_netcdf(temp_path+'/delta_e_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        delta_e_yt[model].to_netcdf(temp_path+'/delta_e_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        delta_e_yh[model].to_netcdf(temp_path+'/delta_e_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        delta_ys[model].to_netcdf(temp_path+'/delta_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        delta_yt[model].to_netcdf(temp_path+'/delta_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        delta_yh[model].to_netcdf(temp_path+'/delta_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        
        print('reloading arrays')
    
        delta_e_ys[model] = xr.open_dataarray(temp_path+'/delta_e_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        delta_e_yt[model] = xr.open_dataarray(temp_path+'/delta_e_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        delta_e_yh[model] = xr.open_dataarray(temp_path+'/delta_e_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        delta_ys[model] = xr.open_dataarray(temp_path+'/delta_ys_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        delta_yt[model] = xr.open_dataarray(temp_path+'/delta_yt_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')
        delta_yh[model] = xr.open_dataarray(temp_path+'/delta_yh_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'.nc')

#---

print('... separando los regimenes dinamicos ...')
delta_e_ys_dynreg = dict()
delta_e_yt_dynreg = dict()
delta_e_yh_dynreg = dict()

delta_ys_dynreg = dict()
delta_yt_dynreg = dict()
delta_yh_dynreg = dict()

cond_dr = dict()

if degrade:
    ys_e_cut_dg_dynreg = dict()
    yt_e_cut_dg_dynreg = dict()
    yh_e_cut_dg_dynreg = dict()
    ys_c_cut_dg_dynreg = dict()
    yt_c_cut_dg_dynreg = dict()
    yh_c_cut_dg_dynreg = dict()

    delta_e_ys_dg_dynreg = dict()
    delta_e_yt_dg_dynreg = dict()
    delta_e_yh_dg_dynreg = dict()
    
    delta_ys_dg_dynreg = dict()
    delta_yt_dg_dynreg = dict()
    delta_yh_dg_dynreg = dict()


for model in models.keys():
    
    print('..... '+model+' ....')

    timename='time'
    if model=='JRA-55': timename='initial_time0_hours'

    lat_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lat" in coord][0]
    lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]

    delta_e_ys_dynreg[model] = dict()
    delta_e_yt_dynreg[model] = dict()
    delta_e_yh_dynreg[model] = dict()
    
    delta_ys_dynreg[model] = dict()
    delta_yt_dynreg[model] = dict()
    delta_yh_dynreg[model] = dict()
    
    # calculo los terciles de high/mid/low dynamic regimes
    auxabs = xr.ufuncs.fabs(data_day_cut[model]['vimfc2d'])
    data_day_cut_terc = auxabs.quantile([0.33, 0.66], dim=timename)

    # condicion de los terciles
    cond_dr[model] = dict()
    cond_dr[model]['low'] = auxabs<data_day_cut_terc[0]
    cond_dr[model]['mid'] = (auxabs>data_day_cut_terc[0]) * (auxabs<data_day_cut_terc[1])
    cond_dr[model]['high'] = auxabs>data_day_cut_terc[1]

    if degrade:
        ys_e_cut_dg_dynreg[model] = dict()
        yt_e_cut_dg_dynreg[model] = dict()
        yh_e_cut_dg_dynreg[model] = dict()
        ys_c_cut_dg_dynreg[model] = dict()
        yt_c_cut_dg_dynreg[model] = dict()
        yh_c_cut_dg_dynreg[model] = dict()

        delta_e_ys_dg_dynreg[model] = dict()
        delta_e_yt_dg_dynreg[model] = dict()
        delta_e_yh_dg_dynreg[model] = dict()
        delta_ys_dg_dynreg[model] = dict()
        delta_yt_dg_dynreg[model] = dict()
        delta_yh_dg_dynreg[model] = dict()


        res_lat = [None if len(ys_e_cut[model][lat_name])%degrade_n==0 else len(ys_e_cut[model][lat_name])%degrade_n*-1][0]
        res_lon = [None if len(ys_e_cut[model][lon_name])%degrade_n==0 else len(ys_e_cut[model][lon_name])%degrade_n*-1][0]
        
        for dr in ['low', 'mid', 'high']:
            print(dr)

            ys_e_cut_dg_dynreg[model][dr] = xr.concat([ys_e_cut[model].where(cond_dr[model][dr])[:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
            yt_e_cut_dg_dynreg[model][dr] = xr.concat([yt_e_cut[model].where(cond_dr[model][dr])[:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
            yh_e_cut_dg_dynreg[model][dr] = xr.concat([yh_e_cut[model].where(cond_dr[model][dr])[:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
            ys_c_cut_dg_dynreg[model][dr] = xr.concat([ys_c_cut[model].where(cond_dr[model][dr])[:,:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
            yt_c_cut_dg_dynreg[model][dr] = xr.concat([yt_c_cut[model].where(cond_dr[model][dr])[:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
            yh_c_cut_dg_dynreg[model][dr] = xr.concat([yh_c_cut[model].where(cond_dr[model][dr])[:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')

            def wrap():
                delta_e_ys_dg_dynreg[model][dr], delta_e_yt_dg_dynreg[model][dr], delta_e_yh_dg_dynreg[model][dr],\
                delta_ys_dg_dynreg[model][dr], delta_yt_dg_dynreg[model][dr], delta_yh_dg_dynreg[model][dr] =   xr.apply_ufunc(boots, ys_e_cut_dg_dynreg[model][dr], yt_e_cut_dg_dynreg[model][dr], yh_e_cut_dg_dynreg[model][dr],
                                                                                                                ys_c_cut_dg_dynreg[model][dr], yt_c_cut_dg_dynreg[model][dr], yh_c_cut_dg_dynreg[model][dr],
                                                                                                                input_core_dims=[[timename], [timename], [timename], ['evtypes', timename], [timename], [timename]],
                                                                                                                output_core_dims=[[], [], [], ['boots'], ['boots'], ['boots']], vectorize=True, dask='parallelized',
                                                                                                                dask_gufunc_kwargs={'allow_rechunk':True, 'output_sizes':{'boots':bootslength}})
                ds_out = delta_e_ys_dg_dynreg[model][dr].to_dataset(name='delta_e_ys_dg_dynreg')
                ds_out['delta_e_yt_dg_dynreg'] = delta_e_yt_dg_dynreg[model][dr]
                ds_out['delta_e_yh_dg_dynreg'] = delta_e_yh_dg_dynreg[model][dr]
                ds_out['delta_ys_dg_dynreg'] = delta_ys_dg_dynreg[model][dr]
                ds_out['delta_yt_dg_dynreg'] = delta_yt_dg_dynreg[model][dr]
                ds_out['delta_yh_dg_dynreg'] = delta_yh_dg_dynreg[model][dr]
                
                return ds_out
            
            init_time = timeit.time.time()
            delta_out = wrap().compute()
            delta_e_ys_dg_dynreg[model][dr] = delta_out['delta_e_ys_dg_dynreg'].copy()
            delta_e_yt_dg_dynreg[model][dr] = delta_out['delta_e_yt_dg_dynreg'].copy()
            delta_e_yh_dg_dynreg[model][dr] = delta_out['delta_e_yh_dg_dynreg'].copy()
            delta_ys_dg_dynreg[model][dr] = delta_out['delta_ys_dg_dynreg'].copy()
            delta_yt_dg_dynreg[model][dr] = delta_out['delta_yt_dg_dynreg'].copy()
            delta_yh_dg_dynreg[model][dr] = delta_out['delta_yh_dg_dynreg'].copy()
            print(str(round((timeit.time.time()-init_time)/60,2))+' min')
                            
            # mas lento
            # delta_e_ys_dg_dynreg[model][dr], delta_e_yt_dg_dynreg[model][dr], delta_e_yh_dg_dynreg[model][dr],\
            # delta_ys_dg_dynreg[model][dr], delta_yt_dg_dynreg[model][dr], delta_yh_dg_dynreg[model][dr] = xr.apply_ufunc(boots, ys_e_cut_dg_dynreg[model][dr], yt_e_cut_dg_dynreg[model][dr], yh_e_cut_dg_dynreg[model][dr],
            #                                                                                                     ys_c_cut_dg_dynreg[model][dr], yt_c_cut_dg_dynreg[model][dr], yh_c_cut_dg_dynreg[model][dr],
            #                                                                                                     input_core_dims=[[timename], [timename], [timename], ['evtypes', timename], [timename], [timename]],
            #                                                                                                     output_core_dims=[[], [], [], ['boots'], ['boots'], ['boots']], vectorize=True, dask='parallelized',
            #                                                                                                     dask_gufunc_kwargs={'allow_rechunk':True, 'output_sizes':{'boots':bootslength}})

        print('Saving arrays in temp')
        for dr in ['low', 'mid', 'high']:
            print(dr)
            delta_e_ys_dg_dynreg[model][dr].to_netcdf(temp_path+'/delta_e_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
            delta_e_yt_dg_dynreg[model][dr].to_netcdf(temp_path+'/delta_e_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
            delta_e_yh_dg_dynreg[model][dr].to_netcdf(temp_path+'/delta_e_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
            delta_ys_dg_dynreg[model][dr].to_netcdf(temp_path+'/delta_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
            delta_yt_dg_dynreg[model][dr].to_netcdf(temp_path+'/delta_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
            delta_yh_dg_dynreg[model][dr].to_netcdf(temp_path+'/delta_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
    
        print('reloading arrays')
        for dr in ['low', 'mid', 'high']:
            delta_e_ys_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
            delta_e_yt_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
            delta_e_yh_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
            delta_ys_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
            delta_yt_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')
            delta_yh_dg_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+'_'+dr+'.nc')

    else:
    
        for dr in ['low', 'mid', 'high']:
            print(dr)
            
            def wrap():
                delta_e_ys_dynreg[model][dr], delta_e_yt_dynreg[model][dr], delta_e_yh_dynreg[model][dr],\
                delta_ys_dynreg[model][dr], delta_yt_dynreg[model][dr], delta_yh_dynreg[model][dr] = xr.apply_ufunc(boots, ys_e_cut[model].where(cond_dr[model][dr]), yt_e_cut[model].where(cond_dr[model][dr]), yh_e_cut[model].where(cond_dr[model][dr]),
                                                                                                                    ys_c_cut[model].where(cond_dr[model][dr]), yt_c_cut[model].where(cond_dr[model][dr]), yh_c_cut[model].where(cond_dr[model][dr]),
                                                                                                                    input_core_dims=[[timename], [timename], [timename], ['evtypes', timename], [timename], [timename]],
                                                                                                                    output_core_dims=[[], [], [], ['boots'], ['boots'], ['boots']], vectorize=True, dask='parallelized',
                                                                                                                    dask_gufunc_kwargs={'allow_rechunk':True, 'output_sizes':{'boots':bootslength}})

                ds_out = delta_e_ys_dynreg[model][dr].to_dataset(name='delta_e_ys_dynreg')
                ds_out['delta_e_yt_dynreg'] = delta_e_yt_dynreg[model][dr]
                ds_out['delta_e_yh_dynreg'] = delta_e_yh_dynreg[model][dr]
                ds_out['delta_ys_dynreg'] = delta_ys_dynreg[model][dr]
                ds_out['delta_yt_dynreg'] = delta_yt_dynreg[model][dr]
                ds_out['delta_yh_dynreg'] = delta_yh_dynreg[model][dr]
                
                return ds_out
            
            init_time = timeit.time.time()
            delta_out = wrap().compute()
            delta_e_ys_dynreg[model][dr] = delta_out['delta_e_ys_dynreg'].copy()
            delta_e_yt_dynreg[model][dr] = delta_out['delta_e_yt_dynreg'].copy()
            delta_e_yh_dynreg[model][dr] = delta_out['delta_e_yh_dynreg'].copy()
            delta_ys_dynreg[model][dr] = delta_out['delta_ys_dynreg'].copy()
            delta_yt_dynreg[model][dr] = delta_out['delta_yt_dynreg'].copy()
            delta_yh_dynreg[model][dr] = delta_out['delta_yh_dynreg'].copy()
            print(str(round((timeit.time.time()-init_time)/60,2))+' min')


    
        print('Saving arrays in temp')
        for dr in ['low', 'mid', 'high']:
            print(dr)
            delta_e_ys_dynreg[model][dr].to_netcdf(temp_path+'/delta_e_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
            delta_e_yt_dynreg[model][dr].to_netcdf(temp_path+'/delta_e_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
            delta_e_yh_dynreg[model][dr].to_netcdf(temp_path+'/delta_e_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
            delta_ys_dynreg[model][dr].to_netcdf(temp_path+'/delta_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
            delta_yt_dynreg[model][dr].to_netcdf(temp_path+'/delta_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
            delta_yh_dynreg[model][dr].to_netcdf(temp_path+'/delta_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
    
        print('reloading arrays')
        for dr in ['low', 'mid', 'high']:
            delta_e_ys_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
            delta_e_yt_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
            delta_e_yh_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_e_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
            delta_ys_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_ys_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
            delta_yt_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_yt_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')
            delta_yh_dynreg[model][dr] = xr.open_dataarray(temp_path+'/delta_yh_dynreg_'+model+'_'+delta_period[0]+'-'+delta_period[1]+'_'+dr+'.nc')



#%% ------------  PLOTS (pcolormesh) ---------------
############################################
##
#           PLOTS (pcolormesh)
##
############################################

#######################################
# Plot of delta percentile (pixeled version)



# calculo los percentiles
delta_e_ys_perc = dict()
delta_e_yt_perc = dict()
delta_e_yh_perc = dict()

delta_e_ys_dg_perc = dict()
delta_e_yt_dg_perc = dict()
delta_e_yh_dg_perc = dict()

for model in models.keys():
    if degrade:

        delta_e_ys_dg_perc[model] = xr.apply_ufunc(stats.percentileofscore, delta_ys_dg[model], delta_e_ys_dg[model], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk':True})
        delta_e_yt_dg_perc[model] = xr.apply_ufunc(stats.percentileofscore, delta_yt_dg[model], delta_e_yt_dg[model], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk':True})
        delta_e_yh_dg_perc[model] = xr.apply_ufunc(stats.percentileofscore, delta_yh_dg[model], delta_e_yh_dg[model], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk':True})
    
    else:
        delta_e_ys_perc[model] = xr.apply_ufunc(stats.percentileofscore, delta_ys[model], delta_e_ys[model], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk':True})
        delta_e_yt_perc[model] = xr.apply_ufunc(stats.percentileofscore, delta_yt[model], delta_e_yt[model], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk':True})
        delta_e_yh_perc[model] = xr.apply_ufunc(stats.percentileofscore, delta_yh[model], delta_e_yh[model], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk':True})
    
plot_titles = ['Spatial preference: '+r'$\mathrm{\delta_e(Y^s)}$',
               'Temporal preference: '+r'$\mathrm{\delta_e(Y^t)}$',
               'Heterogeneity preference: '+r'$\mathrm{\delta_e(Y^h)}$']

# si estoy recargando resultados anteriores tengo que volver a calcular esto
if 'ys_e_cut_dg' not in locals() or 'ys_e_cut_dg' not in globals():
    ys_e_cut_dg = dict()
            

for model in models.keys():
    
    if degrade:
        deltas = [delta_e_ys_dg_perc[model], delta_e_yt_dg_perc[model], delta_e_yh_dg_perc[model]]
    else:
        deltas = [delta_e_ys_perc[model], delta_e_yt_perc[model], delta_e_yh_perc[model]]
    
    # mascara para los puntos con menos de min_events eventos:
    
    if degrade:
        lat_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lat" in coord][0]
        lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]
        res_lat = [None if len(ys_e_cut[model][lat_name])%degrade_n==0 else len(ys_e_cut[model][lat_name])%degrade_n*-1][0]
        res_lon = [None if len(ys_e_cut[model][lon_name])%degrade_n==0 else len(ys_e_cut[model][lon_name])%degrade_n*-1][0]
        if model not in ys_e_cut_dg.keys():
            ys_e_cut_dg[model] = xr.concat([ys_e_cut[model][:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
        min_ev_mask = ys_e_cut_dg[model].count(axis=0).compute()

    else:
        min_ev_mask = ys_e_cut[model].count(axis=0).compute()
    
    
    # crear figura y axes
    fig1, ax = juli_functions.make_figure(model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(13,5), general_fontsize=9)

    clevs = np.array([0,1,2.5,5,10,90,95,97.5,99,100])        # Esta es la escala que usa Guillod

    # barra de colores
    barra= juli_functions.barra_whitecenter(clevs ,colormap=matplotlib.cm.get_cmap('RdBu'), no_extreme_colors=True, normalize=False)
    
    barra = mcolors.ListedColormap([(235/255,235/255,235/255) if x in np.arange(int(barra.N*0.4), int(barra.N*0.6)) else barra(x) for x in np.arange(0, barra.N)])

    # Si son muchos plots
    ax.set_visible(False)
    gs = gridspec.GridSpec(1,3, wspace=0, hspace=0.2) #si quiero multiples subplos
    
    for m, delta in enumerate(deltas):
        # Poner en loop si son muchos plot
        ax1 = plt.subplot(gs[m], projection = proj)
        
        # hago cada plot
        if degrade:
            CS1 = juli_functions.plot_pcolormesh(ax1, delta.where(min_ev_mask>=min_events), lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                                 clevs, barra, numbering[m]+' '+plot_titles[m], titlesize = 12, coastwidth = 1, countrywidth = 1)
        else:
            CS1 = juli_functions.plot_pcolormesh(ax1, delta.where(min_ev_mask>=min_events), lon[model], lat[model], lonproj[model], latproj[model], proj,
                                                 clevs, barra, numbering[m]+' '+plot_titles[m], titlesize = 12, coastwidth = 1, countrywidth = 1)
            
            # anotar el nro de puntos significativos:
            # plt.text(-48,5, str(int( (delta.where(min_ev_mask>=min_events)>90).sum() + (delta.where(min_ev_mask>=min_events)<10).sum() ) ) )
        
    # add colorbar
    #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
    juli_functions.add_colorbar(fig1, CS1, 'neither', '%', cbaxes= [0.9, 0.2, 0.01, 0.6]) #[*left*, *bottom*, *width*,  *height*]
    
    if min_events==0:
        # save 
        juli_functions.savefig(fig1, images_path, 'delta_percentile_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')
    
    else:
        # save 
        juli_functions.savefig(fig1, images_path, 'delta_percentile_min_'+str(min_events)+'_evs_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')
    




# plots de CTL enmascarado donde CLIM es significativo
for model in ['RCA4']:
    if 'CLIM' not in model and model+'CLIM' in models.keys():
        if degrade:
            deltas = [delta_e_ys_dg_perc[model], delta_e_yt_dg_perc[model], delta_e_yh_dg_perc[model]]
            deltasclim = [delta_e_ys_dg_perc[model+'CLIM'], delta_e_yt_dg_perc[model+'CLIM'], delta_e_yh_dg_perc[model+'CLIM']]
        else: 
            deltas = [delta_e_ys_perc[model], delta_e_yt_perc[model], delta_e_yh_perc[model]]
            deltasclim = [delta_e_ys_perc[model+'CLIM'], delta_e_yt_perc[model+'CLIM'], delta_e_yh_perc[model+'CLIM']]
        
        # mascara para los puntos con menos de min_events eventos:
        if degrade:
            min_ev_mask = ys_e_cut_dg[model].count(axis=0).compute()>=min_events
            res_lat = [None if len(ys_e_cut[model][lat_name])%degrade_n==0 else len(ys_e_cut[model][lat_name])%degrade_n*-1][0]
            res_lon = [None if len(ys_e_cut[model][lon_name])%degrade_n==0 else len(ys_e_cut[model][lon_name])%degrade_n*-1][0]
    
        else:
            min_ev_mask = ys_e_cut[model].count(axis=0).compute()>=min_events
        
        # crear figura y axes
        fig1, ax = juli_functions.make_figure(model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(13,5), general_fontsize=9)
    
        clevs = np.array([0,1,2.5,5,10,90,95,97.5,99,100])        # Esta es la escala que usa Guillod
    
        # barra de colores
        barra= juli_functions.barra_whitecenter(clevs ,colormap=matplotlib.cm.get_cmap('RdBu'), no_extreme_colors=True, normalize=False)
        
        barra = mcolors.ListedColormap([(235/255,235/255,235/255) if x in np.arange(int(barra.N*0.4), int(barra.N*0.6)) else barra(x) for x in np.arange(0, barra.N)])
    
        # Si son muchos plots
        ax.set_visible(False)
        gs = gridspec.GridSpec(1,3, wspace=0, hspace=0.2) #si quiero multiples subplos
        
        for m, delta in enumerate(deltas):

            # mascara para valores de CLIM significativos (tambien cumple la condicion de min_events)
            if degrade:
                clim_mask1 =  ( (ys_e_cut_dg[model+'CLIM'].count(axis=0)>=min_events) * (deltasclim[m]>90))
                clim_mask2 =  ( (ys_e_cut_dg[model+'CLIM'].count(axis=0)>=min_events) * (deltasclim[m]<10))
                
            else:
                clim_mask1 =  ( (ys_e_cut[model+'CLIM'].count(axis=0)>=min_events) * (deltasclim[m]>90))
                clim_mask2 =  ( (ys_e_cut[model+'CLIM'].count(axis=0)>=min_events) * (deltasclim[m]<10))

            clim_mask = ~ (clim_mask1 + clim_mask2)


            # Poner en loop si son muchos plot
            ax1 = plt.subplot(gs[m], projection = proj)
            
            # hago cada plot
            if degrade:
                CS1 = juli_functions.plot_pcolormesh(ax1, delta.where(min_ev_mask*clim_mask), lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                                     clevs, barra, numbering[m]+' '+plot_titles[m], titlesize = 12, coastwidth = 1, countrywidth = 1)
            else:
                CS1 = juli_functions.plot_pcolormesh(ax1, delta.where(min_ev_mask*clim_mask), lon[model], lat[model], lonproj[model], latproj[model], proj,
                                                     clevs, barra, numbering[m]+' '+plot_titles[m], titlesize = 12, coastwidth = 1, countrywidth = 1)

        # add colorbar
        #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
        juli_functions.add_colorbar(fig1, CS1, 'neither', '%', cbaxes= [0.9, 0.2, 0.01, 0.6]) #[*left*, *bottom*, *width*,  *height*]
        
        if min_events==0:
            # save 
            juli_functions.savefig(fig1, images_path, 'delta_percentile_'+model+'-'+model+'CLIM'+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')
        
        else:
            # save 
            juli_functions.savefig(fig1, images_path, 'delta_percentile_min_'+str(min_events)+'_evs_'+model+'-'+model+'CLIM'+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')

#%% ------------  PLOTS (pcolormesh) SEPARADO POR REGIMENES ---------------
# Plot of delta percentile (pixeled version)
# SEPARADO POR REGIMENES

from scipy import stats

# calculo los percentiles
delta_e_ys_perc = dict()
delta_e_yt_perc = dict()
delta_e_yh_perc = dict()

delta_e_ys_dg_perc = dict()
delta_e_yt_dg_perc = dict()
delta_e_yh_dg_perc = dict()


for model in models.keys():
    
    if degrade:
        delta_e_ys_dg_perc[model] = dict()
        delta_e_yt_dg_perc[model] = dict()
        delta_e_yh_dg_perc[model] = dict()
    
        for dr in ['low', 'mid', 'high']:
        
            delta_e_ys_dg_perc[model][dr] = xr.apply_ufunc(stats.percentileofscore, delta_ys_dg_dynreg[model][dr], delta_e_ys_dg_dynreg[model][dr], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized')
            delta_e_yt_dg_perc[model][dr] = xr.apply_ufunc(stats.percentileofscore, delta_yt_dg_dynreg[model][dr], delta_e_yt_dg_dynreg[model][dr], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized')
            delta_e_yh_dg_perc[model][dr] = xr.apply_ufunc(stats.percentileofscore, delta_yh_dg_dynreg[model][dr], delta_e_yh_dg_dynreg[model][dr], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized')
        
    else:
        delta_e_ys_perc[model] = dict()
        delta_e_yt_perc[model] = dict()
        delta_e_yh_perc[model] = dict()
    
        for dr in ['low', 'mid', 'high']:
        
            delta_e_ys_perc[model][dr] = xr.apply_ufunc(stats.percentileofscore, delta_ys_dynreg[model][dr], delta_e_ys_dynreg[model][dr], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized')
            delta_e_yt_perc[model][dr] = xr.apply_ufunc(stats.percentileofscore, delta_yt_dynreg[model][dr], delta_e_yt_dynreg[model][dr], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized')
            delta_e_yh_perc[model][dr] = xr.apply_ufunc(stats.percentileofscore, delta_yh_dynreg[model][dr], delta_e_yh_dynreg[model][dr], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized')
    
plot_titles = ['Spatial preference: '+r'$\mathrm{\delta_e(Y^s)}$',
               'Temporal preference: '+r'$\mathrm{\delta_e(Y^t)}$',
               'Heterogeneity preference: '+r'$\mathrm{\delta_e(Y^h)}$']

for model in models.keys():
    
    for dr in ['low', 'mid', 'high']:
        if degrade:
            deltas = [delta_e_ys_dg_perc[model][dr], delta_e_yt_dg_perc[model][dr], delta_e_yh_dg_perc[model][dr]]
        else:
            deltas = [delta_e_ys_perc[model][dr], delta_e_yt_perc[model][dr], delta_e_yh_perc[model][dr]]
        
        # mascara para los puntos con menos de min_events eventos:
        if degrade:
            min_ev_mask = ys_e_cut_dg_dynreg[model][dr].count(axis=0).compute()
            res_lat = [None if len(ys_e_cut_dg_dynreg[model][dr][lat_name])%degrade_n==0 else len(ys_e_cut_dg_dynreg[model][dr][lat_name])%degrade_n*-1][0]
            res_lon = [None if len(ys_e_cut_dg_dynreg[model][dr][lon_name])%degrade_n==0 else len(ys_e_cut_dg_dynreg[model][dr][lon_name])%degrade_n*-1][0]
    
        else:
            min_ev_mask = ys_e_cut[model].where(cond_dr[model][dr]).count(axis=0).compute()
        
        
        # crear figura y axes
        fig1, ax = juli_functions.make_figure(model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1]+' | '+dr+' dynamic regime', figsize=(13,5), general_fontsize=9)
    
        clevs = np.array([0,1,2.5,5,10,90,95,97.5,99,100])        # Esta es la escala que usa Guillod
    
        # barra de colores
        barra= juli_functions.barra_whitecenter(clevs ,colormap=matplotlib.cm.get_cmap('RdBu'), no_extreme_colors=True, normalize=False)
        
        barra = mcolors.ListedColormap([(235/255,235/255,235/255) if x in np.arange(int(barra.N*0.4), int(barra.N*0.6)) else barra(x) for x in np.arange(0, barra.N)])
    
        # Si son muchos plots
        ax.set_visible(False)
        gs = gridspec.GridSpec(1,3, wspace=0, hspace=0.2) #si quiero multiples subplos
        
        for m, delta in enumerate(deltas):
            # Poner en loop si son muchos plot
            ax1 = plt.subplot(gs[m], projection = proj)
            
            # hago cada plot
            if degrade:
                CS1 = juli_functions.plot_pcolormesh(ax1, delta.where(min_ev_mask>=min_events), lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                                     clevs, barra, numbering[m]+' '+plot_titles[m], titlesize = 12, coastwidth = 1, countrywidth = 1)
            else:
                CS1 = juli_functions.plot_pcolormesh(ax1, delta.where(min_ev_mask>=min_events), lon[model], lat[model], lonproj[model], latproj[model], proj,
                                                     clevs, barra, numbering[m]+' '+plot_titles[m], titlesize = 12, coastwidth = 1, countrywidth = 1)
            
        # add colorbar
        #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
        juli_functions.add_colorbar(fig1, CS1, 'neither', '%', cbaxes= [0.9, 0.2, 0.01, 0.6]) #[*left*, *bottom*, *width*,  *height*]
        
        if min_events==0:
            # save 
            juli_functions.savefig(fig1, images_path, 'delta_percentile_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+'_'+dr+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')
        
        else:
            # save 
            juli_functions.savefig(fig1, images_path, 'delta_percentile_min_'+str(min_events)+'_evs_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+'_'+dr+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')
        

#%% --------- Plot of number of events (pixeled version) -------------
#############################################


if load_deltas:
    cond_dr=dict()
    ys_e_cut_dg=dict()
    ys_e_cut_dg_dynreg=dict()
        
    for model in models.keys():
        print('... '+model+' ...')
    
        timename='time'
        if model=='JRA-55': timename='initial_time0_hours'
        
        lat_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lat" in coord][0]
        lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]
        
        # calculo los terciles de high/mid/low dynamic regimes
        auxabs = xr.ufuncs.fabs(data_day_cut[model]['vimfc2d'])
        data_day_cut_terc = auxabs.quantile([0.33, 0.66], dim=timename)
        
        # condicion de los terciles
        cond_dr[model] = dict()
        cond_dr[model]['low'] = auxabs<data_day_cut_terc[0]
        cond_dr[model]['mid'] = (auxabs>data_day_cut_terc[0]) * (auxabs<data_day_cut_terc[1])
        cond_dr[model]['high'] = auxabs>data_day_cut_terc[1]

        if degrade:
            print('calculando ys_e_cut_dg')

            res_lat = [None if len(ys_e_cut[model][lat_name])%degrade_n==0 else len(ys_e_cut[model][lat_name])%degrade_n*-1][0]
            res_lon = [None if len(ys_e_cut[model][lon_name])%degrade_n==0 else len(ys_e_cut[model][lon_name])%degrade_n*-1][0]
            
            ys_e_cut_dg[model] = xr.concat([ys_e_cut[model][:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override').compute()
            
            ys_e_cut_dg_dynreg[model]=dict()
            
            for dr in ['low', 'mid', 'high']:
                ys_e_cut_dg_dynreg[model][dr] = xr.concat([ys_e_cut[model].where(cond_dr[model][dr])[:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override').compute()


#################### set colorbar.
preclevs = np.arange(0,110,10)
prebarra = matplotlib.cm.get_cmap('Reds').copy() # premade colorbar
prebarra.set_under('white')

preclevs2 = np.arange(200,1100,100)
prebarra2 = matplotlib.cm.get_cmap('summer').copy() # premade colorbar

# Nueva barra
barra = mcolors.ListedColormap([prebarra(int(round(x*prebarra.N/preclevs[-1]))) for x in preclevs[1:]]+[prebarra2(int(round(x*prebarra2.N/preclevs2[-1]))) for x in preclevs2[:]])
#barra.set_over('Navy')
clevs = np.concatenate((preclevs,preclevs2))


# clevs = [x for x in np.arange(0,110,10)]+[x for x in np.arange(200,1100,100)]
# barra = matplotlib.cm.get_cmap('YlOrRd').copy() # premade colorbar
# barra.set_under('white')

# new array with number of events
n_events = dict()
for model in models.keys():
    n_events[model] = dict()
    
    if degrade:
        n_events[model]['all'] = (ys_e_cut_dg[model].count(axis=0)).where(ys_e_cut_dg[model].count(axis=0)!=0)
        res_lat = [None if len(ys_e_cut_dg[model][lat_name])%degrade_n==0 else len(ys_e_cut_dg[model][lat_name])%degrade_n*-1][0]
        res_lon = [None if len(ys_e_cut_dg[model][lon_name])%degrade_n==0 else len(ys_e_cut_dg[model][lon_name])%degrade_n*-1][0]
    else:
        n_events[model]['all'] = (ys_e_cut[model].count(axis=0)).where(ys_e_cut[model].count(axis=0)!=0)


for model in models.keys():
    # crear figura y axes
    fig1, ax = juli_functions.make_figure('Number of afternoon precip events \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(5,5), general_fontsize=9)

    
    if degrade:
        CS1 = juli_functions.plot_pcolormesh(ax, n_events[model]['all'], lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                             clevs, barra, '', titlesize = 12, coastwidth = 1, countrywidth = 1)
    else:
        CS1 = juli_functions.plot_pcolormesh(ax, n_events[model]['all'], lon[model], lat[model], lonproj[model], latproj[model], proj,
                                             clevs, barra, '', titlesize = 12, coastwidth = 1, countrywidth = 1)
    
    # add colorbar
    #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
    juli_functions.add_colorbar(fig1, CS1, 'max', '', cbaxes= [0.9, 0.2, 0.02, 0.6]) #[*left*, *bottom*, *width*,  *height*]
    
    # save 
    juli_functions.savefig(fig1, images_path, 'n_events_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')



#############################################
# Plot of number of events (pixeled version) SEPARADO POR REGIMENES

# new array with number of events
for model in models.keys():
    
    for dr in ['low', 'mid', 'high']:
        if degrade:
            n_events[model][dr] = (ys_e_cut_dg_dynreg[model][dr].count(axis=0)).where(ys_e_cut_dg_dynreg[model][dr].count(axis=0)!=0)
        else:
            aux = ys_e_cut[model].where(cond_dr[model][dr]).count(axis=0)
            n_events[model][dr] = aux.where(aux!=0).copy()


for model in models.keys():
    # crear figura y axes
    fig1, ax = juli_functions.make_figure('Number of afternoon precip events \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(13,5), general_fontsize=9)


    # Si son muchos plots
    ax.set_visible(False)
    gs = gridspec.GridSpec(1,3, wspace=0, hspace=0.2) #si quiero multiples subplos
    
    for m, dr in enumerate(['low', 'mid', 'high']):
        # Poner en loop si son muchos plot
        ax1 = plt.subplot(gs[m], projection = proj)
        
        if degrade:
            CS1 = juli_functions.plot_pcolormesh(ax1, n_events[model][dr], lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                                 clevs, barra, numbering[m]+' '+dr+' dynamic regime', titlesize = 12, coastwidth = 1, countrywidth = 1)
        else:
            CS1 = juli_functions.plot_pcolormesh(ax1, n_events[model][dr], lon[model], lat[model], lonproj[model], latproj[model], proj,
                                                 clevs, barra, numbering[m]+' '+dr+' dynamic regime', titlesize = 12, coastwidth = 1, countrywidth = 1)
    # add colorbar
    #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
    juli_functions.add_colorbar(fig1, CS1, 'max', '', cbaxes= [0.9, 0.2, 0.01, 0.6]) #[*left*, *bottom*, *width*,  *height*]
    
    # save 
    juli_functions.savefig(fig1, images_path, 'n_events_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+'_regimes'+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')



#%%exit the program early
from sys import exit
exit()

#para limpiar variables
from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')


#%% ####### -----------------------  Number of consecutive afternoon P events (i.e. number of P events wich are precided by a P event on the previous day)
how = '%' # 'n' for absolute value of consecutive events, '%' for percentage of events that are consecutive


#################### set colorbar.
preclevs = np.arange(0,110,10)
prebarra = matplotlib.cm.get_cmap('Reds').copy() # premade colorbar
prebarra.set_under('white')

preclevs2 = np.arange(200,1100,100)
prebarra2 = matplotlib.cm.get_cmap('summer').copy() # premade colorbar

# Nueva barra
barra = mcolors.ListedColormap([prebarra(int(round(x*prebarra.N/preclevs[-1]))) for x in preclevs[1:]]+[prebarra2(int(round(x*prebarra2.N/preclevs2[-1]))) for x in preclevs2[:]])
#barra.set_over('Navy')
clevs = np.concatenate((preclevs,preclevs2))

if how=='%':
    clevs = np.arange(0,30,1)
    barra = matplotlib.cm.get_cmap('Reds').copy() # premade colorbar
    barra.set_under('white')

# new array with number of events
n_cevents = dict()
for model in models.keys():
    n_cevents[model] = dict()
    
    if degrade:
        shifted = ~np.isnan(ys_e_cut_dg[model].shift({timename:-1}))
        consec = shifted * (~np.isnan(ys_e_cut_dg[model]))
        n_cevents[model]['all'] = (consec.sum(axis=0)).where(consec.sum(axis=0)!=0)
        res_lat = [None if len(ys_e_cut_dg[model][lat_name])%degrade_n==0 else len(ys_e_cut_dg[model][lat_name])%degrade_n*-1][0]
        res_lon = [None if len(ys_e_cut_dg[model][lon_name])%degrade_n==0 else len(ys_e_cut_dg[model][lon_name])%degrade_n*-1][0]
    else:
        shifted = ~np.isnan(ys_e_cut[model].shift({timename:-1}))
        consec = shifted * (~np.isnan(ys_e_cut[model]))
        n_cevents[model]['all'] = (consec.sum(axis=0)).where(consec.sum(axis=0)!=0)


for model in models.keys():
    # crear figura y axes
    fig1, ax = juli_functions.make_figure(['Number' if how=='n' else '%' if how=='%' else 'wrong "how" selection'][0]+' of consecutive afternoon precip events \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(5,5), general_fontsize=9)

    
    plot = [n_cevents[model]['all'] if how=='n' else n_cevents[model]['all']/n_events[model]['all']*100 ][0]

    if degrade:
        CS1 = juli_functions.plot_pcolormesh(ax, plot, lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                             clevs, barra, '', titlesize = 12, coastwidth = 1, countrywidth = 1)
    else:
        CS1 = juli_functions.plot_pcolormesh(ax, plot, lon[model], lat[model], lonproj[model], latproj[model], proj,
                                             clevs, barra, '', titlesize = 12, coastwidth = 1, countrywidth = 1)
    
    # add colorbar
    #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
    juli_functions.add_colorbar(fig1, CS1, 'max', '', cbaxes= [0.9, 0.2, 0.02, 0.6]) #[*left*, *bottom*, *width*,  *height*]
    
    # save 
    juli_functions.savefig(fig1, images_path, 'n_consecutive_events_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')



#############################################
# Plot of number of events (pixeled version) SEPARADO POR REGIMENES

# new array with number of events
for model in models.keys():
    
    for dr in ['low', 'mid', 'high']:
        if degrade:
            shifted = ~np.isnan(ys_e_cut_dg_dynreg[model][dr].shift({timename:-1}))
            consec = shifted * (~np.isnan(ys_e_cut_dg_dynreg[model][dr]))
            n_cevents[model][dr] = (consec.sum(axis=0)).where(consec.sum(axis=0)!=0)
        else:
            shifted = ~np.isnan(cond_dr[model][dr].shift({timename:-1}))
            consec = shifted * (cond_dr[model][dr])
            aux = ys_e_cut[model].where(consec).count(axis=0)
            n_cevents[model][dr] = aux.where(aux!=0).copy()


for model in models.keys():
    # crear figura y axes
    fig1, ax = juli_functions.make_figure(['Number' if how=='n' else '%' if how=='%' else 'wrong "how" selection'][0]+' of consecutive afternoon precip events \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(13,5), general_fontsize=9)


    # Si son muchos plots
    ax.set_visible(False)
    gs = gridspec.GridSpec(1,3, wspace=0, hspace=0.2) #si quiero multiples subplos
    
    for m, dr in enumerate(['low', 'mid', 'high']):
        # Poner en loop si son muchos plot
        ax1 = plt.subplot(gs[m], projection = proj)
        
        plot = [n_cevents[model][dr] if how=='n' else n_cevents[model][dr]/n_events[model][dr]*100 ][0]

        if degrade:
            CS1 = juli_functions.plot_pcolormesh(ax1, plot, lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                                 clevs, barra, numbering[m]+' '+dr+' dynamic regime', titlesize = 12, coastwidth = 1, countrywidth = 1)
        else:
            CS1 = juli_functions.plot_pcolormesh(ax1, plot, lon[model], lat[model], lonproj[model], latproj[model], proj,
                                                 clevs, barra, numbering[m]+' '+dr+' dynamic regime', titlesize = 12, coastwidth = 1, countrywidth = 1)
    # add colorbar
    #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
    juli_functions.add_colorbar(fig1, CS1, 'max', '', cbaxes= [0.9, 0.2, 0.01, 0.6]) #[*left*, *bottom*, *width*,  *height*]
    
    # save 
    juli_functions.savefig(fig1, images_path, 'n_consecutive_events_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+'_regimes'+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')



#%% ####### -----------------------  Number of afternoon P events with previous-day precipitation
how = '%' # 'n' for absolute value, '%' for percentage of events 

#creo data_day de pre
if degrade: data_day_cut_dg=dict()
    
for model in models.keys():
    # primero muevo el tiempo de pre a LST  (aprox, tomo UTC-3 para todo)
    data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] - 3*3600000000000
    
    data_day[model] = dict()
    data_day[model]['pre'] = dict()
    data_day[model]['pre'][''] = data[model]['pre'][''].loc[{timename: slice(start_date,end_date)}].resample({timename:'D'}).mean().compute()*mult_hd['pre'][model][0]
    
    data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] + 3*3600000000000

    data_day_cut[model]['pre'] = data_day[model]['pre'][''][season_sel(daily_time_months)].loc[{timename:slice(delta_period[0], delta_period[1])}]
    
    if degrade:
        res_lat = [None if len(ys_e_cut[model][lat_name])%degrade_n==0 else len(ys_e_cut[model][lat_name])%degrade_n*-1][0]
        res_lon = [None if len(ys_e_cut[model][lon_name])%degrade_n==0 else len(ys_e_cut[model][lon_name])%degrade_n*-1][0]

        data_day_cut_dg[model]=dict()
        data_day_cut_dg[model]['pre'] = xr.concat([data_day_cut[model]['pre'][:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')   


#################### set colorbar.
preclevs = np.arange(0,110,10)
prebarra = matplotlib.cm.get_cmap('Reds').copy() # premade colorbar
prebarra.set_under('white')

preclevs2 = np.arange(200,1100,100)
prebarra2 = matplotlib.cm.get_cmap('summer').copy() # premade colorbar

# Nueva barra
barra = mcolors.ListedColormap([prebarra(int(round(x*prebarra.N/preclevs[-1]))) for x in preclevs[1:]]+[prebarra2(int(round(x*prebarra2.N/preclevs2[-1]))) for x in preclevs2[:]])
#barra.set_over('Navy')
clevs = np.concatenate((preclevs,preclevs2))

if how=='%':
    clevs = np.arange(0,110,10)
    barra = matplotlib.cm.get_cmap('Reds').copy() # premade colorbar
    barra.set_under('white')

# new array with number of events
n_cevents = dict()
for model in models.keys():
    n_cevents[model] = dict()
    
    if degrade:
        shifted = (data_day_cut_dg[model]['pre'].shift({timename:1}) > 1)
        consec = shifted * (~np.isnan(ys_e_cut_dg[model]))
        n_cevents[model]['all'] = (consec.sum(axis=0)).where(consec.sum(axis=0)!=0)
        res_lat = [None if len(ys_e_cut_dg[model][lat_name])%degrade_n==0 else len(ys_e_cut_dg[model][lat_name])%degrade_n*-1][0]
        res_lon = [None if len(ys_e_cut_dg[model][lon_name])%degrade_n==0 else len(ys_e_cut_dg[model][lon_name])%degrade_n*-1][0]
    else:
        shifted = (data_day_cut[model]['pre'].shift({timename:1}) > 1)
        consec = shifted * (~np.isnan(ys_e_cut[model]))
        n_cevents[model]['all'] = (consec.sum(axis=0)).where(consec.sum(axis=0)!=0)


for model in models.keys():
    # crear figura y axes
    fig1, ax = juli_functions.make_figure(['Number' if how=='n' else '%' if how=='%' else 'wrong "how" selection'][0]+' of afternoon P events w/ previous-day P > 1mm \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(5,5), general_fontsize=9)

    
    plot = [n_cevents[model]['all'] if how=='n' else n_cevents[model]['all']/n_events[model]['all']*100 ][0]

    if degrade:
        CS1 = juli_functions.plot_pcolormesh(ax, plot, lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                             clevs, barra, '', titlesize = 12, coastwidth = 1, countrywidth = 1)
    else:
        CS1 = juli_functions.plot_pcolormesh(ax, plot, lon[model], lat[model], lonproj[model], latproj[model], proj,
                                             clevs, barra, '', titlesize = 12, coastwidth = 1, countrywidth = 1)
    
    # add colorbar
    #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
    juli_functions.add_colorbar(fig1, CS1, ['neither' if how=='%' and clevs[-1]==100 else 'max'][0], '', cbaxes= [0.9, 0.2, 0.02, 0.6]) #[*left*, *bottom*, *width*,  *height*]
    
    # save 
    juli_functions.savefig(fig1, images_path, 'n_events_w_previousP_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')



#############################################
# Plot of number of events (pixeled version) SEPARADO POR REGIMENES

# new array with number of events
for model in models.keys():
    
    for dr in ['low', 'mid', 'high']:
        if degrade:
            shifted = (data_day_cut_dg[model]['pre'].shift({timename:1}) ).where(~np.isnan(ys_e_cut_dg_dynreg[model][dr]))
            consec = shifted > 1
            n_cevents[model][dr] = (consec.sum(axis=0)).where(consec.sum(axis=0)!=0)
        else:
            shifted = (data_day_cut[model]['pre'].shift({timename:1}) > 1)
            consec = shifted * cond_dr[model][dr]
            aux = ys_e_cut[model].where(consec).count(axis=0)
            n_cevents[model][dr] = aux.where(aux!=0).copy()


for model in models.keys():
    # crear figura y axes
    fig1, ax = juli_functions.make_figure(['Number' if how=='n' else '%' if how=='%' else 'wrong "how" selection'][0]+' of afternoon P events w/ previous-day P > 1mm \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(13,5), general_fontsize=9)


    # Si son muchos plots
    ax.set_visible(False)
    gs = gridspec.GridSpec(1,3, wspace=0, hspace=0.2) #si quiero multiples subplos
    
    for m, dr in enumerate(['low', 'mid', 'high']):
        # Poner en loop si son muchos plot
        ax1 = plt.subplot(gs[m], projection = proj)
        
        plot = [n_cevents[model][dr] if how=='n' else n_cevents[model][dr]/n_events[model][dr]*100 ][0]

        if degrade:
            CS1 = juli_functions.plot_pcolormesh(ax1, plot, lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                                 clevs, barra, numbering[m]+' '+dr+' dynamic regime', titlesize = 12, coastwidth = 1, countrywidth = 1)
        else:
            CS1 = juli_functions.plot_pcolormesh(ax1, plot, lon[model], lat[model], lonproj[model], latproj[model], proj,
                                                 clevs, barra, numbering[m]+' '+dr+' dynamic regime', titlesize = 12, coastwidth = 1, countrywidth = 1)
    # add colorbar
    #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
    juli_functions.add_colorbar(fig1, CS1, ['neither' if how=='%' and clevs[-1]==100 else 'max'][0], '', cbaxes= [0.9, 0.2, 0.01, 0.6]) #[*left*, *bottom*, *width*,  *height*]
    
    # save 
    juli_functions.savefig(fig1, images_path, 'n_events_w_previousP_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+'_regimes'+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')



#%% ####### -----------------------  Number of afternoon P events with previous-day precipitation excluding events
how = '%' # 'n' for absolute value, '%' for percentage of events 

#creo data_day de pre
if degrade: data_day_cut_dg=dict()
    
for model in models.keys():
    # primero muevo el tiempo de pre a LST  (aprox, tomo UTC-3 para todo)
    data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] - 3*3600000000000
    
    data_day[model] = dict()
    data_day[model]['pre'] = dict()
    data_day[model]['pre'][''] = data[model]['pre'][''].loc[{timename: slice(start_date,end_date)}].resample({timename:'D'}).mean().compute()*mult_hd['pre'][model][0]
    
    data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] + 3*3600000000000

    data_day_cut[model]['pre'] = data_day[model]['pre'][''][season_sel(daily_time_months)].loc[{timename:slice(delta_period[0], delta_period[1])}]
    
    if degrade:
        res_lat = [None if len(ys_e_cut[model][lat_name])%degrade_n==0 else len(ys_e_cut[model][lat_name])%degrade_n*-1][0]
        res_lon = [None if len(ys_e_cut[model][lon_name])%degrade_n==0 else len(ys_e_cut[model][lon_name])%degrade_n*-1][0]

        data_day_cut_dg[model]=dict()
        data_day_cut_dg[model]['pre'] = xr.concat([data_day_cut[model]['pre'][:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')   


#################### set colorbar.
preclevs = np.arange(0,110,10)
prebarra = matplotlib.cm.get_cmap('Reds').copy() # premade colorbar
prebarra.set_under('white')

preclevs2 = np.arange(200,1100,100)
prebarra2 = matplotlib.cm.get_cmap('summer').copy() # premade colorbar

# Nueva barra
barra = mcolors.ListedColormap([prebarra(int(round(x*prebarra.N/preclevs[-1]))) for x in preclevs[1:]]+[prebarra2(int(round(x*prebarra2.N/preclevs2[-1]))) for x in preclevs2[:]])
#barra.set_over('Navy')
clevs = np.concatenate((preclevs,preclevs2))

if how=='%':
    clevs = np.arange(0,110,10)
    barra = matplotlib.cm.get_cmap('Reds').copy() # premade colorbar
    barra.set_under('white')

# new array with number of events
n_cevents = dict()
for model in models.keys():
    n_cevents[model] = dict()
    
    if degrade:
        shifted = (data_day_cut_dg[model]['pre'].shift({timename:1}) > 1)
        consec = shifted * (~np.isnan(ys_e_cut_dg[model])) * (np.isnan(ys_e_cut_dg[model].shift({timename:1})) )
        n_cevents[model]['all'] = (consec.sum(axis=0)).where(consec.sum(axis=0)!=0)
        res_lat = [None if len(ys_e_cut_dg[model][lat_name])%degrade_n==0 else len(ys_e_cut_dg[model][lat_name])%degrade_n*-1][0]
        res_lon = [None if len(ys_e_cut_dg[model][lon_name])%degrade_n==0 else len(ys_e_cut_dg[model][lon_name])%degrade_n*-1][0]
    else:
        shifted = (data_day_cut[model]['pre'].shift({timename:1}) > 1)
        consec = shifted * (~np.isnan(ys_e_cut[model]))
        n_cevents[model]['all'] = (consec.sum(axis=0)).where(consec.sum(axis=0)!=0)


for model in models.keys():
    # crear figura y axes
    fig1, ax = juli_functions.make_figure(['Number' if how=='n' else '%' if how=='%' else 'wrong "how" selection'][0]+' of afternoon P events w/ previous-day non-ev P > 1mm \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(5,5), general_fontsize=9)

    
    plot = [n_cevents[model]['all'] if how=='n' else n_cevents[model]['all']/n_events[model]['all']*100 ][0]

    if degrade:
        CS1 = juli_functions.plot_pcolormesh(ax, plot, lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                             clevs, barra, '', titlesize = 12, coastwidth = 1, countrywidth = 1)
    else:
        CS1 = juli_functions.plot_pcolormesh(ax, plot, lon[model], lat[model], lonproj[model], latproj[model], proj,
                                             clevs, barra, '', titlesize = 12, coastwidth = 1, countrywidth = 1)
    
    # add colorbar
    #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
    juli_functions.add_colorbar(fig1, CS1, ['neither' if how=='%' and clevs[-1]==100 else 'max'][0], '', cbaxes= [0.9, 0.2, 0.02, 0.6]) #[*left*, *bottom*, *width*,  *height*]
    
    # save 
    juli_functions.savefig(fig1, images_path, 'n_events_w_previousP_and_no_event_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')



#############################################
# Plot of number of events (pixeled version) SEPARADO POR REGIMENES

# new array with number of events
for model in models.keys():
    
    for dr in ['low', 'mid', 'high']:
        if degrade:
            shifted = (data_day_cut_dg[model]['pre'].shift({timename:1}) ).where(~np.isnan(ys_e_cut_dg_dynreg[model][dr]))
            consec = shifted > 1
            n_cevents[model][dr] = (consec.sum(axis=0)).where(consec.sum(axis=0)!=0)
        else:
            shifted = (data_day_cut[model]['pre'].shift({timename:1}) > 1)
            consec = shifted * cond_dr[model][dr]
            aux = ys_e_cut[model].where(consec).count(axis=0)
            n_cevents[model][dr] = aux.where(aux!=0).copy()


for model in models.keys():
    # crear figura y axes
    fig1, ax = juli_functions.make_figure(['Number' if how=='n' else '%' if how=='%' else 'wrong "how" selection'][0]+' of afternoon P events w/ previous-day P > 1mm \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(13,5), general_fontsize=9)


    # Si son muchos plots
    ax.set_visible(False)
    gs = gridspec.GridSpec(1,3, wspace=0, hspace=0.2) #si quiero multiples subplos
    
    for m, dr in enumerate(['low', 'mid', 'high']):
        # Poner en loop si son muchos plot
        ax1 = plt.subplot(gs[m], projection = proj)
        
        plot = [n_cevents[model][dr] if how=='n' else n_cevents[model][dr]/n_events[model][dr]*100 ][0]

        if degrade:
            CS1 = juli_functions.plot_pcolormesh(ax1, plot, lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                                 clevs, barra, numbering[m]+' '+dr+' dynamic regime', titlesize = 12, coastwidth = 1, countrywidth = 1)
        else:
            CS1 = juli_functions.plot_pcolormesh(ax1, plot, lon[model], lat[model], lonproj[model], latproj[model], proj,
                                                 clevs, barra, numbering[m]+' '+dr+' dynamic regime', titlesize = 12, coastwidth = 1, countrywidth = 1)
    # add colorbar
    #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
    juli_functions.add_colorbar(fig1, CS1, ['neither' if how=='%' and clevs[-1]==100 else 'max'][0], '', cbaxes= [0.9, 0.2, 0.01, 0.6]) #[*left*, *bottom*, *width*,  *height*]
    
    # save 
    juli_functions.savefig(fig1, images_path, 'n_events_w_previousP_and_no_event_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+'_regimes'+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')


#%% ####### -----------------------  Terrestrial and atmospheric coupling index (usa delta_period para recortar el periodo)

lagged = 0
anom_type = 'month'

ci = ['sm1-evapot', 'evapot-pre', 'sm1-pre']

ci_vars = set([x.split('-')[0] for x in ci]+[x.split('-')[1] for x in ci])

data_daily = dict() 

# Calculo climatologias mensuales para quitar la estacionalidad y hago anomalias diarias
month_mean = dict()
data_daily_anoms = dict()

            
print(' Calculando anomalías')
for model in models.keys():
    month_mean[model] = dict()
    data_daily_anoms[model] = dict()
    data_daily[model] = dict()

    timename = 'time'
    if model=='JRA-55': timename = 'initial_time0_hours'
    
    for var in ci_vars:

        lonname = [coord for coord in set(data[model][var][''].coords.keys()) if "lon" in coord][0]
        latname = [coord for coord in set(data[model][var][''].coords.keys()) if "lat" in coord][0]
        timename = 'time'
        if model=='JRA-55': timename='initial_time0_hours'
        
        # muevo el tiempo a LST
        data[model][var][''].coords[timename] = data[model][var][''][timename] - 3*3600000000000

        data_daily[model][var] = data[model][var][''].loc[{timename:slice(delta_period[0][:-1]+str(int(delta_period[0][-2:])+1),delta_period[1][:-2]+str(int(delta_period[1][-2:])-1))}].resample({timename:'D'}).mean(dim=timename).compute()*mult_hd[var][model][0]+mult_hd[var][model][1]
        
        # vuelvo el tiempo a como era originalmente
        data[model][var][''].coords[timename] = data[model][var][''][timename] + 3*3600000000000
        
            
        if anom_type == 'month':
            month_mean[model][var] = data_daily[model][var].groupby(timename+'.month').mean(dim=timename, skipna=True)
            
            aux = np.empty_like(data_daily[model][var])
            
            for m,mm in enumerate(data_daily[model][var][timename+'.month']):
                aux[m] = month_mean[model][var][mm-1]
            
            data_daily_anoms[model][var] = data_daily[model][var] - aux

        elif anom_type == 'day':
            month_mean[model][var] = data_daily[model][var].groupby(timename+'.dayofyear').mean(dim=timename, skipna=True)
            
            data_daily_anoms[model][var] = data_daily[model][var].groupby(timename+'.dayofyear') - month_mean[model][var]

print(' ##### Calculando TCI / ACI #####')
# modos para calcular el TCI: 'total' es manual de toda la serie (con las anomalias como fueron definidas al ppio de esta seccion)
# month o season calculan por mes o estacion con el paquete coupling metrics

tci_mode = 'total' 
tci = dict()
pval = dict()

seasons_n = {'ONDJFM': [10,11,12,1,2,3],
             'DJF': [12,1,2]}

ci_seasons = ['ONDJFM']

for model in models.keys():
    print(' #### '+model+' ####')
    tci[model] = dict()
    pval[model] = dict()

    timename='time'
    if model =='JRA-55': timename='initial_time0_hours'
    
    for var in ci:
        print(' ... '+var+' ... ')
        tci[model][var] = dict()
        pval[model][var] = dict()
        
        for seas in ci_seasons:
            print(seas)
            tci[model][var][seas] = dict()
            pval[model][var][seas] = dict()
            
            try:
                if tci_mode in ['month','season']:
                    print('no configurado para tci_mode='+tci_mode)
                
                elif tci_mode == 'total':
                    
                    # para todos los dias de la estacion (no solo los eventos)
                    var1 = data_daily_anoms[model][var.split('-')[0]].where(juli_functions.is_month(data_daily_anoms[model][var.split('-')[0]][timename+'.month'], seasons_n[seas]), drop=True)
                    var2 = data_daily_anoms[model][var.split('-')[1]].where(juli_functions.is_month(data_daily_anoms[model][var.split('-')[1]][timename+'.month'], seasons_n[seas]), drop=True)
                      
                    # calculo version vieja, a mano
                    # covar = ((var1 - var1.mean(dim=timename)) * (var2 - var2.mean(dim=timename))).sum(dim=timename) / var1.count(dim=timename)
                    # tci[model][var][seas]['all'] = covar / var1.std(dim=timename)
                    
                    # pval[model][var][seas]['all'] = xr.apply_ufunc(scipy.stats.pearsonr, var1, var2, input_core_dims=[[timename], [timename]],
                    #                                             output_core_dims=[[],[]], vectorize=True)[1]
                    
                    # calculo version nueva, usando linregress (dan lo mismo)
                    slope, intercept, r, p, se = xr.apply_ufunc(scipy.stats.linregress, var1, var2, input_core_dims=[[timename], [timename]],
                                                                 output_core_dims=[[],[],[],[],[]], vectorize=True)
                    
                    tci[model][var][seas]['all'] = var1.std(dim=timename) * slope
                    
                    pval[model][var][seas]['all'] = p
                    
                else: print('tci_mode defined is not recognized')
            except KeyError: print('not available')


#%% Plots TCI
clevs = np.arange(-1.2, 1.4, 0.2)
p_value = 0.01 # menor que esto es significativo

ev_names = {'all': ''}

images_path_ci = '/home/julian.giles/datos/CTL/Images/coupling_index/'

for seas in ci_seasons:
    for model in models.keys():
        
        for var in ci:
            var0=var.split('-')[0]
            var00=var.split('-')[1]
            
            lonname = [coord for coord in set(data[model][var.split('-')[0]][''].coords.keys()) if "lon" in coord][0]
            latname = [coord for coord in set(data[model][var.split('-')[0]][''].coords.keys()) if "lat" in coord][0]

            if len(ev_names)==4:
                fig1, ax = juli_functions.make_figure('Coupling Index '+var, figsize=(6,7), general_fontsize=9)
                cbaxes = [0.2,0.05, 0.6, 0.01]#[*left*, *bottom*, *width*,  *height*]
            else:
                fig1, ax = juli_functions.make_figure('Coupling Index '+var, figsize=(8,4), general_fontsize=9)
                cbaxes = [0.7,0.1, 0.01, 0.8]
            
            ax.set_visible(False)
            if len(ev_names)==4:
                gs = gridspec.GridSpec(2,2) # multiples subplots
            else:
                gs = gridspec.GridSpec(1,len(ev_names))

            for nn,ev in enumerate(ev_names):
                
                # Poner en loop si son muchos plot
                ax1 = plt.subplot(gs[nn], projection = proj)
                
                barra= juli_functions.barra_zerocenter(clevs ,colormap=matplotlib.cm.get_cmap('RdYlBu_r'), no_extreme_colors=True)
                barra.set_over(matplotlib.cm.get_cmap('RdYlBu_r')(256))
                barra.set_under(matplotlib.cm.get_cmap('RdYlBu_r')(0))

                pll = (False, False)
                if len(ev_names)==4:
                    if nn==0:
                        pll = (False, True)
                    elif nn==2:
                        pll = (True, True)
                    elif nn==3:
                        pll = (True, False)

                else:
                    if nn==0:
                        pll = (True, True)
                    elif nn>=1:
                        pll = (True, False)
                
                ax1.add_feature(cartopy.feature.OCEAN, zorder=1, facecolor='white')
                
                # hago cada plot
                CS1 = juli_functions.plot_pcolormesh(ax1, tci[model][var][seas][ev],
                                                     data_daily[model][var0][lonname], data_daily[model][var0][latname],
                                                     data_daily[model][var0][lonname], data_daily[model][var0][latname],
                                                     proj, clevs, barra, model+' '+seas+' '+delta_period[0]+'-'+delta_period[1]+' '+ev_names[ev], titlesize = 12, coastwidth = 1, countrywidth = 1, 
                                                     countryalpha=0.5, printlonslats=pll)
                # para hatchear la zona no significativa
#                CS2 = ax1.pcolor(data_daily_anoms[model][var0][lonname], data_daily_anoms[model][var0][latname],
#                               data_daily_anoms[model][var][seas][ev][:,:,1].where(data_daily_anoms[model][var][seas][ev][:,:,1]>p_value), alpha=0., cmap= 'Greys', hatch='x')

                # para sombrear la zona no significativa
                CS2 = ax1.pcolormesh(data_daily_anoms[model][var0][lonname], data_daily_anoms[model][var0][latname],
                               tci[model][var][seas][ev].where(pval[model][var][seas][ev]>p_value)*0, alpha=1, cmap= 'Greys')
                                                
                
                # add colorbar
                if len(ev_names)==4 and nn==1:
                    #fig1.subplots_adjust( left=0.3,bottom=0.2)# left=0.23
                    juli_functions.add_colorbar(fig1, CS1, 'both', units_labels[var00], cbaxes= cbaxes, orientation='horizontal', labelpad=0) #[*left*, *bottom*, *width*,  *height*]
                
                else:
                    #fig1.subplots_adjust( left=0.3,bottom=0.2)# left=0.23
                    juli_functions.add_colorbar(fig1, CS1, 'both', units_labels[var00], cbaxes= cbaxes, orientation='vertical', labelpad=0) #[*left*, *bottom*, *width*,  *height*]
                
            # save 
            juli_functions.savefig(fig1, images_path_ci+seas+'/', model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas+'_'+var+'_ci_'+tci_mode+'.png', bbox_inches='tight')



        if 'sm1-evapot' in ci and 'evapot-pre' in ci:
            # var0=var.split('-')[0]
            # var00=var.split('-')[1]
            
            # lonname = [coord for coord in set(data[model][var.split('-')[0]][''].coords.keys()) if "lon" in coord][0]
            # latname = [coord for coord in set(data[model][var.split('-')[0]][''].coords.keys()) if "lat" in coord][0]

            if len(ev_names)==4:
                fig1, ax = juli_functions.make_figure('Total Feedback (TCIxACI) sm1-evapot-pre', figsize=(6,7), general_fontsize=9)
                cbaxes = [0.2,0.05, 0.6, 0.01]#[*left*, *bottom*, *width*,  *height*]
            else:
                fig1, ax = juli_functions.make_figure('Total Feedback (TCIxACI) sm1-evapot-pre', figsize=(8,4), general_fontsize=9)
                cbaxes = [0.7,0.1, 0.01, 0.8]
            
            ax.set_visible(False)
            if len(ev_names)==4:
                gs = gridspec.GridSpec(2,2) # multiples subplots
            else:
                gs = gridspec.GridSpec(1,len(ev_names))

            for nn,ev in enumerate(ev_names):
                
                # Poner en loop si son muchos plot
                ax1 = plt.subplot(gs[nn], projection = proj)
                
                barra= juli_functions.barra_zerocenter(clevs ,colormap=matplotlib.cm.get_cmap('RdYlBu_r'), no_extreme_colors=True)
                barra.set_over(matplotlib.cm.get_cmap('RdYlBu_r')(256))
                barra.set_under(matplotlib.cm.get_cmap('RdYlBu_r')(0))

                pll = (False, False)
                if len(ev_names)==4:
                    if nn==0:
                        pll = (False, True)
                    elif nn==2:
                        pll = (True, True)
                    elif nn==3:
                        pll = (True, False)

                else:
                    if nn==0:
                        pll = (True, True)
                    elif nn>=1:
                        pll = (True, False)
                
                ax1.add_feature(cartopy.feature.OCEAN, zorder=1, facecolor='white')
                
                # hago cada plot
                CS1 = juli_functions.plot_pcolormesh(ax1, tci[model]['sm1-evapot'][seas][ev]*tci[model]['evapot-pre'][seas][ev],
                                                     data_daily[model][var0][lonname], data_daily[model][var0][latname],
                                                     data_daily[model][var0][lonname], data_daily[model][var0][latname],
                                                     proj, clevs, barra, model+' '+seas+' '+delta_period[0]+'-'+delta_period[1]+' '+ev_names[ev], titlesize = 12, coastwidth = 1, countrywidth = 1, 
                                                     countryalpha=0.5, printlonslats=pll)

                # para sombrear la zona no significativa
                CS2 = ax1.pcolormesh(data_daily_anoms[model][var0][lonname], data_daily_anoms[model][var0][latname],
                                tci[model]['sm1-evapot'][seas][ev].where((pval[model]['sm1-evapot'][seas][ev]>p_value)*(pval[model]['evapot-pre'][seas][ev]>p_value))*0, alpha=1, cmap= 'Greys')
                                                
                # para hatchear la zona donde TCI o ACI son negativos
                CS2 = ax1.pcolor(data_daily_anoms[model][var0][lonname], data_daily_anoms[model][var0][latname],
                              data_daily_anoms[model][var0][1].where((tci[model]['sm1-evapot'][seas][ev]<0)+(tci[model]['evapot-pre'][seas][ev]<0)), alpha=0., cmap= 'Greys', hatch='xx', shading='auto')
                
                # add colorbar
                if len(ev_names)==4 and nn==1:
                    #fig1.subplots_adjust( left=0.3,bottom=0.2)# left=0.23
                    juli_functions.add_colorbar(fig1, CS1, 'both', '$\\mathregular{mm^{2}⋅day^{-2}}$', cbaxes= cbaxes, orientation='horizontal', labelpad=0) #[*left*, *bottom*, *width*,  *height*]
                
                else:
                    #fig1.subplots_adjust( left=0.3,bottom=0.2)# left=0.23
                    juli_functions.add_colorbar(fig1, CS1, 'both', '$\\mathregular{mm^{2}⋅day^{-2}}$', cbaxes= cbaxes, orientation='vertical', labelpad=0) #[*left*, *bottom*, *width*,  *height*]
                
            # save 
            juli_functions.savefig(fig1, images_path_ci+seas+'/', model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas+'_'+'sm1-evapot-pre'+'_tf_'+tci_mode+'.png', bbox_inches='tight')


#%% ####### -----------------------  Temporal distribution of events (events by month)
if 'SALLJ' in regions.keys(): regions.pop('SALLJ')
if 'JEXIT' in regions.keys(): regions.pop('JEXIT')

# new array with number of events
n_events_month = dict()
for model in models.keys():
    n_events_month[model] = dict()
    
    for dr in ['all', 'low', 'mid', 'high']:
        n_events_month[model][dr] = dict()

        for reg in regions.keys():        
            if dr=='all':
                n_events_month[model]['all'][reg] = ys_e_cut[model].loc[{lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].groupby(timename+'.month').count(dim=(timename,lat_name, lon_name))
            else:
                n_events_month[model][dr][reg] = ys_e_cut[model].loc[{lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].where(cond_dr[model][dr]).groupby(timename+'.month').count(dim=(timename,lat_name, lon_name)).compute()


for model in models.keys():
    # crear figura y axes
    fig1, ax = juli_functions.make_figure('Number of afternoon precip events by month \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(6,5), general_fontsize=6)
    

    # Si son muchos plots
    ax.set_visible(False)
    gs = gridspec.GridSpec(4,4, wspace=0.3, hspace=0.2) #si quiero multiples subplos
    
    n=0
    for m, dr in enumerate(['all', 'low', 'mid', 'high']):
        for reg in regions.keys(): 
            # Poner en loop si son muchos plot
            ax1 = plt.subplot(gs[n])
            
            juli_functions.plot_line(ax1, [1,2,3,4,5,6], np.concatenate([n_events_month[model][dr][reg][3:6].values, n_events_month[model][dr][reg][0:3].values]), None, None, None, None, None, title=reg, xticksrot = 0)
            
            plt.xticks([1,2,3,4,5,6],['O','N','D','J','F','M'])
            
            n=n+1

    # save 
    juli_functions.savefig(fig1, images_path, 'n_events_bymonth_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')



#%% ####### -----------------------  Mean trend of P and SM around event (mean daily P and SM)

if 'SALLJ' in regions.keys(): regions.pop('SALLJ')
if 'JEXIT' in regions.keys(): regions.pop('JEXIT')

dr_to_plot = ['all'] # si quiero los regimenes dinamicos ['all', 'low', 'mid', 'high'], primero tengo que correr cond_dr[model][dr]


#creo data_day de pre y sm    
for model in models.keys():
    for var in ['pre', 'sm1']:

        timename='time'
        if model=='JRA-55': timename='initial_time0_hours'
    
        lat_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lat" in coord][0]
        lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]

        # primero muevo el tiempo de pre a LST  (aprox, tomo UTC-4 para todo)
        data[model][var][''].coords[timename] = data[model][var][''][timename] - 4*3600000000000
        
        data_day[model][var] = dict()
        if var == 'sm1':
            data_day[model][var][''] = data[model][var][''][data[model][var][''][timename+'.hour']==6].loc[{timename: slice(start_date,end_date)}].compute()*mult_hd[var][model][0]
            #corro a la hora cero para que se seleccionen bien los dias luego
            data_day[model][var][''].coords[timename] = data_day[model][var][''][timename] - 6*3600000000000
        else:
            data_day[model][var][''] = data[model][var][''].loc[{timename: slice(start_date,end_date)}].resample({timename:'D'}).mean().compute()*mult_hd[var][model][0]
        
        data[model][var][''].coords[timename] = data[model][var][''][timename] + 4*3600000000000
    
        #data_day_cut[model][var] = data_day[model][var][''][season_sel(daily_time_months)].loc[{timename:slice(delta_period[0], delta_period[1])}]
    


# new array with number of events
pre_trend = dict()
sm_trend = dict()
sm_trend_box = dict() # data for the pixels surrounding the central
for model in models.keys():
    pre_trend[model] = dict()
    sm_trend[model] = dict()
    sm_trend_box[model] = dict()
    
    for dr in dr_to_plot:
        pre_trend[model][dr] = dict()
        sm_trend[model][dr] = dict()
        sm_trend_box[model][dr] = dict()

        for reg in regions.keys():        
            if dr=='all':
                pre_trend[model]['all'][reg] = list()
                sm_trend[model]['all'][reg] = list()
                sm_trend_box[model]['all'][reg] = list()

                for shift in np.arange(-15, 16):
                    shiftedcond = ~np.isnan(ys_e_cut[model]).compute()
                    shiftedcond.coords[timename] = shiftedcond[timename] + shift*24*3600000000000
                    pre_trend[model]['all'][reg].append( float(data_day[model]['pre'][''].where(shiftedcond).loc[{timename:slice(delta_period[0], delta_period[1]), lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].mean(dim=(timename, lat_name, lon_name))) )
                    sm_trend[model]['all'][reg].append( float(data_day[model]['sm1'][''].where(shiftedcond).loc[{timename:slice(delta_period[0], delta_period[1]), lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].mean(dim=(timename, lat_name, lon_name))) )
                    
                    # repeat for SM in surrounding pixels
                    shiftedcond_box = xr.concat([shiftedcond.shift({lat_name:ii, lon_name:jj}, fill_value=False) for ii,jj in list(itertools.product(range(-box_size2[0],box_size2[0]+1), repeat=2)) if (ii,jj)!=(0,0)], dim= 'shift').sum(dim='shift')
                    sm_trend_box[model]['all'][reg].append( float(data_day[model]['sm1'][''].where(shiftedcond_box).loc[{timename:slice(delta_period[0], delta_period[1]), lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].mean(dim=(timename, lat_name, lon_name))) )
                
            else:
                pre_trend[model][dr][reg] = list()
                sm_trend[model][dr][reg] = list()

                for shift in np.arange(-15, 16):
                    shiftedcond = ~np.isnan(ys_e_cut[model].where(cond_dr[model][dr])).compute()
                    shiftedcond.coords[timename] = shiftedcond[timename] + shift*24*3600000000000
                    pre_trend[model][dr][reg].append( float(data_day[model]['pre'][''].where(shiftedcond).loc[{timename:slice(delta_period[0], delta_period[1]), lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].mean(dim=(timename, lat_name, lon_name))) )
                    sm_trend[model][dr][reg].append( float(data_day[model]['sm1'][''].where(shiftedcond).loc[{timename:slice(delta_period[0], delta_period[1]), lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].mean(dim=(timename, lat_name, lon_name))) )



rows= [len(dr_to_plot) if len(dr_to_plot)>1 else len(regions.keys())][0] #si tomo los regimenes se hace hoizontal, sino vertical
cols= [len(regions.keys()) if len(dr_to_plot)>1 else 1][0]

figsize= [(2,5) if cols==1 else (5,5)][0]

barcolor= 'steelblue'
centerbarcolor= '#f29f37' #'orange'
linecolor= 'firebrick'# 'red'

for model in models.keys():
    with plt.rc_context({'axes.edgecolor':'#212121', 'xtick.color':'#212121', 'ytick.color':'#212121'}):
        
        # crear figura y axes
        fig1, ax = juli_functions.make_figure('Daily mean P and SM around events \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=figsize, general_fontsize=6)
        
    
        # Si son muchos plots
        ax.set_visible(False)
        gs = gridspec.GridSpec(rows,cols, wspace=0.3, hspace=0.2) #si quiero multiples subplos
        
        n=0
        
        ylims=[ [0,16], [0,14], [0,10] ]
        yticks=[ np.arange(0,16,5), np.arange(0,14,2.5), np.arange(0,12,2) ]
        
        for m, dr in enumerate(dr_to_plot):
            for l, reg in enumerate(regions.keys()): 
                # Poner en loop si son muchos plot
                ax1 = plt.subplot(gs[n])
                
                barax = plt.bar(np.arange(-15,16), pre_trend[model][dr][reg], color=barcolor)
                plt.bar(0, pre_trend[model][dr][reg][15], color=centerbarcolor)
                plt.ylim(ylims[l])
                plt.yticks(yticks[l])
                
                if cols==1 and l<rows-1 or cols>1 and m==len(dr_to_plot)-1: ax1.axes.xaxis.set_ticklabels([]) #plt.xticks([])
                
                lineax = ax1.twinx()
                # juli_functions.plot_line(lineax, np.arange(-15,16), sm_trend[model][dr][reg], None, None, None, None, None, title=[reg if n<3 and cols>1 else ''][0], xticksrot = 0, color='red')
                
                # juli_functions.plot_line(lineax, np.arange(-15,16), [np.array(sm_trend[model][dr][reg]).mean()]*31, None, None, None, None, None, title=[reg if n<3 else ''][0], xticksrot = 0, color='red', ls='dotted')
                # juli_functions.plot_line(lineax, np.arange(-15,16), [np.array(sm_trend[model][dr][reg][0:15]).mean()]*15+[np.nan]*16, None, None, None, None, None, title=[reg if n<3 else ''][0], xticksrot = 0, color='red', ls='dotted')
                
                lineax.plot(np.arange(-15,16), sm_trend[model][dr][reg], color=linecolor)
                
                lineax.plot(np.arange(-15,16), [np.array(sm_trend[model][dr][reg]).mean()]*31, color=linecolor, ls='dotted')
                lineax.plot(np.arange(-15,16), [np.array(sm_trend[model][dr][reg][0:15]).mean()]*15+[np.nan]*16, color=linecolor, ls='dotted')
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                                     
                # lo mismo para los puntos de alrededor
                lineax.plot(np.arange(-15,16), sm_trend_box[model][dr][reg], color='#00441b')
                
                lineax.plot(np.arange(-15,16), [np.array(sm_trend_box[model][dr][reg]).mean()]*31, color='#00441b', ls='dotted')
                lineax.plot(np.arange(-15,16), [np.array(sm_trend_box[model][dr][reg][0:15]).mean()]*15+[np.nan]*16, color='#00441b', ls='dotted')

                #set subtitles
                lineax.set_title([reg if n<3 and cols>1 else ''][0])
                
    
                n=n+1
    
        # save 
        juli_functions.savefig(fig1, images_path, 'mean_p_sm_trends_events_supplementary_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+'.png')



#%% ####### -----------------------  Interannual daily mean SM and number of events along the season 


for model in models.keys():
    # crear figura y axes
    fig1, ax = juli_functions.make_figure('Interannual daily mean SM and number of events in the season \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(5,4), general_fontsize=6)
    

    # Si son muchos plots
    ax.set_visible(False)
    gs = gridspec.GridSpec(4,3, wspace=0.5, hspace=0.3) #si quiero multiples subplos
    
    n=0
    for reg in regions.keys(): 
        # Poner en loop si son muchos plot
        ax1 = plt.subplot(gs[n])
        
        sm_mean = data_mor[model]['sm1'].loc[{timename:slice(delta_period[0], delta_period[1]), lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].groupby(timename+'.dayofyear').mean(dim=(timename, lat_name, lon_name))[:-1]
        sm_std = data_mor[model]['sm1'].loc[{timename:slice(delta_period[0], delta_period[1]), lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].groupby(timename+'.dayofyear').std(dim=(timename, lat_name, lon_name))[:-1]

        #si quiero la SM absoluta en lugar de las anomalias
        # sm_mean = data_day[model]['sm1'][''].loc[{timename:slice(delta_period[0], delta_period[1]), lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].groupby(timename+'.dayofyear').mean(dim=(timename, lat_name, lon_name))[:-1]
        # sm_std = data_day[model]['sm1'][''].loc[{timename:slice(delta_period[0], delta_period[1]), lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].groupby(timename+'.dayofyear').std(dim=(timename, lat_name, lon_name))[:-1]
        
        pre_mean = (~np.isnan(ys_e[model])).loc[{timename:slice(delta_period[0], delta_period[1]), lat_name: slice(regions[reg][2], regions[reg][3]), lon_name: slice(regions[reg][0], regions[reg][1])}].groupby(timename+'.dayofyear').sum(dim=(timename, lat_name, lon_name))[:-1]
        
        pre_mean_ondjfm = np.concatenate((pre_mean.where(sm_mean.dayofyear>273, drop=True), pre_mean.where(sm_mean.dayofyear<=90, drop=True)))
        sm_mean_ondjfm = np.concatenate((sm_mean.where(sm_mean.dayofyear>273, drop=True), sm_mean.where(sm_mean.dayofyear<=90, drop=True)))
        sm_std_ondjfm = np.concatenate((sm_std.where(sm_mean.dayofyear>273, drop=True), sm_std.where(sm_mean.dayofyear<=90, drop=True)))
        
        barax = plt.bar(np.arange(1,90+92+1), pre_mean_ondjfm, color='steelblue')
        lineax = ax1.twinx()
        plt.fill_between(np.arange(90+92), sm_mean_ondjfm-sm_std_ondjfm, sm_mean_ondjfm+sm_std_ondjfm, color='red', alpha=0.2)
        juli_functions.plot_line(lineax, np.arange(90+92), sm_mean_ondjfm, None, None, None, None, None, title=[reg if n<3 else ''][0], xticksrot = 0, color='red', lw=0.5)
        
        n=n+1

    # save 
    juli_functions.savefig(fig1, images_path, 'mean_daily_p_sm_season'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')


#%% ####### -----------------------  Robustez de las distintas formas de calcular los deltas (coincidencia de las distintas metodologías en un mismo dataset)

# hace falta correr la celda  RECORTO LA ESTACION DESEADA
# Se calcula para degrade=True o degrade=False usando las mismas variables, no diferencia en el calculo. Es decir, no se pueden hacer ambos a la vez.

# load deltas from different methods
version='16' # version del calculo
temp_path2 = '/home/julian.giles/datos/temp/heterog_sm_pp/'
methods = ['5', '7.1', '7.2', '7.3'] #metodos a considerar

thresholds = [1,99]

deltas_e_ys = dict()
deltas_e_yt = dict()
deltas_e_yh = dict()

deltas_ys = dict()
deltas_yt = dict()
deltas_yh = dict()

deltas_e_ys_perc = dict()
deltas_e_yt_perc = dict()
deltas_e_yh_perc = dict()

for model in models.keys():
    deltas_e_ys[model] = dict()
    deltas_e_yt[model] = dict()
    deltas_e_yh[model] = dict()
    
    deltas_ys[model] = dict()
    deltas_yt[model] = dict()
    deltas_yh[model] = dict()

    deltas_e_ys_perc[model] = dict()
    deltas_e_yt_perc[model] = dict()
    deltas_e_yh_perc[model] = dict()
    
    for method in methods:
        deltas_e_ys[model][method] = dict()
        deltas_e_yt[model][method] = dict()
        deltas_e_yh[model][method] = dict()
        
        deltas_ys[model][method] = dict()
        deltas_yt[model][method] = dict()
        deltas_yh[model][method] = dict()

        deltas_e_ys_perc[model][method] = dict()
        deltas_e_yt_perc[model][method] = dict()
        deltas_e_yh_perc[model][method] = dict()

        for dr in ['', 'low', 'mid', 'high']:
            
            if degrade:
                deltas_e_ys[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_e_ys_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+['_'+dr if dr != '' else ''][0]+'.nc')
                deltas_e_yt[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_e_yt_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+['_'+dr if dr != '' else ''][0]+'.nc')
                deltas_e_yh[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_e_yh_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+['_'+dr if dr != '' else ''][0]+'.nc')
                deltas_ys[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_ys_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+['_'+dr if dr != '' else ''][0]+'.nc')
                deltas_yt[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_yt_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+['_'+dr if dr != '' else ''][0]+'.nc')
                deltas_yh[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_yh_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+'_dg'+str(degrade_n)+['_'+dr if dr != '' else ''][0]+'.nc')            
            
            else:
                deltas_e_ys[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_e_ys_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+['_'+dr if dr != '' else ''][0]+'.nc')
                deltas_e_yt[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_e_yt_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+['_'+dr if dr != '' else ''][0]+'.nc')
                deltas_e_yh[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_e_yh_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+['_'+dr if dr != '' else ''][0]+'.nc')
                deltas_ys[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_ys_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+['_'+dr if dr != '' else ''][0]+'.nc')
                deltas_yt[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_yt_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+['_'+dr if dr != '' else ''][0]+'.nc')
                deltas_yh[model][method][dr] = xr.open_dataarray(temp_path2+'/run'+method+'_v'+version+'/delta_yh_'+['dynreg_' if dr != '' else ''][0]+model+'_'+delta_period[0]+'-'+delta_period[1]+['_'+dr if dr != '' else ''][0]+'.nc')            


            deltas_e_ys_perc[model][method][dr] = xr.apply_ufunc(stats.percentileofscore, deltas_ys[model][method][dr], deltas_e_ys[model][method][dr], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk':True})
            deltas_e_yt_perc[model][method][dr] = xr.apply_ufunc(stats.percentileofscore, deltas_yt[model][method][dr], deltas_e_yt[model][method][dr], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk':True})
            deltas_e_yh_perc[model][method][dr] = xr.apply_ufunc(stats.percentileofscore, deltas_yh[model][method][dr], deltas_e_yh[model][method][dr], input_core_dims=[['boots'], []], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk':True})
    
plot_titles = ['Spatial preference: '+r'$\mathrm{\delta_e(Y^s)}$',
               'Temporal preference: '+r'$\mathrm{\delta_e(Y^t)}$',
               'Heterogeneity preference: '+r'$\mathrm{\delta_e(Y^h)}$']

# si estoy recargando resultados anteriores tengo que volver a calcular esto
if 'ys_e_cut_dg' not in locals() or 'ys_e_cut_dg' not in globals():
    ys_e_cut_dg = dict()


for model in models.keys():
    for dr in ['']:
        
        deltas = [xr.zeros_like(deltas_e_ys_perc[model][method][dr]) + xr.concat([deltas_e_ys_perc[model][method][dr]>thresholds[1] for method in methods], dim='methods').sum(dim='methods') - xr.concat([deltas_e_ys_perc[model][method][dr]<thresholds[0] for method in methods], dim='methods').sum(dim='methods'),
                  xr.zeros_like(deltas_e_yt_perc[model][method][dr]) + xr.concat([deltas_e_yt_perc[model][method][dr]>thresholds[1] for method in methods], dim='methods').sum(dim='methods') - xr.concat([deltas_e_yt_perc[model][method][dr]<thresholds[0] for method in methods], dim='methods').sum(dim='methods'),
                  xr.zeros_like(deltas_e_yh_perc[model][method][dr]) + xr.concat([deltas_e_yh_perc[model][method][dr]>thresholds[1] for method in methods], dim='methods').sum(dim='methods') - xr.concat([deltas_e_yh_perc[model][method][dr]<thresholds[0] for method in methods], dim='methods').sum(dim='methods')]
        
        # mascara para los puntos con menos de min_events eventos:
        
        if degrade:
            lat_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lat" in coord][0]
            lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]
            res_lat = [None if len(ys_e_cut[model][lat_name])%degrade_n==0 else len(ys_e_cut[model][lat_name])%degrade_n*-1][0]
            res_lon = [None if len(ys_e_cut[model][lon_name])%degrade_n==0 else len(ys_e_cut[model][lon_name])%degrade_n*-1][0]
            if model not in ys_e_cut_dg.keys():
                ys_e_cut_dg[model] = xr.concat([ys_e_cut[model][:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
            min_ev_mask = ys_e_cut_dg[model].count(axis=0).compute()
    
        else:
            min_ev_mask = ys_e_cut[model].count(axis=0).compute()
        
        
        # crear figura y axes
        fig1, ax = juli_functions.make_figure(model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(13,5), general_fontsize=9)
    
        clevs = np.array([-4.5,-3.5,-2.5,-1.5,-0.5,0,0.5,1.5,2.5,3.5,4.5])       # Esta es la escala que usa Guillod
    
        # barra de colores
        barra= juli_functions.barra_zerocenter(clevs ,colormap=matplotlib.cm.get_cmap('RdBu'), no_extreme_colors=False)
        
        barra = mcolors.ListedColormap([(235/255,235/255,235/255) if x in np.arange(int(barra.N*0.4), int(barra.N*0.6)) else barra(x) for x in np.arange(0, barra.N)])
    
        # Si son muchos plots
        ax.set_visible(False)
        gs = gridspec.GridSpec(1,3, wspace=0, hspace=0.2) #si quiero multiples subplos
        
        for m, delta in enumerate(deltas):
            # Poner en loop si son muchos plot
            ax1 = plt.subplot(gs[m], projection = proj)
            
            # hago cada plot
            if degrade:
                CS1 = juli_functions.plot_pcolormesh(ax1, delta.where(min_ev_mask>=min_events), lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                                     clevs, barra, numbering[m]+' '+plot_titles[m], titlesize = 12, coastwidth = 1, countrywidth = 1)
            else:
                CS1 = juli_functions.plot_pcolormesh(ax1, delta.where(min_ev_mask>=min_events), lon[model], lat[model], lonproj[model], latproj[model], proj,
                                                     clevs, barra, numbering[m]+' '+plot_titles[m], titlesize = 12, coastwidth = 1, countrywidth = 1)
    
        # add colorbar
        #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
        juli_functions.add_colorbar(fig1, CS1, 'neither', '', cbaxes= [0.9, 0.2, 0.01, 0.6], ticks=[-4,-3,-2,-1,0,1,2,3,4]) #[*left*, *bottom*, *width*,  *height*]
        
        if min_events==0:
            # save 
            juli_functions.savefig(fig1, images_path, 'delta_percentile_robust_'+str(thresholds[0])+'-'+str(thresholds[1])+'_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_'+dr if dr!='' else ''][0]+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')
        
        else:
            # save 
            juli_functions.savefig(fig1, images_path, 'delta_percentile_robust_'+str(thresholds[0])+'-'+str(thresholds[1])+'_min_'+str(min_events)+'_evs_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_'+dr if dr!='' else ''][0]+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')


#%% ####### -----------------------  Cambio de heterogeneidad debido al evento

#############################
########################
#################               CHEQUEAR: no estoy seguro al mover el tiempo a LST, luego seleccionar, y volver a moverlo, si la seleccion se queda en el tiempo LST o si se vuelve a mover
#########################
##############################

data_sm_mor = dict()
data_sm_aft = dict()

sm_mor_std=dict()
sm_aft_std=dict()

sm_std_change=dict()
sm_std_change0=dict()

for model in models.keys():

    print('..... '+model+' ....')
    
    
    timename='time'
    if model=='JRA-55': timename='initial_time0_hours'

    lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]
    lat_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lat" in coord][0]
    
    # LIMITES DE LAS TIMEZONES #########################################
    timezones = [5,4,3,2] # A MEJORAR: SE PODRIA AGREGAR EL NEGATIVO DENTRO DE ESTA VARIABLE (son franjas -3, -4, etc)
    zonelimits = np.array([float(lon[model][0]), -67.5-0.24, -52.5-0.24, -37.5-0.24, float(lon[model][-1])]) 
            
    #diccionario temporal para guardar calculos por franjas horarias
    sm_aft = dict()
    sm_mor = dict()
    
    # creo datos diarios promediando las horas de interés para cada caso
    print('calculando por franjas horarias')
    init_time = timeit.time.time()           

    for nn,zone in enumerate(timezones):
                
        print(str(nn+1)+'/'+str(len(timezones)))
        
        # Muevo el tiempo, calculo y lo vuelvo al estado original
        data[model]['sm1'][''].coords[timename] = data[model]['sm1'][''][timename] - zone*3600000000000
        
        if 'JRA-55' in model: 
            sm_mor[zone] = data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}].loc[{timename:((data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}][timename+'.hour'])<=rango_sm[1] ) * ((data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}][timename+'.hour'])>rango_sm[1]-3), lon_name:slice(zonelimits[nn],zonelimits[nn+1])}]
    
            sm_aft[zone] = data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}].loc[{timename:((data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}][timename+'.hour'])<=rango_pre_aft[1]) * ((data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}][timename+'.hour'])>rango_pre_aft[1]-3), lon_name:slice(zonelimits[nn],zonelimits[nn+1])}]
        
        else:
            sm_mor[zone] = data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}].loc[{timename:(data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}][timename+'.hour'])==rango_sm[1], lon_name:slice(zonelimits[nn],zonelimits[nn+1])}]
    
            sm_aft[zone] = data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}].loc[{timename:(data[model]['sm1'][''].loc[{timename:slice(start_date, end_date)}][timename+'.hour'])==rango_pre_aft[1], lon_name:slice(zonelimits[nn],zonelimits[nn+1])}]

        data[model]['sm1'][''].coords[timename] = data[model]['sm1'][''][timename] + zone*3600000000000
        
    print(str(round((timeit.time.time()-init_time)/60,2))+' min')
    
    print('concateno, calculo std, selecciono dias evento')
    init_time = timeit.time.time()              
    
    # concateno    
    data_sm_mor[model] = xr.concat([sm_mor[i] for i in timezones], dim=lon_name, join='override')

    data_sm_aft[model] = xr.concat([sm_aft[i] for i in timezones], dim=lon_name, join='override')

    # calculo std de los alrededores
    sm_mor_std[model] = xr.concat([data_sm_mor[model].shift({lat_name:ii, lon_name:jj}) for ii,jj in list(itertools.product(range(-box_size2[0],box_size2[0]+1), repeat=2))], dim='shifted').std(dim='shifted')#.compute()
    sm_mor_std[model].to_netcdf(temp_path+'/sm_mor_std_'+model+'_'+start_date+'-'+end_date+'.nc')
    sm_mor_std[model] = xr.open_dataarray(temp_path+'/sm_mor_std_'+model+'_'+start_date+'-'+end_date+'.nc', chunks={timename:chunksize})
    
    sm_aft_std[model] = xr.concat([data_sm_aft[model].shift({lat_name:ii, lon_name:jj}) for ii,jj in list(itertools.product(range(-box_size2[0],box_size2[0]+1), repeat=2))], dim='shifted').std(dim='shifted')#.compute()
    sm_aft_std[model].to_netcdf(temp_path+'/sm_aft_std_'+model+'_'+start_date+'-'+end_date+'.nc')
    sm_aft_std[model] = xr.open_dataarray(temp_path+'/sm_aft_std_'+model+'_'+start_date+'-'+end_date+'.nc', chunks={timename:chunksize})
    
    # calculo la diferencia y selecciono los dias con evento
    hour_dif = int(sm_aft_std[model][0][timename+'.hour'] - sm_mor_std[model][0][timename+'.hour'])
    sm_mor_std[model].coords[timename] = sm_mor_std[model][timename] + hour_dif*3600000000000
    
    sm_std_change0[model] = (sm_aft_std[model] - sm_mor_std[model])
    sm_std_change0[model].coords[timename] = sm_std_change0[model][timename] - int(sm_std_change0[model][timename+'.hour'][0])*3600000000000
    
    sm_std_change[model] = sm_std_change0[model].where(~np.isnan(ys_e[model])).compute()

    print(str(round((timeit.time.time()-init_time)/60,2))+' min')


#------------- PLOTS
for model in models.keys():

    print('####### Plotting ########')
    print('..... '+model+' ....')
    
    
    timename='time'
    if model=='JRA-55': timename='initial_time0_hours'

    lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]
    lat_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lat" in coord][0]

    def season_sel(month):
        return np.asarray([m in seas for m in month])
    
    daily_time_months = np.asarray(sm_std_change[model].loc[{timename:slice(delta_period[0], delta_period[1])}][timename+'.month'])


    if degrade:
        res_lat = [None if len(ys_e[model][lat_name])%degrade_n==0 else len(ys_e[model][lat_name])%degrade_n*-1][0]
        res_lon = [None if len(ys_e[model][lon_name])%degrade_n==0 else len(ys_e[model][lon_name])%degrade_n*-1][0]
        
        sm_std_change_final = xr.concat([sm_std_change[model].loc[{timename:slice(delta_period[0], delta_period[1])}][season_sel(daily_time_months)][:,i:res_lat:degrade_n,j:res_lon:degrade_n] for i,j in list(itertools.product(range(0,degrade_n), repeat=2))], dim=timename, join='override')
        min_ev_mask = sm_std_change_final.count(axis=0).compute()

    else:
        sm_std_change_final = sm_std_change[model].loc[{timename:slice(delta_period[0], delta_period[1])}][season_sel(daily_time_months)]
        min_ev_mask = sm_std_change_final.count(axis=0).compute()


    # crear figura y axes
    fig1, ax = juli_functions.make_figure('Mean change in SM heterogeneity due to event \n'+model+' '+seas_name+' '+delta_period[0]+' - '+delta_period[1], figsize=(5,5), general_fontsize=9)
        
    # barra de colores
    if 'RCA4' in model: clevs = np.arange(-0.0001, 0.000125, 0.000025) 
    if 'ERA5' in model: clevs = np.arange(-0.001, 0.00125, 0.00025) 
    if 'JRA-55' in model: clevs = np.arange(-0.05, 0.06, 0.01) 

    barra= juli_functions.barra_zerocenter(clevs ,colormap=matplotlib.cm.get_cmap('RdBu'), no_extreme_colors=True)
    barra.set_over('#053061')
    barra.set_under('#67001f')
    
    
    if degrade:
        CS1 = juli_functions.plot_pcolormesh(ax, sm_std_change_final.mean(dim=timename).where(min_ev_mask>min_events), lon[model][:res_lon:degrade_n], lat[model][:res_lat:degrade_n], lonproj[model][:res_lat:degrade_n,:res_lon:degrade_n], latproj[model][:res_lat:degrade_n,:res_lon:degrade_n], proj,
                                             clevs, barra, '', titlesize = 12, coastwidth = 1, countrywidth = 1)
    else:
        CS1 = juli_functions.plot_pcolormesh(ax, sm_std_change_final.mean(dim=timename).where(min_ev_mask>min_events), lon[model], lat[model], lonproj[model], latproj[model], proj,
                                             clevs, barra, '', titlesize = 12, coastwidth = 1, countrywidth = 1)
    
    # add colorbar
    #fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
    juli_functions.add_colorbar(fig1, CS1, 'both', units_labels['sm1'], cbaxes= [0.9, 0.2, 0.02, 0.6]) #[*left*, *bottom*, *width*,  *height*]
    
    # save 
    juli_functions.savefig(fig1, images_path, 'mean_heterog_change_'+model+'_'+delta_period[0][0:4]+'-'+delta_period[1][0:4]+'_'+seas_name+['_degraded'+str(degrade_n) if degrade else ''][0]+'.png')


#%% ####### -----------------------  Ejemplo de deteccion de eventos
latslice=slice(-8.25,-6.5)
lonslice=slice(-64,-62.25)

colorbar = mcolors.ListedColormap(['#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#6baed6','#4292c6','#2171b5','#08519c','#08306b'])

datatoplot = (data_aft['ERA5']['pre'][102,:,:].loc[{lat_name:latslice, lon_name:lonslice}]/pre_multipliers)



#intento 1
plt.figure( figsize=(6,5),dpi=300) 
datatoplot.plot(cmap=plt.cm.Blues, levels=[0,1,2,3,4,5,6,7,8], hatch='x')

plt.pcolor(datatoplot.longitude, datatoplot.latitude, datatoplot.where(datatoplot<4), shading= 'nearest', cmap=mcolors.ListedColormap(['none']),  hatch='.', edgecolor='lightgray', lw=0)

#intento 2
fig1, ax = juli_functions.make_figure('example event', figsize=(5,5), general_fontsize=9)

clevs=[0,1,2,3,4,5,6,7,8,9]
barra= matplotlib.cm.get_cmap('Blues')
norm = mcolors.BoundaryNorm(boundaries=clevs, ncolors=barra.N)

CS1 = plt.pcolormesh(datatoplot.longitude, datatoplot.latitude, datatoplot, shading= 'nearest', cmap=plt.cm.Blues, norm=norm)

plt.colorbar(extend='max')

juli_functions.add_colorbar(fig1, CS1, 'max', units_labels['pre'], cbaxes= [0.9, 0.2, 0.02, 0.6]) #[*left*, *bottom*, *width*,  *height*]



#intento 3
# crear figura y axes
fig1, ax = juli_functions.make_figure('example event', figsize=(5,5), general_fontsize=9)
    
clevs=[0,1,2,3,4,5,6,7,8]
barra= juli_functions.barra_zerocenter(clevs ,colormap=matplotlib.cm.get_cmap('Blues'), no_extreme_colors=True)
barra.set_over('#053061')
barra.set_under('#67001f')


CS1 = juli_functions.plot_pcolormesh(ax, datatoplot, datatoplot.longitude, datatoplot.latitude, datatoplot.longitude, datatoplot.latitude, proj,
                                         clevs, barra, '', titlesize = 12, coastwidth = 1, countrywidth = 1)

# add colorbar
#fig1.subplots_adjust(right=0.1, left=, top=, bottom=)
juli_functions.add_colorbar(fig1, CS1, 'both', units_labels['sm1'], cbaxes= [0.9, 0.2, 0.02, 0.6]) #[*left*, *bottom*, *width*,  *height*]


#%%exit the program early
from sys import exit
exit()

#%% ####### -----------------------  Paso la SM de RCA4CLIM a la climatología como deberia ser

sm_grouped = data_xr['RCA4CLIM']['sm1']['var252'].loc[{'time': data_xr['RCA4CLIM']['sm1']['var252']['time.hour']==0}].groupby('time.dayofyear').mean(dim='time').compute()

sm_fixed = xr.zeros_like(data_xr['RCA4CLIM']['sm1']['var252']['time.hour']==0).groupby('time.dayofyear') + sm_grouped

sm_fixed_h = sm_fixed.resample({'time':'H'}).pad()

sm_fixed_h.to_netcdf(homepath+'CLIM/Data/1980-2012/sm/sm1_198001_201212_sSA_fixed2.nc')

lala3=xr.open_dataarray(homepath+'CLIM/Data/1980-2012/sm/sm1_198001_201212_sSA_fixed.nc', chunks={'time':(chunksize*10)})

plt.plot(np.array(lala3[:,0,50,50].loc[{'time':'1983'}][:]))
plt.plot(np.array(lala3[:,0,50,50].loc[{'time':'1984'}][:]))
plt.plot(np.array(lala3[:,0,50,50].loc[{'time':'1985'}][:]))
plt.plot(np.array(lala3[:,0,50,50].loc[{'time':'1986'}][:]))    
plt.plot(np.array(lala2[:,0,50,50][:24*4]))


#%% Pruebas chequeando si los eventos tienen P previa

#un evento
(ys_e[model].loc[{timename:slice('1998-03-13','1998-03-16'),lat_name:-20, lon_name:-50}].compute()).plot()


(ys_e[model].loc[{timename:slice('1998-02-01','1998-02-04'),lat_name:-20, lon_name:-50}].compute()).plot()


#para ver la P alrededor de ese evento
(data[model]['pre'][''].loc[{timename:slice('1998-03-14','1998-03-15'),lat_name:-20, lon_name:-50}].compute()*1000).plot()

(data[model]['pre'][''].loc[{timename:slice('1998-02-01','1998-02-04'),lat_name:-20, lon_name:-50}].compute()).plot()


# Ciclos diurnos medios del dia evento y su dia previo, promedio para todos los eventos del mapa

evcond = ~np.isnan(ys_e_cut[model])
evcondh = evcond.resample({timename:'H'}).pad().compute()

evcondh2 = evcondh.loc[{timename:slice(delta_period[0], '2012-12-30')}]

# faltaria seleccionar bien las bandas horarias, ahora aproximo todo a UTC-4
data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] + 4*3600000000000

pre_evcondh = data[model]['pre'][''].loc[{timename:slice(delta_period[0], '2012-12-30')}].where(evcondh2).groupby(timename+'.hour').median(dim=(timename, lat_name, lon_name),skipna=True).compute()

data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] + 24*3600000000000

pre_evcondh2 = data[model]['pre'][''].loc[{timename:slice(delta_period[0], '2012-12-30')}].where(evcondh2).groupby(timename+'.hour').median(dim=(timename, lat_name, lon_name),skipna=True).compute()

data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] - 28*3600000000000

plt.plot(xr.concat([pre_evcondh2, pre_evcondh], dim='hour', join='override').values)


# LO MISMO QUE ARRIBA PERO PARA LA MEDIANA faltaria seleccionar bien las bandas horarias, ahora aproximo todo a UTC-4
data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] + 4*3600000000000

pre_evcondh = data[model]['pre'][''].loc[{timename:slice(delta_period[0], '2012-12-30')}].where(evcondh2).groupby(timename+'.hour').median().compute()
pre_evcondhh = pre_evcondh.median(dim=(lat_name, lon_name),skipna=True).compute()

data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] + 24*3600000000000

pre_evcondh2 = data[model]['pre'][''].loc[{timename:slice(delta_period[0], '2012-12-30')}].where(evcondh2).groupby(timename+'.hour').median().compute()
pre_evcondhh2 = pre_evcondh2.median(dim=(lat_name, lon_name),skipna=True).compute()

data[model]['pre'][''].coords[timename] = data[model]['pre'][''][timename] - 28*3600000000000



#para trihorario
plt.plot(np.arange(3,51,3), xr.concat([pre_evcondhh2, pre_evcondhh], dim='hour', join='override').values)

#%% Pruebas mezcla GLEAM+CMORPH+ERA5

data['CMORPH']['pre'][''][0:80,100,100]
data['GLEAM']['sm1'][''][0:10,100,101]
data['ERA5']['vimfc2d'][''][0:10,98,101]

plt.plot(data['CMORPH']['pre'][''][0:360,100,100].resample({'time':'D'}).mean(dim='time').compute())
plt.plot((data['GLEAM']['sm1'][''][0:40,100,101]-0.4)*100)
plt.plot(data['ERA5']['vimfc2d'][''][0:960,98,101].resample({'time':'D'}).mean(dim='time').compute()*10000)

#%% Histogram SM vs Percentile
mask_clim = True #enmascarar la simulacion UNC?
normalized = True #normalizar las barras en funcion del total de deltas

for model in ['RCA4']:
    if 'CLIM' not in model:
    
        timename='time'
        if model=='JRA-55': timename='initial_time0_hours'
    
        lat_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lat" in coord][0]
        lon_name = [coord for coord in set(data[model][var_list[0]][''].coords.keys()) if "lon" in coord][0]
    
        sm_climatol = data[model]['sm1'][seas_name].mean(dim=timename).compute()
    
        deltas = [delta_e_ys_perc[model], delta_e_yt_perc[model], delta_e_yh_perc[model]]
        deltasclim = [delta_e_ys_perc[model+'CLIM'], delta_e_yt_perc[model+'CLIM'], delta_e_yh_perc[model+'CLIM']]
        
        # mascara para los puntos con menos de min_events eventos:
        min_ev_mask = ys_e_cut[model].count(axis=0)>=min_events
        
        # crear figura y axes
        fig1, ax = juli_functions.make_figure(model+' '+seas_name+' '+str(start_date)+' - '+str(end_date), figsize=(14,4), general_fontsize=9)
    
        
        # Si son muchos plots
        ax.set_visible(False)
        gs = gridspec.GridSpec(1,3, wspace=0.2, hspace=0.2) #si quiero multiples subplos
        
        for m, delta in enumerate(deltas):

            # mascara para valores de CLIM significativos (tambien cumple la condicion de min_events)
            clim_mask1 =  ( (ys_e_cut[model+'CLIM'].count(axis=0)>=min_events) * (deltasclim[m]>90))
            clim_mask2 =  ( (ys_e_cut[model+'CLIM'].count(axis=0)>=min_events) * (deltasclim[m]<10))
            clim_mask = ~ (clim_mask1 + clim_mask2)

            if mask_clim:
                total_mask = min_ev_mask * clim_mask
                model_save_name = model+'-'+model+'CLIM' #variable para en nombre del archivo de la imagen
            else:
                total_mask = min_ev_mask
                model_save_name = model #variable para en nombre del archivo de la imagen
            
            # Poner en loop si son muchos plot
            ax1 = plt.subplot(gs[m])
            
            # hago cada plot
            # ax1.scatter(sm_climatol, delta.where(min_ev_mask*(delta>90)), 1, color='blue', alpha=1)
            # ax1.scatter(sm_climatol, delta.where(min_ev_mask*(delta<10)), 1, color='red', alpha=1)
            
            
            
            hist_values, bins, aux = ax1.hist([np.asarray(sm_climatol.where(total_mask*(delta>90))).flatten(),
                      np.asarray(sm_climatol.where(total_mask*(delta<10))).flatten()],
                     bins=10, color=['#2e77ae', '#f2524e'])
            
            if normalized:
                total_hist_values = hist_values[0] + hist_values[1]
                
                ax1.bar((bins[:-1]+bins[1:])/2, hist_values[0]/total_hist_values*100, width=(bins[1]-bins[0])/1.2, color='#2e77ae')
                ax1.bar((bins[:-1]+bins[1:])/2, hist_values[1]/total_hist_values*100, width=(bins[1]-bins[0])/1.2, color='#f2524e', bottom=hist_values[0]/total_hist_values*100)
                ax1.set_ylim(0,100)
                
                xlim1, xlim2 = ax1.get_xlim()
                plt.plot([xlim1,xlim2],[50,50], ls='dashed', color= 'yellow', linewidth=3)
                
                ax1.set_xlim(xlim1, xlim2)
                
                ax2 = ax1.twinx()
                ax2.plot((bins[:-1]+bins[1:])/2, hist_values[0]+hist_values[1], color='black', linewidth=2)
                
            ax1.set_title(numbering[m]+' '+plot_titles[m], fontsize=12, loc='center')
            # etiquetas
            plt.xlabel('HS'+' ['+units_labels['sm1']+']')

        
        if min_events==0:
            # save 
            juli_functions.savefig(fig1, images_path, 'histogram_delta_percentile_SM_'+['normalized_' if normalized else None][0]+model_save_name+'_'+start_date[0:4]+'-'+end_date[0:4]+'_'+seas_name+'.png')
        
        else:
            # save 
            juli_functions.savefig(fig1, images_path, 'histogram_delta_percentile_SM_min_'+str(min_events)+'_evs_'+['normalized_' if normalized else None][0]+model_save_name+'_'+start_date[0:4]+'-'+end_date[0:4]+'_'+seas_name+'.png')
    

    


#%%%% ---------------- Intercomparación de variabilidades de SM de distintas fuentes
model='RCA4'
seas = 'ONDJFM'
var = 'sm1'

data['GLEAM'][var][seas] = data['GLEAM'][var][seas].transpose('time', 'lat', 'lon')

mult = {'RCA4': 1/0.07,
        'RCA4CLIM': 1/0.07,
        'ERA5': 1,
        'GLEAM': 1,
        'GLDASNOAH': 1/1000/0.1,
        'ESACCI': 1}

################# Variabilidad interanual diaria
data_D = dict()
data_D_iv = dict()
for nax, model in enumerate(models.keys()):
    print(model)
    if 'GLEAM' in model: data_D[model] = data['GLEAM'][var][seas]
    else: data_D[model] = data[model][var][seas].resample({'time':'D'}).mean()*mult[model]
    
    data_D_iv[model] = (data_D[model].groupby('time.dayofyear') - data_D[model].groupby('time.dayofyear').mean() ).std(dim='time').compute()


# crear figura y axes
fig1, ax = juli_functions.make_figure('', figsize=(10,8), general_fontsize=10) #fig size in inches (width, height)

# Si son muchos plots
ax.set_visible(False)
gs = gridspec.GridSpec(2,3, wspace=0.01, hspace=0.2) #si quiero multiples subplos

for nax, model in enumerate(models.keys()):
    # Poner en loop si son muchos plot
    ax1 = plt.subplot(gs[nax], projection = proj)
    
    # barra de colores
    clevs = np.arange(0,0.1,0.01)
    barra= juli_functions.barra_whitecenter(clevs ,colormap=matplotlib.cm.get_cmap('Reds'), no_extreme_colors=True)
    barra.set_over('Black')
    barra.set_under('white')

    pll = (False, False)
    if nax == 0:
        pll = (False, True)
    elif nax == gs.nrows*gs.ncols-gs.ncols:
        pll = (True, True)
    elif int(nax/gs.ncols) == gs.nrows-1:
        pll = (True, False)

    # hago cada plot
    if model in ['RCA4', 'RCA4CLIM', 'ERA5']:
        CS1 = juli_functions.plot_pcolormesh(ax1, data_D_iv[model].where(data[model]['lsmask'][seas][0]), lon[model], lat[model], lonproj[model], latproj[model],
                                         proj, clevs, barra, model, titlesize = 12, coastwidth = 1, countrywidth = 1, printlonslats=pll)
    else:
        CS1 = juli_functions.plot_pcolormesh(ax1, data_D_iv[model], lon[model], lat[model], lonproj[model], latproj[model],
                                         proj, clevs, barra, model, titlesize = 12, coastwidth = 1, countrywidth = 1, printlonslats=pll)        
        
# add colorbar
#fig1.subplots_adjust(right=, left=, top=, bottom=)
cbar = juli_functions.add_colorbar(fig1, CS1, 'max', '$\mathregular{m^3⋅m^{-3}}$', cientific=(0,0),  cbaxes= [0.9, 0.2, 0.02, 0.6], labelpad=4) #[*left*, *bottom*, *width*,  *height*]

# save 
juli_functions.savefig(fig1, '/home/julian.giles/datos/CTL/Images/climatologies/', 'daily_anom_std_sm1_'+'_'.join(models.keys())+'_'+seas+'_'+start_date[0:4]+'-'+end_date[0:4]+'.png')

############# Variabilidad diaria sin media movil 90 dias
matplotlib.rcParams.update({'font.size':6})
fig, axs = plt.subplots(2, 3, figsize=(9,6), subplot_kw=dict(projection=proj), sharex=True, sharey=True)
for nax, model in enumerate(models.keys()):
    (data_D[model] - data_D[model].compute().rolling({'time':90}, center=True, min_periods=45).mean() ).std(dim='time').plot(ax=axs.flatten()[nax], vmin=0, vmax=0.1, cmap='Reds')
    axs.flatten()[nax].set_title(model)
    
############# Variabilidad total (ciudado que GLEAM es diario asi que la variabilidad diurna no está incluida)
matplotlib.rcParams.update({'font.size':6})
fig, axs = plt.subplots(2, 3, figsize=(9,6), subplot_kw=dict(projection=proj), sharex=True, sharey=True)
for nax, model in enumerate(models.keys()):
    (data[model][var][seas].std(dim='time')*mult[model]).plot(ax=axs.flatten()[nax], vmin=0, vmax=0.1, cmap='Reds')
    axs.flatten()[nax].set_title(model)




# Segmenting & clustering housing estates in the county of Argolis, Greece

## Introduction

Argos is the largest city of the county of Argolis, Peloponnese region, in Greece. It is one of the oldest continuously inhabited cities in the entire world.

Argos presents great potential for investors. The types of investments may vary, but what is certain is that an exploratory analysis regarding the venues in each estate of Argolis should take place in order to indicate potential market gaps and patterns. It is understood that this analysis is conducted only for the purpose of obtaining a general view, or "feeling", of the current conditions.

With the use of the Foursquare API, an exploratory analysis takes place regarding the neighborhoods in the city of Athens, Greece. The purpose is to get the most common venue categories for each neighborhood. Neighborhoods are then grouped into clusters, using the k-means machine learning algorithm.

## Data

The data used consists of two parts.

The first part is the data from http://geodata.gov.gr/, which have to do with the geographical region of the counties in Greece. The analysis targeted the county of Argolis in particular.

The second part is the data from Foursquare, which we have obtained through the Foursquare API and have to do with the venues regarding the county of Argolis in particular.

## Methodology

##### At first we are going to import the necessary libraries to conduct the analysis.


```python
import pandas as pd
import numpy as np
import requests
import urllib.request
import time
```

What we did, is that we entered the website: "http://geodata.gov.gr/" in order to obtain the necessary data for the analysis. The next step is to read the file into a pandas dataframe.


```python
df_greece = pd.read_csv('greece_housing_estates.csv')
print('The dimension of the initial dataset: ', df_greece.shape)
print('')
df_greece.head(3)
```

    The dimension of the initial dataset:  (13259, 20)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FID</th>
      <th>objectid</th>
      <th>CODE_OIK</th>
      <th>NAME_OIK</th>
      <th>CODE_GDIAM</th>
      <th>NAMEF_OIK</th>
      <th>point_x</th>
      <th>point_y</th>
      <th>lat</th>
      <th>lon</th>
      <th>h</th>
      <th>edra_diam</th>
      <th>CODE_DIAM</th>
      <th>NAME_DIAM</th>
      <th>CODE_OTA</th>
      <th>NAME_OTA</th>
      <th>CODE_NOM</th>
      <th>NAME_NOM</th>
      <th>NAME_GDIAM</th>
      <th>the_geom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f45c73bd-d733-4fe0-871b-49f270c56a75.0</td>
      <td>1</td>
      <td>04130201</td>
      <td>Αετός</td>
      <td>0</td>
      <td>Αετός,ο</td>
      <td>538487.1875</td>
      <td>4207068.0</td>
      <td>38.010479</td>
      <td>24.438425</td>
      <td>42</td>
      <td>1</td>
      <td>04130200</td>
      <td>Τ.Δ.Αετού</td>
      <td>04130000</td>
      <td>ΔΗΜΟΣ ΚΑΡΥΣΤΟΥ</td>
      <td>04</td>
      <td>ΝΟΜΟΣ ΕΥΒΟΙΑΣ</td>
      <td>ΛΟΙΠΗ ΣΤΕΡΕΑ ΕΛΛΑΣ ΚΑΙ ΕΥΒΟ</td>
      <td>POINT (538487.2029 4207068.2091)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f45c73bd-d733-4fe0-871b-49f270c56a75.1</td>
      <td>2</td>
      <td>04130101</td>
      <td>Κάρυστος</td>
      <td>0</td>
      <td>Κάρυστος,η</td>
      <td>536774.0000</td>
      <td>4207489.5</td>
      <td>38.014347</td>
      <td>24.418932</td>
      <td>22</td>
      <td>1</td>
      <td>04130100</td>
      <td>Τ.Δ.Καρύστου</td>
      <td>04130000</td>
      <td>ΔΗΜΟΣ ΚΑΡΥΣΤΟΥ</td>
      <td>04</td>
      <td>ΝΟΜΟΣ ΕΥΒΟΙΑΣ</td>
      <td>ΛΟΙΠΗ ΣΤΕΡΕΑ ΕΛΛΑΣ ΚΑΙ ΕΥΒΟ</td>
      <td>POINT (536774.0006999996 4207489.7432)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f45c73bd-d733-4fe0-871b-49f270c56a75.2</td>
      <td>3</td>
      <td>04130601</td>
      <td>Πλατανιστός</td>
      <td>0</td>
      <td>Πλατανιστός,ο</td>
      <td>545163.4375</td>
      <td>4207678.0</td>
      <td>38.015667</td>
      <td>24.514513</td>
      <td>193</td>
      <td>1</td>
      <td>04130600</td>
      <td>Τ.Δ.Πλατανιστού</td>
      <td>04130000</td>
      <td>ΔΗΜΟΣ ΚΑΡΥΣΤΟΥ</td>
      <td>04</td>
      <td>ΝΟΜΟΣ ΕΥΒΟΙΑΣ</td>
      <td>ΛΟΙΠΗ ΣΤΕΡΕΑ ΕΛΛΑΣ ΚΑΙ ΕΥΒΟ</td>
      <td>POINT (545163.4286000001 4207678)</td>
    </tr>
  </tbody>
</table>
</div>



##### Change the column names of the newly created dataframe in order for them to match the assignment requirements.


```python
df_greece.rename(columns={'NAME_OIK':'Estate', 'NAME_OTA':'Borough', 'lat':'Latitude', 'lon':'Longitude', 'NAME_NOM':'County'}, inplace=True)
df_greece2 = df_greece[['Estate', 'Borough', 'County', 'Latitude', 'Longitude']]
df_greece2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Borough</th>
      <th>County</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Αετός</td>
      <td>ΔΗΜΟΣ ΚΑΡΥΣΤΟΥ</td>
      <td>ΝΟΜΟΣ ΕΥΒΟΙΑΣ</td>
      <td>38.010479</td>
      <td>24.438425</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Κάρυστος</td>
      <td>ΔΗΜΟΣ ΚΑΡΥΣΤΟΥ</td>
      <td>ΝΟΜΟΣ ΕΥΒΟΙΑΣ</td>
      <td>38.014347</td>
      <td>24.418932</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Πλατανιστός</td>
      <td>ΔΗΜΟΣ ΚΑΡΥΣΤΟΥ</td>
      <td>ΝΟΜΟΣ ΕΥΒΟΙΑΣ</td>
      <td>38.015667</td>
      <td>24.514513</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Γραμπιά</td>
      <td>ΔΗΜΟΣ ΚΑΡΥΣΤΟΥ</td>
      <td>ΝΟΜΟΣ ΕΥΒΟΙΑΣ</td>
      <td>38.031082</td>
      <td>24.425997</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Μύλοι</td>
      <td>ΔΗΜΟΣ ΚΑΡΥΣΤΟΥ</td>
      <td>ΝΟΜΟΣ ΕΥΒΟΙΑΣ</td>
      <td>38.028870</td>
      <td>24.435293</td>
    </tr>
  </tbody>
</table>
</div>



##### Proceed with what it's necessary in order to fulfill the assignment requirements.


```python
print('Are there any missing data in any of the columns of the dataframe?')
print('')

df_greece2.isnull().any()
```

    Are there any missing data in any of the columns of the dataframe?
    





    Estate       False
    Borough      False
    County       False
    Latitude     False
    Longitude    False
    dtype: bool



##### Let's see how many data points lie under each County.


```python
df_greece2['County'].value_counts()
```




    ΝΟΜΟΣ ΚΥΚΛΑΔΩΝ                                        628
    ?????????? ???????????????? ?????? ???????????????    571
    ΝΟΜΟΣ ΑΧΑΪΑΣ                                          525
    ΝΟΜΟΣ ΜΕΣΣΗΝΙΑΣ                                       514
    ΝΟΜΟΣ ΙΩΑΝΝΙΝΩΝ                                       495
    ΝΟΜΟΣ ΧΑΝΙΩΝ                                          478
    ΝΟΜΟΣ ΗΡΑΚΛΕΙΟΥ                                       460
    ΝΟΜΟΣ ΗΛΕΙΑΣ                                          436
    ΝΟΜΟΣ ΛΑΚΩΝΙΑΣ                                        436
    ΝΟΜΟΣ ΕΥΒΟΙΑΣ                                         427
    ΝΟΜΟΣ ΑΡΚΑΔΙΑΣ                                        425
    ΝΟΜΟΣ ΛΑΣΙΘΙΟΥ                                        328
    ΝΟΜΟΣ ΚΕΡΚΥΡΑΣ                                        309
    ΝΟΜΟΣ ΡΕΘΥΜΝΗΣ                                        290
    ΝΟΜΟΣ ΑΡΤΗΣ                                           284
    ΝΟΜΟΣ ΦΘΙΩΤΙΔΟΣ                                       284
    ΝΟΜΟΣ ΚΑΡΔΙΤΣΗΣ                                       282
    ΝΟΜΟΣ ΛΑΡΙΣΗΣ                                         276
    ΝΟΜΟΣ ΔΩΔΕΚΑΝΗΣΟΥ                                     261
    ΝΟΜΟΣ ΛΕΣΒΟΥ                                          258
    ΝΟΜΟΣ ΤΡΙΚΑΛΩΝ                                        242
    ΝΟΜΟΣ ΜΑΓΝΗΣΙΑΣ                                       242
    ΝΟΜΟΣ ΣΑΜΟΥ                                           223
    ΝΟΜΟΣ ΚΟΖΑΝΗΣ                                         221
    ΝΟΜΟΣ ΚΟΡΙΝΘΙΑΣ                                       214
    ΝΟΜΑΡΧΙΑ ΠΕΙΡΑΙΩΣ                                     193
    ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ                                       192
    ΝΟΜΟΣ ΣΕΡΡΩΝ                                          192
    ΝΟΜΑΡΧΙΑ ΑΝΑΤΟΛΙΚΗΣ ΑΤΤΙΚΗ                            188
    ΝΟΜΟΣ ΕΥΡΥΤΑΝΙΑΣ                                      184
    ΝΟΜΟΣ ΚΕΦΑΛΛΗΝΙΑΣ                                     183
    ΝΟΜΟΣ ΡΟΔΟΠΗΣ                                         183
    ΝΟΜΟΣ ΞΑΝΘΗΣ                                          180
    ΝΟΜΟΣ ΧΑΛΚΙΔΙΚΗΣ                                      174
    ΝΟΜΟΣ ΘΕΣΠΡΩΤΙΑΣ                                      174
    ΝΟΜΟΣ ΕΒΡΟΥ                                           173
    ΝΟΜΟΣ ΘΕΣΣΑΛΟΝΙΚΗΣ                                    167
    ΝΟΜΟΣ ΚΙΛΚΙΣ                                          166
    ΝΟΜΟΣ ΧΙΟΥ                                            153
    ΝΟΜΟΣ ΦΩΚΙΔΟΣ                                         148
    ΝΟΜΟΣ ΚΑΒΑΛΑΣ                                         146
    ΝΟΜΟΣ ΠΡΕΒΕΖΗΣ                                        140
    ΝΟΜΟΣ ΠΕΛΛΗΣ                                          132
    ΝΟΜΟΣ ΒΟΙΩΤΙΑΣ                                        124
    ΝΟΜΟΣ ΔΡΑΜΑΣ                                          121
    ΝΟΜΟΣ ΓΡΕΒΕΝΩΝ                                        117
    ΝΟΜΟΣ ΚΑΣΤΟΡΙΑΣ                                       113
    ΝΟΜΟΣ ΗΜΑΘΙΑΣ                                         110
    ΝΟΜΟΣ ΦΛΩΡΙΝΗΣ                                        105
    ΝΟΜΟΣ ΠΙΕΡΙΑΣ                                          90
    ΝΟΜΟΣ ΖΑΚΥΝΘΟΥ                                         88
    ΝΟΜΟΣ ΛΕΥΚΑΔΟΣ                                         71
    ΝΟΜΑΡΧΙΑ ΑΘΗΝΩΝ                                        51
    ΝΟΜΑΡΧΙΑ ΔΥΤΙΚΗΣ ΑΤΤΙΚΗΣ                               50
    ΑΓΙΟΝ ΟΡΟΣ (ΑΥΤΟΔΙΟΙΚΗΤΟ)                              42
    Name: County, dtype: int64



##### It would be nice to explore the situation in the Peloponnese region, targeting the county of Argolis. We proceed as follows.


```python
df_argolis = df_greece2[df_greece2['County'].str.contains('ΑΡΓΟΛΙΔΟΣ')]
df_argolis.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Borough</th>
      <th>County</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8422</th>
      <td>Πορτοχέλιον</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.324680</td>
      <td>23.140156</td>
    </tr>
    <tr>
      <th>8462</th>
      <td>Κρανίδιον</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.377983</td>
      <td>23.157290</td>
    </tr>
    <tr>
      <th>8467</th>
      <td>Ερμιόνη</td>
      <td>ΔΗΜΟΣ ΕΡΜΙΟΝΗΣ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.383213</td>
      <td>23.242247</td>
    </tr>
    <tr>
      <th>8484</th>
      <td>Θερμησία</td>
      <td>ΔΗΜΟΣ ΕΡΜΙΟΝΗΣ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.409878</td>
      <td>23.322416</td>
    </tr>
    <tr>
      <th>8488</th>
      <td>Κοιλάς</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.412220</td>
      <td>23.123734</td>
    </tr>
  </tbody>
</table>
</div>



##### We are going to have to install geocoder in order to be able to proceed.


```python
! pip install geocoder
```

    Collecting geocoder
    [?25l  Downloading https://files.pythonhosted.org/packages/4f/6b/13166c909ad2f2d76b929a4227c952630ebaf0d729f6317eb09cbceccbab/geocoder-1.38.1-py2.py3-none-any.whl (98kB)
    [K     |████████████████████████████████| 102kB 10.5MB/s ta 0:00:01
    [?25hCollecting ratelim (from geocoder)
      Downloading https://files.pythonhosted.org/packages/f2/98/7e6d147fd16a10a5f821db6e25f192265d6ecca3d82957a4fdd592cad49c/ratelim-0.1.6-py2.py3-none-any.whl
    Requirement already satisfied: requests in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from geocoder) (2.22.0)
    Collecting future (from geocoder)
    [?25l  Downloading https://files.pythonhosted.org/packages/45/0b/38b06fd9b92dc2b68d58b75f900e97884c45bedd2ff83203d933cf5851c9/future-0.18.2.tar.gz (829kB)
    [K     |████████████████████████████████| 829kB 27.7MB/s eta 0:00:01
    [?25hRequirement already satisfied: six in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from geocoder) (1.13.0)
    Collecting click (from geocoder)
    [?25l  Downloading https://files.pythonhosted.org/packages/fa/37/45185cb5abbc30d7257104c434fe0b07e5a195a6847506c074527aa599ec/Click-7.0-py2.py3-none-any.whl (81kB)
    [K     |████████████████████████████████| 81kB 17.4MB/s eta 0:00:01
    [?25hRequirement already satisfied: decorator in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from ratelim->geocoder) (4.4.1)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->geocoder) (1.25.7)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->geocoder) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->geocoder) (2.8)
    Requirement already satisfied: certifi>=2017.4.17 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->geocoder) (2019.9.11)
    Building wheels for collected packages: future
      Building wheel for future (setup.py) ... [?25ldone
    [?25h  Stored in directory: /home/jupyterlab/.cache/pip/wheels/8b/99/a0/81daf51dcd359a9377b110a8a886b3895921802d2fc1b2397e
    Successfully built future
    Installing collected packages: ratelim, future, click, geocoder
    Successfully installed click-7.0 future-0.18.2 geocoder-1.38.1 ratelim-0.1.6



```python
!conda install -c conda-forge geopy --yes
!conda install -c conda-forge folium=0.5.0 --yes
```

    Solving environment: done
    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.11
      latest version: 4.8.0
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    
    
    ## Package Plan ##
    
      environment location: /home/jupyterlab/conda/envs/python
    
      added / updated specs: 
        - geopy
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        certifi-2019.11.28         |           py36_0         149 KB  conda-forge
        scikit-learn-0.20.1        |   py36h22eb022_0         5.7 MB
        liblapack-3.8.0            |      11_openblas          10 KB  conda-forge
        scipy-1.3.2                |   py36h921218d_0        18.0 MB  conda-forge
        geographiclib-1.50         |             py_0          34 KB  conda-forge
        libopenblas-0.3.6          |       h5a2b251_2         7.7 MB
        liblapacke-3.8.0           |      11_openblas          10 KB  conda-forge
        numpy-1.17.3               |   py36h95a1406_0         5.2 MB  conda-forge
        libcblas-3.8.0             |      11_openblas          10 KB  conda-forge
        libblas-3.8.0              |      11_openblas          10 KB  conda-forge
        geopy-1.20.0               |             py_0          57 KB  conda-forge
        blas-2.11                  |         openblas          10 KB  conda-forge
        ------------------------------------------------------------
                                               Total:        36.9 MB
    
    The following NEW packages will be INSTALLED:
    
        geographiclib: 1.50-py_0                              conda-forge
        geopy:         1.20.0-py_0                            conda-forge
        libblas:       3.8.0-11_openblas                      conda-forge
        libcblas:      3.8.0-11_openblas                      conda-forge
        liblapack:     3.8.0-11_openblas                      conda-forge
        liblapacke:    3.8.0-11_openblas                      conda-forge
        libopenblas:   0.3.6-h5a2b251_2                                  
    
    The following packages will be UPDATED:
    
        blas:          1.1-openblas                           conda-forge --> 2.11-openblas         conda-forge
        certifi:       2019.9.11-py36_0                       conda-forge --> 2019.11.28-py36_0     conda-forge
        numpy:         1.16.2-py36_blas_openblash1522bff_0    conda-forge [blas_openblas] --> 1.17.3-py36h95a1406_0 conda-forge
        scipy:         1.2.1-py36_blas_openblash1522bff_0     conda-forge [blas_openblas] --> 1.3.2-py36h921218d_0  conda-forge
    
    The following packages will be DOWNGRADED:
    
        scikit-learn:  0.20.1-py36_blas_openblashebff5e3_1200 conda-forge [blas_openblas] --> 0.20.1-py36h22eb022_0            
    
    
    Downloading and Extracting Packages
    certifi-2019.11.28   | 149 KB    | ##################################### | 100% 
    scikit-learn-0.20.1  | 5.7 MB    | ##################################### | 100% 
    liblapack-3.8.0      | 10 KB     | ##################################### | 100% 
    scipy-1.3.2          | 18.0 MB   | ##################################### | 100% 
    geographiclib-1.50   | 34 KB     | ##################################### | 100% 
    libopenblas-0.3.6    | 7.7 MB    | ##################################### | 100% 
    liblapacke-3.8.0     | 10 KB     | ##################################### | 100% 
    numpy-1.17.3         | 5.2 MB    | ##################################### | 100% 
    libcblas-3.8.0       | 10 KB     | ##################################### | 100% 
    libblas-3.8.0        | 10 KB     | ##################################### | 100% 
    geopy-1.20.0         | 57 KB     | ##################################### | 100% 
    blas-2.11            | 10 KB     | ##################################### | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    Solving environment: done
    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.11
      latest version: 4.8.0
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    
    
    # All requested packages already installed.
    



```python
import geocoder
from geopy.geocoders import Nominatim
```

##### Which are the coordinates of Argolis, Greece?


```python
address = 'Argolis, Greece'
geolocator = Nominatim(user_agent="argolis_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Argolis, Greece are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of Argolis, Greece are 37.56861385, 22.8605054603859.


##### Now let's visualize the map of estates in the Argolis region.


```python
# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

import folium # map rendering library
```


```python
map_argolis = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_argolis['Latitude'], df_argolis['Longitude'], df_argolis['Borough'], df_argolis['Estate']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_argolis)  
    
map_argolis
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOScsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbMzcuNTY4NjEzODUsMjIuODYwNTA1NDYwMzg1OV0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfZTQ4NDM5MWY2ODM0NGI1MDgyMWVhYjQwMDViMmQ5NmQgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMzMzUzM2Y4NzRkOTQ4ZjlhODdjNGU3ZGYwN2VlYjJkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuMzI0NjgwMzMsMjMuMTQwMTU1NzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDM1NTJkYWNhMmZlNGI5OGIxODUyNGUyZWNmMjczNTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTc2YTIzM2Q2YjFkNDI0Zjk0MGUzY2MxYWY3NjY4ZTMgPSAkKCc8ZGl2IGlkPSJodG1sX2U3NmEyMzNkNmIxZDQyNGY5NDBlM2NjMWFmNzY2OGUzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OoM6/z4HPhM6/z4fOrc67zrnOv869LCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDM1NTJkYWNhMmZlNGI5OGIxODUyNGUyZWNmMjczNTguc2V0Q29udGVudChodG1sX2U3NmEyMzNkNmIxZDQyNGY5NDBlM2NjMWFmNzY2OGUzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMzMzUzM2Y4NzRkOTQ4ZjlhODdjNGU3ZGYwN2VlYjJkLmJpbmRQb3B1cChwb3B1cF80MzU1MmRhY2EyZmU0Yjk4YjE4NTI0ZTJlY2YyNzM1OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82OTAzYjgwNDliODE0ZWFkOTEwMDI0YjE3OWU1MzQ0MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjM3Nzk4MzA5LDIzLjE1NzI4OTUxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QxZWZjMmQyOGVlZDRiMWU5ZDQ1MWM1NDMwZDZhZDY1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg4YmI5YzhiMGY3NjRlY2ZiYzBjNjkxZTg2OGFiN2M4ID0gJCgnPGRpdiBpZD0iaHRtbF84OGJiOWM4YjBmNzY0ZWNmYmMwYzY5MWU4NjhhYjdjOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprPgc6xzr3Or860zrnOv869LCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDFlZmMyZDI4ZWVkNGIxZTlkNDUxYzU0MzBkNmFkNjUuc2V0Q29udGVudChodG1sXzg4YmI5YzhiMGY3NjRlY2ZiYzBjNjkxZTg2OGFiN2M4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY5MDNiODA0OWI4MTRlYWQ5MTAwMjRiMTc5ZTUzNDQxLmJpbmRQb3B1cChwb3B1cF9kMWVmYzJkMjhlZWQ0YjFlOWQ0NTFjNTQzMGQ2YWQ2NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZTAwOTQwMGEzMGM0NDRhODcyZmEzYzMzNTM2ZTdhOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjM4MzIxMzA0LDIzLjI0MjI0NjYzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU4Y2M5YWE4YWIzNTQyM2ViMzMwNTNkMjliYTU5NGM5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJjOGMwNTE1NzA2YzRkYmU4N2FmMmFjZWY2MjM5MmZkID0gJCgnPGRpdiBpZD0iaHRtbF8yYzhjMDUxNTcwNmM0ZGJlODdhZjJhY2VmNjIzOTJmZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpXPgc68zrnPjM69zrcsIM6UzpfOnM6fzqMgzpXOoc6czpnOn86dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNThjYzlhYThhYjM1NDIzZWIzMzA1M2QyOWJhNTk0Yzkuc2V0Q29udGVudChodG1sXzJjOGMwNTE1NzA2YzRkYmU4N2FmMmFjZWY2MjM5MmZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzllMDA5NDAwYTMwYzQ0NGE4NzJmYTNjMzM1MzZlN2E4LmJpbmRQb3B1cChwb3B1cF81OGNjOWFhOGFiMzU0MjNlYjMzMDUzZDI5YmE1OTRjOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mZjJlY2JhMjY3YWY0ZmRhYTAwODBhZWI4NjU0ZmRlYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQwOTg3Nzc4LDIzLjMyMjQxNjMxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I0MDdlNTdiOTQ0NTQwN2JhZGJlYjI1NGZiMGNjNDMxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RhNjU3NDdiNWI0MzQ0MWY4NDQyMmE0NzMxMjcyZTQ2ID0gJCgnPGRpdiBpZD0iaHRtbF9kYTY1NzQ3YjViNDM0NDFmODQ0MjJhNDczMTI3MmU0NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpjOtc+BzrzOt8+Dzq/OsSwgzpTOl86czp/OoyDOlc6hzpzOmc6fzp3Ol86jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNDA3ZTU3Yjk0NDU0MDdiYWRiZWIyNTRmYjBjYzQzMS5zZXRDb250ZW50KGh0bWxfZGE2NTc0N2I1YjQzNDQxZjg0NDIyYTQ3MzEyNzJlNDYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmYyZWNiYTI2N2FmNGZkYWEwMDgwYWViODY1NGZkZWMuYmluZFBvcHVwKHBvcHVwX2I0MDdlNTdiOTQ0NTQwN2JhZGJlYjI1NGZiMGNjNDMxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRjYWM3OTI1ZTkyNjRiN2NhZTFhZDJjODFhZTFhYTU2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDEyMjIsMjMuMTIzNzMzNTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWM2YWZjYzAzM2M3NDkyM2FmYWEzZmQxYjMwMzM2NTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDk3NjljOWI5Y2I3NDUyMWFmYTYxNzQ0MWEzYTQ5MTYgPSAkKCc8ZGl2IGlkPSJodG1sXzQ5NzY5YzliOWNiNzQ1MjFhZmE2MTc0NDFhM2E0OTE2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6/zrnOu86sz4IsIM6UzpfOnM6fzqMgzprOoc6Rzp3Omc6UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hYzZhZmNjMDMzYzc0OTIzYWZhYTNmZDFiMzAzMzY1My5zZXRDb250ZW50KGh0bWxfNDk3NjljOWI5Y2I3NDUyMWFmYTYxNzQ0MWEzYTQ5MTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGNhYzc5MjVlOTI2NGI3Y2FlMWFkMmM4MWFlMWFhNTYuYmluZFBvcHVwKHBvcHVwX2FjNmFmY2MwMzNjNzQ5MjNhZmFhM2ZkMWIzMDMzNjUzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2JlNjQ3N2ViNWUwNzQ4ZjhiNzRkN2EzMzgyMGE1YTAzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDI4MzE0MjEsMjMuMTc2Mzc0NDRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjFhYTY4NzIxNWY1NGYzM2FlNjk4MjEzMWQ5YWI2OWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTAxNzYxM2NjNzg4NDk5ZDg5OTM3Y2FjYzk0MWVkZjggPSAkKCc8ZGl2IGlkPSJodG1sX2UwMTc2MTNjYzc4ODQ5OWQ4OTkzN2NhY2M5NDFlZGY4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Ops6/z43Pgc69zr/OuSwgzpTOl86czp/OoyDOms6hzpHOnc6ZzpTOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYxYWE2ODcyMTVmNTRmMzNhZTY5ODIxMzFkOWFiNjliLnNldENvbnRlbnQoaHRtbF9lMDE3NjEzY2M3ODg0OTlkODk5MzdjYWNjOTQxZWRmOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iZTY0NzdlYjVlMDc0OGY4Yjc0ZDdhMzM4MjBhNWEwMy5iaW5kUG9wdXAocG9wdXBfNjFhYTY4NzIxNWY1NGYzM2FlNjk4MjEzMWQ5YWI2OWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWIyMWYzM2NhMTE2NGY3MmEzM2ZjZWJlNjM5MzVmYTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40NDE3NTMzOSwyMy4yNjQ4MDI5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZjU4MjMyMWNmN2M0NjQxOWMwNjFiN2YwMTlkZDVhMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82Y2MwNjA0OWM4ZGY0N2MwYjI0YjAxOTUyZWNkYmQ3OSA9ICQoJzxkaXYgaWQ9Imh0bWxfNmNjMDYwNDljOGRmNDdjMGIyNGIwMTk1MmVjZGJkNzkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6XzrvOuc+MzrrOsc+Dz4TPgc6/zr0sIM6UzpfOnM6fzqMgzpXOoc6czpnOn86dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmY1ODIzMjFjZjdjNDY0MTljMDYxYjdmMDE5ZGQ1YTAuc2V0Q29udGVudChodG1sXzZjYzA2MDQ5YzhkZjQ3YzBiMjRiMDE5NTJlY2RiZDc5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzliMjFmMzNjYTExNjRmNzJhMzNmY2ViZTYzOTM1ZmEwLmJpbmRQb3B1cChwb3B1cF82ZjU4MjMyMWNmN2M0NjQxOWMwNjFiN2YwMTlkZDVhMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MmJiODA0MzdlM2Y0M2I1OWYzZWFhM2VkODE0ODhiZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQ2MDkyNjA2LDIzLjE3MTA3OTY0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU4NWUwY2E3N2ZlMjQ1ZTg5YTY4NTI2Yjc4Y2U5N2M2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlkZTQxNTBiZjc1OTQyZmNhMjFhZjlhOTEzYzNmNzAxID0gJCgnPGRpdiBpZD0iaHRtbF85ZGU0MTUwYmY3NTk0MmZjYTIxYWY5YTkxM2MzZjcwMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpTOr860z4XOvM6xLCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTg1ZTBjYTc3ZmUyNDVlODlhNjg1MjZiNzhjZTk3YzYuc2V0Q29udGVudChodG1sXzlkZTQxNTBiZjc1OTQyZmNhMjFhZjlhOTEzYzNmNzAxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYyYmI4MDQzN2UzZjQzYjU5ZjNlYWEzZWQ4MTQ4OGJkLmJpbmRQb3B1cChwb3B1cF81ODVlMGNhNzdmZTI0NWU4OWE2ODUyNmI3OGNlOTdjNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mMzk3OGQxZTVhNjQ0OWE4YWQ4NGJjZDkwOTQ4YzRlZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQ2NzgxMTU4LDIyLjYwNzIwNDQ0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVhMzZjZGUyODM5OTQ4MjE4ZGY5ZDIxMTRmNTI0OGM0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRhOWQ2NDQxMGQ2MDRlMDk5ZDdhZDc5YWI1OWVkY2VhID0gJCgnPGRpdiBpZD0iaHRtbF80YTlkNjQ0MTBkNjA0ZTA5OWQ3YWQ3OWFiNTllZGNlYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpHOvc60z4HOr8+Ez4POsSwgzpTOl86czp/OoyDOm86VzqHOnc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVhMzZjZGUyODM5OTQ4MjE4ZGY5ZDIxMTRmNTI0OGM0LnNldENvbnRlbnQoaHRtbF80YTlkNjQ0MTBkNjA0ZTA5OWQ3YWQ3OWFiNTllZGNlYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMzk3OGQxZTVhNjQ0OWE4YWQ4NGJjZDkwOTQ4YzRlZS5iaW5kUG9wdXAocG9wdXBfNWEzNmNkZTI4Mzk5NDgyMThkZjlkMjExNGY1MjQ4YzQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTEyYzY5M2FmNmQxNDI5NGExNTI5MzdlNzFjZWQ5ODYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40ODI4NzU4MiwyMy4wMTAyMTE5NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lNjlhMTY5YjkwNmM0NzNmYTg0MDU2OGVlMTFlMWE3ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yNDUzNjkzMzZlMjk0N2YyOTllMzVkZTY2ZmMyMTYwMiA9ICQoJzxkaXYgaWQ9Imh0bWxfMjQ1MzY5MzM2ZTI5NDdmMjk5ZTM1ZGU2NmZjMjE2MDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Kz4HOuc6xLCDOlM6XzpzOn86jIM6RzqPOmc6dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTY5YTE2OWI5MDZjNDczZmE4NDA1NjhlZTExZTFhN2Uuc2V0Q29udGVudChodG1sXzI0NTM2OTMzNmUyOTQ3ZjI5OWUzNWRlNjZmYzIxNjAyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ExMmM2OTNhZjZkMTQyOTRhMTUyOTM3ZTcxY2VkOTg2LmJpbmRQb3B1cChwb3B1cF9lNjlhMTY5YjkwNmM0NzNmYTg0MDU2OGVlMTFlMWE3ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kOWUwYjhjM2M0NzU0ODgyODJlZDlhNDgzOTg5MDllZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQ5ODIzMzgsMjMuMDUxMTMyMTk5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZhMGQ2YjM2YmZkZDQxMjM4NGEzMjE3Y2I2NjNkN2ViID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc0NDNhMzczYjFhNDRiNmFhOGRjNzZjMjk2YzYwMmQ1ID0gJCgnPGRpdiBpZD0iaHRtbF83NDQzYTM3M2IxYTQ0YjZhYThkYzc2YzI5NmM2MDJkNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOsc+Bzr3Otc62zrHOr865zrrOsSwgzpTOl86czp/OoyDOkc6jzpnOnc6XzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZhMGQ2YjM2YmZkZDQxMjM4NGEzMjE3Y2I2NjNkN2ViLnNldENvbnRlbnQoaHRtbF83NDQzYTM3M2IxYTQ0YjZhYThkYzc2YzI5NmM2MDJkNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kOWUwYjhjM2M0NzU0ODgyODJlZDlhNDgzOTg5MDllZC5iaW5kUG9wdXAocG9wdXBfZmEwZDZiMzZiZmRkNDEyMzg0YTMyMTdjYjY2M2Q3ZWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjMwMTkwODliMjQ1NDc5YThkNTJmYWIwN2MwYzUwNmIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41MTY0OTQ3NSwyMi44NjUyODIwNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNDRjMDBhNWI1Y2Q0ZmQ5ODgwODA1NmZlZWFjYTExOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80ODE1YWExZDk3YzQ0NjkzYWQ4NmUxYTYzODFjZmYxMCA9ICQoJzxkaXYgaWQ9Imh0bWxfNDgxNWFhMWQ5N2M0NDY5M2FkODZlMWE2MzgxY2ZmMTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6azr/Pgc+Jzr3Ors+DzrksIM6UzpfOnM6fzqMgzpHOo86Zzp3Ol86jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jNDRjMDBhNWI1Y2Q0ZmQ5ODgwODA1NmZlZWFjYTExOS5zZXRDb250ZW50KGh0bWxfNDgxNWFhMWQ5N2M0NDY5M2FkODZlMWE2MzgxY2ZmMTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjMwMTkwODliMjQ1NDc5YThkNTJmYWIwN2MwYzUwNmIuYmluZFBvcHVwKHBvcHVwX2M0NGMwMGE1YjVjZDRmZDk4ODA4MDU2ZmVlYWNhMTE5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U3M2EzOGZmNGZmMTRlNThhZGMyZjZhNGI3NGIxMGE3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTIwMDExOSwyMi43MjkyNjMzMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNjA3N2E3Zjc2MzA0YjZhYTQ3NzdiNjFlYTY4Y2FmOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iN2FmMTk4YTgwZTc0Njg3OTMyMzA0NDIwYzc0ZDQ0MiA9ICQoJzxkaXYgaWQ9Imh0bWxfYjdhZjE5OGE4MGU3NDY4NzkzMjMwNDQyMGM3NGQ0NDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6azrnOss6tz4HOuc6/zr0sIM6UzpfOnM6fzqMgzpvOlc6hzp3Okc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNjA3N2E3Zjc2MzA0YjZhYTQ3NzdiNjFlYTY4Y2FmOC5zZXRDb250ZW50KGh0bWxfYjdhZjE5OGE4MGU3NDY4NzkzMjMwNDQyMGM3NGQ0NDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTczYTM4ZmY0ZmYxNGU1OGFkYzJmNmE0Yjc0YjEwYTcuYmluZFBvcHVwKHBvcHVwXzI2MDc3YTdmNzYzMDRiNmFhNDc3N2I2MWVhNjhjYWY4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FhYTQ1ZjM5YTU3MTQzYjk5NmIzNjJlNTA4YjNkMWE1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTIyNjQ3ODYsMjIuNTgwMzA4OTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGM3NmUzMTg4NjY1NDM5YjgyMWYxZDFlNTRmNTE1MmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGZkNGNlZWExMDYxNDRkNzlmYzRiMmFjYjJiOTcxMmYgPSAkKCc8ZGl2IGlkPSJodG1sXzBmZDRjZWVhMTA2MTQ0ZDc5ZmM0YjJhY2IyYjk3MTJmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc+HzrvOsc60z4zOus6xzrzPgM6/z4IsIM6azp/Omc6dzp/OpM6XzqTOkSDOkc6nzpvOkc6Uzp/Oms6RzpzOoM6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBjNzZlMzE4ODY2NTQzOWI4MjFmMWQxZTU0ZjUxNTJjLnNldENvbnRlbnQoaHRtbF8wZmQ0Y2VlYTEwNjE0NGQ3OWZjNGIyYWNiMmI5NzEyZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hYWE0NWYzOWE1NzE0M2I5OTZiMzYyZTUwOGIzZDFhNS5iaW5kUG9wdXAocG9wdXBfMGM3NmUzMTg4NjY1NDM5YjgyMWYxZDFlNTRmNTE1MmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2ZiMjU3MWI1NzUyNGJlNWIwOTIxY2JjZTRjY2QxNjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41MzgwODIxMiwyMi44OTAzMTIxOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wMTRmOWI3ZmFhZWE0MjY0YjE4NjM0MGI1ZjI4MWNmYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iYzVhMWQzOTQyMDQ0YzFmOGE5NTk4MTk3NTM0OWRlMSA9ICQoJzxkaXYgaWQ9Imh0bWxfYmM1YTFkMzk0MjA0NGMxZjhhOTU5ODE5NzUzNDlkZTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Uz4HOrc+AzrHOvc6/zr0sIM6UzpfOnM6fzqMgzpHOo86Zzp3Ol86jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMTRmOWI3ZmFhZWE0MjY0YjE4NjM0MGI1ZjI4MWNmYS5zZXRDb250ZW50KGh0bWxfYmM1YTFkMzk0MjA0NGMxZjhhOTU5ODE5NzUzNDlkZTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2ZiMjU3MWI1NzUyNGJlNWIwOTIxY2JjZTRjY2QxNjQuYmluZFBvcHVwKHBvcHVwXzAxNGY5YjdmYWFlYTQyNjRiMTg2MzQwYjVmMjgxY2ZhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y2N2M1NmZkOGJjYTQzMThiMjJmY2VmNGY0NGYwNGRkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTQzMDQxMjMsMjIuODYxNTM3OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmVhY2FmN2Q3ZjZiNDIzY2I4YjM2NzUxYmFjMDc5NTkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGEwYmJkNWNmODA3NGNhZjkyMjM5MTk2ZTUwYmE3OGYgPSAkKCc8ZGl2IGlkPSJodG1sXzBhMGJiZDVjZjgwNzRjYWY5MjIzOTE5NmU1MGJhNzhmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc+Dzq/Ovc63LCDOlM6XzpzOn86jIM6RzqPOmc6dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMmVhY2FmN2Q3ZjZiNDIzY2I4YjM2NzUxYmFjMDc5NTkuc2V0Q29udGVudChodG1sXzBhMGJiZDVjZjgwNzRjYWY5MjIzOTE5NmU1MGJhNzhmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y2N2M1NmZkOGJjYTQzMThiMjJmY2VmNGY0NGYwNGRkLmJpbmRQb3B1cChwb3B1cF8yZWFjYWY3ZDdmNmI0MjNjYjhiMzY3NTFiYWMwNzk1OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYWJmYjA5NzMyODU0MjljYTE4YzJjY2Y3NTk2MWZhZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU1NzMwMDU3LDIyLjg1OTA2NzkyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBlYzZlMTJlNDczOTRhZDdiNGQwOTQ5MWQ4NDE5ZjM1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzljOTZhODE2NjUzNzRjMzg5NjQ4ODEwMDU2ZmNjYzU5ID0gJCgnPGRpdiBpZD0iaHRtbF85Yzk2YTgxNjY1Mzc0YzM4OTY0ODgxMDA1NmZjY2M1OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpvOtc+FzrrOrM66zrnOsSwgzpTOl86czp/OoyDOnc6RzqXOoM6bzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wZWM2ZTEyZTQ3Mzk0YWQ3YjRkMDk0OTFkODQxOWYzNS5zZXRDb250ZW50KGh0bWxfOWM5NmE4MTY2NTM3NGMzODk2NDg4MTAwNTZmY2NjNTkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmFiZmIwOTczMjg1NDI5Y2ExOGMyY2NmNzU5NjFmYWQuYmluZFBvcHVwKHBvcHVwXzBlYzZlMTJlNDczOTRhZDdiNGQwOTQ5MWQ4NDE5ZjM1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIxNTc4YmRhMDNmOTRjOWJiNTkzZGI1OWZjNTFmYjFmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTYyOTY1MzksMjMuMTQ5NDc3MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfN2EwMzZlYmI2ZjZmNDU5ZmIxNjA2Y2ZjNmY3ZTljYWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTAxYjk3NjYxNzA0NGFiMWE3NzhkMzY0NzIwOGJhZTMgPSAkKCc8ZGl2IGlkPSJodG1sXzkwMWI5NzY2MTcwNDRhYjFhNzc4ZDM2NDcyMDhiYWUzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OpM+BzrHPh861zrnOrCwgzpTOl86czp/OoyDOlc6gzpnOlM6RzqXOoc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdhMDM2ZWJiNmY2ZjQ1OWZiMTYwNmNmYzZmN2U5Y2FjLnNldENvbnRlbnQoaHRtbF85MDFiOTc2NjE3MDQ0YWIxYTc3OGQzNjQ3MjA4YmFlMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMTU3OGJkYTAzZjk0YzliYjU5M2RiNTlmYzUxZmIxZi5iaW5kUG9wdXAocG9wdXBfN2EwMzZlYmI2ZjZmNDU5ZmIxNjA2Y2ZjNmY3ZTljYWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2NiOGRlZWRiOWY4NDBjZWFmN2JmMzBmMjQxMWJhYWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NjMxNjc1NywyMi42ODUwODcyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE4YWVhOTk1YTgyOTQzNGNiNTE2MjIzZmNkODBiYTg5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg0NzlkM2VlNzg5ZDQ4NzdiMDY2NzhlYzJhYTM4NzFiID0gJCgnPGRpdiBpZD0iaHRtbF84NDc5ZDNlZTc4OWQ0ODc3YjA2Njc4ZWMyYWEzODcxYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqPOus6xz4bOuc60zqzOus65zr/OvSwgzpTOl86czp/OoyDOm86VzqHOnc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE4YWVhOTk1YTgyOTQzNGNiNTE2MjIzZmNkODBiYTg5LnNldENvbnRlbnQoaHRtbF84NDc5ZDNlZTc4OWQ0ODc3YjA2Njc4ZWMyYWEzODcxYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zY2I4ZGVlZGI5Zjg0MGNlYWY3YmYzMGYyNDExYmFhZS5iaW5kUG9wdXAocG9wdXBfMThhZWE5OTVhODI5NDM0Y2I1MTYyMjNmY2Q4MGJhODkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2IwNjNmNThkZTIwNDUzOTlmNDFiMjMyYTE2OTYzN2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NjM5MTkwNywyMi43OTcwNTQyOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMjI5YzllMTk2YTU0MjdjYmRiNjg2OGRiZjcxYTRmYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80YTE2MTFjZjcyOWE0M2I2YTA0NDBkMTljMjQ5MmJmMiA9ICQoJzxkaXYgaWQ9Imh0bWxfNGExNjExY2Y3MjlhNDNiNmEwNDQwZDE5YzI0OTJiZjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6dzrHPjc+AzrvOuc6/zr0sIM6UzpfOnM6fzqMgzp3Okc6lzqDOm86Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTIyOWM5ZTE5NmE1NDI3Y2JkYjY4NjhkYmY3MWE0ZmEuc2V0Q29udGVudChodG1sXzRhMTYxMWNmNzI5YTQzYjZhMDQ0MGQxOWMyNDkyYmYyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NiMDYzZjU4ZGUyMDQ1Mzk5ZjQxYjIzMmExNjk2MzdhLmJpbmRQb3B1cChwb3B1cF8xMjI5YzllMTk2YTU0MjdjYmRiNjg2OGRiZjcxYTRmYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZjA4YmRjMjk4OGM0YTJmODEzY2E1NDc2NDIyNjhlNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU2ODM5NzUyLDIyLjgyODE0MjE3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E4MmYzZTJlMDg5NTRmMDU4NjhmNDFjYzNlYmNmZTNhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlmY2M3MDEwZmVlOTQ0YmQ4M2Y3NjAyMmY1YjJlMDI1ID0gJCgnPGRpdiBpZD0iaHRtbF85ZmNjNzAxMGZlZTk0NGJkODNmNzYwMjJmNWIyZTAyNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbPgc65zrEsIM6UzpfOnM6fzqMgzp3Okc6lzqDOm86Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTgyZjNlMmUwODk1NGYwNTg2OGY0MWNjM2ViY2ZlM2Euc2V0Q29udGVudChodG1sXzlmY2M3MDEwZmVlOTQ0YmQ4M2Y3NjAyMmY1YjJlMDI1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FmMDhiZGMyOTg4YzRhMmY4MTNjYTU0NzY0MjI2OGU3LmJpbmRQb3B1cChwb3B1cF9hODJmM2UyZTA4OTU0ZjA1ODY4ZjQxY2MzZWJjZmUzYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xYWUyMzA2YjJkZmE0YTNkYWQ5NTQzZjc0NTVkYjUxNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU4MDUwOTE5LDIyLjg3Nzc1NjEyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IyNTg5YTJiODAxZjQ0YzVhMjc5YTgwMjJjNmI5ZTQxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZiNTRlOWM5NDI3MDRiOTg5ZGY4YTFjNzA2YTlmNGZjID0gJCgnPGRpdiBpZD0iaHRtbF82YjU0ZTljOTQyNzA0Yjk4OWRmOGExYzcwNmE5ZjRmYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDPhc+BzrPOuc+Oz4TOuc66zrEsIM6UzpfOnM6fzqMgzp3Okc6lzqDOm86Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjI1ODlhMmI4MDFmNDRjNWEyNzlhODAyMmM2YjllNDEuc2V0Q29udGVudChodG1sXzZiNTRlOWM5NDI3MDRiOTg5ZGY4YTFjNzA2YTlmNGZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFhZTIzMDZiMmRmYTRhM2RhZDk1NDNmNzQ1NWRiNTE0LmJpbmRQb3B1cChwb3B1cF9iMjU4OWEyYjgwMWY0NGM1YTI3OWE4MDIyYzZiOWU0MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zNTJjMjY0NDViOTU0MTkxODg3ZTcyODNmMGJkYzE0MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU3ODc1MDYxLDIyLjUzMTUyNDY2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzIxNDJmYWQ4MGQ0NzRmMGQ5YmM3NzE5NDk2ODIyYzJkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg3NzJkYWI2YTRhZjQ5Y2RhM2VmODkyZDE5Zjg2OTYwID0gJCgnPGRpdiBpZD0iaHRtbF84NzcyZGFiNmE0YWY0OWNkYTNlZjg5MmQxOWY4Njk2MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprPgc+Fzr/Ovc6tz4HOuc6/zr0sIM6UzpfOnM6fzqMgzpHOoc6Tzp/Opc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yMTQyZmFkODBkNDc0ZjBkOWJjNzcxOTQ5NjgyMmMyZC5zZXRDb250ZW50KGh0bWxfODc3MmRhYjZhNGFmNDljZGEzZWY4OTJkMTlmODY5NjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzUyYzI2NDQ1Yjk1NDE5MTg4N2U3MjgzZjBiZGMxNDIuYmluZFBvcHVwKHBvcHVwXzIxNDJmYWQ4MGQ0NzRmMGQ5YmM3NzE5NDk2ODIyYzJkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzljZGVkM2QyMTIwMDQ1MmU4NjllYzNhYTY0OGU5NjdmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTg0NTE4NDMsMjIuNzQzNDc0OTZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTU4YjNkZjhhMzkyNDRmOGJkZWE4ZDMyMmM3YWUyMmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWE5MjZhZTI3ZGE1NDU1NGFjODNhZmU0MzE0MzBiNDkgPSAkKCc8ZGl2IGlkPSJodG1sX2VhOTI2YWUyN2RhNTQ1NTRhYzgzYWZlNDMxNDMwYjQ5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Onc6tzrEgzprOr86/z4IsIM6UzpfOnM6fzqMgzp3Olc6RzqMgzprOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U1OGIzZGY4YTM5MjQ0ZjhiZGVhOGQzMjJjN2FlMjJjLnNldENvbnRlbnQoaHRtbF9lYTkyNmFlMjdkYTU0NTU0YWM4M2FmZTQzMTQzMGI0OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85Y2RlZDNkMjEyMDA0NTJlODY5ZWMzYWE2NDhlOTY3Zi5iaW5kUG9wdXAocG9wdXBfZTU4YjNkZjhhMzkyNDRmOGJkZWE4ZDMyMmM3YWUyMmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTkzZDk2NTM5Y2NjNGVhYjkwNjk1MTYwMDQ3NTExNjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41OTUwMzkzNywyMi45NTQwMTM4Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zOGZlNTVjZDNkNmY0YWJiYmU1ZTVlNjVlNTgzNmU2NCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81OWY2ODYxZTRjZmY0ZWQ2OWZlMzliMTZkMzUwYTA2MyA9ICQoJzxkaXYgaWQ9Imh0bWxfNTlmNjg2MWU0Y2ZmNGVkNjlmZTM5YjE2ZDM1MGEwNjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Rz4HOus6xzrTOuc66z4zOvSwgzpTOl86czp/OoyDOkc6jzprOm86XzqDOmc6VzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zOGZlNTVjZDNkNmY0YWJiYmU1ZTVlNjVlNTgzNmU2NC5zZXRDb250ZW50KGh0bWxfNTlmNjg2MWU0Y2ZmNGVkNjlmZTM5YjE2ZDM1MGEwNjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTkzZDk2NTM5Y2NjNGVhYjkwNjk1MTYwMDQ3NTExNjYuYmluZFBvcHVwKHBvcHVwXzM4ZmU1NWNkM2Q2ZjRhYmJiZTVlNWU2NWU1ODM2ZTY0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk0Yjk5NWZmZmE2ZTRmYWZiMmYzYjIxOTQ2YTM3YmU3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTkzNTAyMDQsMjIuNzAwNDk4NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzA4ZDFjY2YzMGMxNDc4MWIxMGNhNzgzZjAyZDk0YzIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDAwMTc3NTZkMzJkNDNhZjg5N2ZhYjdkNTAyMWRjMjIgPSAkKCc8ZGl2IGlkPSJodG1sXzQwMDE3NzU2ZDMyZDQzYWY4OTdmYWI3ZDUwMjFkYzIyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms61z4bOsc67zqzPgc65zr/OvSwgzpTOl86czp/OoyDOkc6hzpPOn86lzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMwOGQxY2NmMzBjMTQ3ODFiMTBjYTc4M2YwMmQ5NGMyLnNldENvbnRlbnQoaHRtbF80MDAxNzc1NmQzMmQ0M2FmODk3ZmFiN2Q1MDIxZGMyMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NGI5OTVmZmZhNmU0ZmFmYjJmM2IyMTk0NmEzN2JlNy5iaW5kUG9wdXAocG9wdXBfMzA4ZDFjY2YzMGMxNDc4MWIxMGNhNzgzZjAyZDk0YzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTE3NDAwNDE4OWY1NDlkYTlkZTk0YjVjMTRhMjNjYmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41OTcyNDQyNiwyMi44NDM2NjQxN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMDI0YjQ4ZGI4ZjQ0MDNkOGMxZGQ4OGQ0MjRmOWM1ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YWQ0ZDFjMmYyNjI0ZWRhOTk2MzgzYjVkMjk0ZjcwOCA9ICQoJzxkaXYgaWQ9Imh0bWxfN2FkNGQxYzJmMjYyNGVkYTk5NjM4M2I1ZDI5NGY3MDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPsK2zrPOuc6/z4IgzpHOtM+BzrnOsc69z4zPgiwgzpTOl86czp/OoyDOnc6VzpHOoyDOpM6ZzqHOpc6dzpjOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMDI0YjQ4ZGI4ZjQ0MDNkOGMxZGQ4OGQ0MjRmOWM1Zi5zZXRDb250ZW50KGh0bWxfN2FkNGQxYzJmMjYyNGVkYTk5NjM4M2I1ZDI5NGY3MDgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTE3NDAwNDE4OWY1NDlkYTlkZTk0YjVjMTRhMjNjYmMuYmluZFBvcHVwKHBvcHVwX2QwMjRiNDhkYjhmNDQwM2Q4YzFkZDg4ZDQyNGY5YzVmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNmMjVmNzg5ZjBlNTQ2NDU5ZDVlZDE4YTIwNGZmNjFlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjAzNDY2MDMsMjIuOTQ0MDI2OTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDQxOTIzOGY0YmVkNGI0NDk0NmZkZjc2YjAxNGY1MGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGRkNDI1NjgyZjViNGI0ZmI0MGEyNTk5MTE3NTk5OTUgPSAkKCc8ZGl2IGlkPSJodG1sXzBkZDQyNTY4MmY1YjRiNGZiNDBhMjU5OTExNzU5OTk1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM61z4TPjM+HzrnOv869LCDOlM6XzpzOn86jIM6RzqPOms6bzpfOoM6ZzpXOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q0MTkyMzhmNGJlZDRiNDQ5NDZmZGY3NmIwMTRmNTBhLnNldENvbnRlbnQoaHRtbF8wZGQ0MjU2ODJmNWI0YjRmYjQwYTI1OTkxMTc1OTk5NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zZjI1Zjc4OWYwZTU0NjQ1OWQ1ZWQxOGEyMDRmZjYxZS5iaW5kUG9wdXAocG9wdXBfZDQxOTIzOGY0YmVkNGI0NDk0NmZkZjc2YjAxNGY1MGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTgwNzIzOTU2MDBhNGMzMWFhNWNjYjdiNWFmOGU4OTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MDUzMDg1MywyMi44MTgyNDg3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85YmE0YTg2ZDRiNzU0M2NkYjM3ZGIwYzM5MGQ0OTE2NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hYzc1NTdjZTgzOGY0NWY1YTlkNWFiZmMzOGMwMmI0MSA9ICQoJzxkaXYgaWQ9Imh0bWxfYWM3NTU3Y2U4MzhmNDVmNWE5ZDVhYmZjMzhjMDJiNDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6dzq3OsSDOpM6vz4HPhc69z4IsIM6UzpfOnM6fzqMgzp3Olc6RzqMgzqTOmc6hzqXOnc6YzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWJhNGE4NmQ0Yjc1NDNjZGIzN2RiMGMzOTBkNDkxNjYuc2V0Q29udGVudChodG1sX2FjNzU1N2NlODM4ZjQ1ZjVhOWQ1YWJmYzM4YzAyYjQxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU4MDcyMzk1NjAwYTRjMzFhYTVjY2I3YjVhZjhlODk0LmJpbmRQb3B1cChwb3B1cF85YmE0YTg2ZDRiNzU0M2NkYjM3ZGIwYzM5MGQ0OTE2Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MzUxMjViM2I2ZjE0OTcxOTllOThiYjg1MTA2NDNhMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYxMTE3MTcyLDIzLjAzNjk0NzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M3MmEwMmFkM2JkYTRkMTBiMTczMGEyMjc1MjFhNTRjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBhZjVkNzc5YjJlNzQzZjQ4MGU4YWNmOThjM2Q3NGQzID0gJCgnPGRpdiBpZD0iaHRtbF8wYWY1ZDc3OWIyZTc0M2Y0ODBlOGFjZjk4YzNkNzRkMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpvPhc6zzr/Pjc+BzrnOv869LCDOlM6XzpzOn86jIM6RzqPOms6bzpfOoM6ZzpXOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M3MmEwMmFkM2JkYTRkMTBiMTczMGEyMjc1MjFhNTRjLnNldENvbnRlbnQoaHRtbF8wYWY1ZDc3OWIyZTc0M2Y0ODBlOGFjZjk4YzNkNzRkMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MzUxMjViM2I2ZjE0OTcxOTllOThiYjg1MTA2NDNhMi5iaW5kUG9wdXAocG9wdXBfYzcyYTAyYWQzYmRhNGQxMGIxNzMwYTIyNzUyMWE1NGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzVhNmEzNGRlZTUyNGNjYjljNmFhMzVkN2Y0ZWFiODggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MDYyMjQwNiwyMi41OTY3OTIyMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wNjNiYjc3ZmQyNzY0YzM2ODE5ZWQyMTQyNzFjODM2ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yYTg0NTEyMzQ2OWQ0ODlhOGM4ZDQyMjc1ZTE3MTVlYSA9ICQoJzxkaXYgaWQ9Imh0bWxfMmE4NDUxMjM0NjlkNDg5YThjOGQ0MjI3NWUxNzE1ZWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6mz4HOrc6zzrrOsc65zr3OsSwgzpTOl86czp/OoyDOm86lzqHOms6VzpnOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNjNiYjc3ZmQyNzY0YzM2ODE5ZWQyMTQyNzFjODM2ZC5zZXRDb250ZW50KGh0bWxfMmE4NDUxMjM0NjlkNDg5YThjOGQ0MjI3NWUxNzE1ZWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzVhNmEzNGRlZTUyNGNjYjljNmFhMzVkN2Y0ZWFiODguYmluZFBvcHVwKHBvcHVwXzA2M2JiNzdmZDI3NjRjMzY4MTllZDIxNDI3MWM4MzZkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk3MDYxZjZlMmI0NzQ3NTZiZjE0MjI5OWNhYzE0YjkyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjEzMTAxOTYsMjIuODU2NjUxMzFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDJjOGMyMTkyM2NhNGViM2I2ZDk0ZTg2NWM3MmVhNjUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWIyNGYwYjk1ZmU2NDhhNWEyZTI3YWNmZjQxMjgxZTUgPSAkKCc8ZGl2IGlkPSJodG1sX2FiMjRmMGI5NWZlNjQ4YTVhMmUyN2FjZmY0MTI4MWU1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Onc6tzr/OvSDOoc6/zrXOuc69z4zOvSwgzpTOl86czp/OoyDOnc6VzpHOoyDOpM6ZzqHOpc6dzpjOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMmM4YzIxOTIzY2E0ZWIzYjZkOTRlODY1YzcyZWE2NS5zZXRDb250ZW50KGh0bWxfYWIyNGYwYjk1ZmU2NDhhNWEyZTI3YWNmZjQxMjgxZTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTcwNjFmNmUyYjQ3NDc1NmJmMTQyMjk5Y2FjMTRiOTIuYmluZFBvcHVwKHBvcHVwXzAyYzhjMjE5MjNjYTRlYjNiNmQ5NGU4NjVjNzJlYTY1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBmMDYyYzJlNzc4ODRmZmI5ZTg3OWFiMGY1YWNkM2Y4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjE1ODkwNTAwMDAwMDA2LDIyLjc2OTQ5ODgzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q1MjMwNjgxYjUxZDQxZmZiMWMzYWM1ODI4YjlkMjFkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2UyMWI4MjUxYTg0MzRkOWQ5NTVmMTdkYmM5NGU2ZTJkID0gJCgnPGRpdiBpZD0iaHRtbF9lMjFiODI1MWE4NDM0ZDlkOTU1ZjE3ZGJjOTRlNmUyZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpTOsc67zrHOvM6xzr3OrM+BzrEsIM6UzpfOnM6fzqMgzpHOoc6Tzp/Opc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNTIzMDY4MWI1MWQ0MWZmYjFjM2FjNTgyOGI5ZDIxZC5zZXRDb250ZW50KGh0bWxfZTIxYjgyNTFhODQzNGQ5ZDk1NWYxN2RiYzk0ZTZlMmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGYwNjJjMmU3Nzg4NGZmYjllODc5YWIwZjVhY2QzZjguYmluZFBvcHVwKHBvcHVwX2Q1MjMwNjgxYjUxZDQxZmZiMWMzYWM1ODI4YjlkMjFkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I0MmNiODNmYjY5MTQ2NTZiYmQ1Mzk4OTk1ZWM1NjJlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjI5MDYyNjUsMjIuNzgxMDAzOTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODIxNGM2ZGM4MThlNDExY2I3NGQ2MzE5ZTg5ODJkMGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzQyNTFmY2M0ZTg0NDk3MTk1OTJmMjcyZTM4OTQyYjMgPSAkKCc8ZGl2IGlkPSJodG1sX2M0MjUxZmNjNGU4NDQ5NzE5NTkyZjI3MmUzODk0MmIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Om86szrvOv8+FzrrOsc+CLCDOlM6XzpzOn86jIM6RzqHOk86fzqXOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODIxNGM2ZGM4MThlNDExY2I3NGQ2MzE5ZTg5ODJkMGEuc2V0Q29udGVudChodG1sX2M0MjUxZmNjNGU4NDQ5NzE5NTkyZjI3MmUzODk0MmIzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I0MmNiODNmYjY5MTQ2NTZiYmQ1Mzk4OTk1ZWM1NjJlLmJpbmRQb3B1cChwb3B1cF84MjE0YzZkYzgxOGU0MTFjYjc0ZDYzMTllODk4MmQwYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNjM5ZGFjOWUxYjA0NGRiYTQ2N2FkMTk4Yjg3Y2I3OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYzMDI5MDk5LDIyLjc2NTM2NTYwMDAwMDAwM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wMDE2OThkNTE5MDk0ZmU3YTc5YTRlZmEzNDNhODgyNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MDI2NzUxNzExMTE0OGNkYTlkMTExYjk1ZjA2ZDU5YSA9ICQoJzxkaXYgaWQ9Imh0bWxfNTAyNjc1MTcxMTExNDhjZGE5ZDExMWI5NWYwNmQ1OWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gz4XPgc6zzq3Ou867zrEsIM6UzpfOnM6fzqMgzpHOoc6Tzp/Opc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMDE2OThkNTE5MDk0ZmU3YTc5YTRlZmEzNDNhODgyNC5zZXRDb250ZW50KGh0bWxfNTAyNjc1MTcxMTExNDhjZGE5ZDExMWI5NWYwNmQ1OWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTYzOWRhYzllMWIwNDRkYmE0NjdhZDE5OGI4N2NiNzguYmluZFBvcHVwKHBvcHVwXzAwMTY5OGQ1MTkwOTRmZTdhNzlhNGVmYTM0M2E4ODI0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZiNDY4MDU0MjE4ODQwNDQ4YzIxYjg5MWIyZWI2ODE2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjM1MzMwMjAwMDAwMDA2LDIzLjE1MzUwMzQyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IxNTk5YTQ3YWI3NjRkNWI5NjNmYTI2Njk2ZTQ0Mzg1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M5NjlmMTE2NGUzZjQxZWNiNWU4N2IyZGQ3ZTU3NjA5ID0gJCgnPGRpdiBpZD0iaHRtbF9jOTY5ZjExNjRlM2Y0MWVjYjVlODdiMmRkN2U1NzYwOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpHPgc+HzrHOr86xIM6Vz4DOr860zrHPhc+Bzr/PgiwgzpTOl86czp/OoyDOlc6gzpnOlM6RzqXOoc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IxNTk5YTQ3YWI3NjRkNWI5NjNmYTI2Njk2ZTQ0Mzg1LnNldENvbnRlbnQoaHRtbF9jOTY5ZjExNjRlM2Y0MWVjYjVlODdiMmRkN2U1NzYwOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YjQ2ODA1NDIxODg0MDQ0OGMyMWI4OTFiMmViNjgxNi5iaW5kUG9wdXAocG9wdXBfYjE1OTlhNDdhYjc2NGQ1Yjk2M2ZhMjY2OTZlNDQzODUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjMyMDM1ZjdjYmIwNDUzYWE2YjI4Y2IxNzQzNTgxNTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MzIwMzQzLDIyLjcyNzc2NDEzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZiZmJmZDE3ODEzODRlZWRiN2U5ZDI5ODQwMjgwYjNhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzcyNWRmMzJhY2ZiZDQxNDZhMTUyMTFjYjI0ZjkxNmE5ID0gJCgnPGRpdiBpZD0iaHRtbF83MjVkZjMyYWNmYmQ0MTQ2YTE1MjExY2IyNGY5MTZhOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbPgc6zzr/PgiwgzpTOl86czp/OoyDOkc6hzpPOn86lzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZiZmJmZDE3ODEzODRlZWRiN2U5ZDI5ODQwMjgwYjNhLnNldENvbnRlbnQoaHRtbF83MjVkZjMyYWNmYmQ0MTQ2YTE1MjExY2IyNGY5MTZhOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MzIwMzVmN2NiYjA0NTNhYTZiMjhjYjE3NDM1ODE1NC5iaW5kUG9wdXAocG9wdXBfZmJmYmZkMTc4MTM4NGVlZGI3ZTlkMjk4NDAyODBiM2EpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDY4YzdhYWZjYmQxNGJmZWJkN2IwMWIyYmRhOTQ3YjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MzQyOTY0MiwyMi44MDQ3NzUyNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xZGFkMDFiYmM2MzI0NjhiYTNkNWZiZjQ4MTQzZmRkNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNmQ1MmY3YTIyYTU0NjBjYTFlZTY1ZDM0YzM3YzY0YSA9ICQoJzxkaXYgaWQ9Imh0bWxfMTZkNTJmN2EyMmE1NDYwY2ExZWU2NWQzNGMzN2M2NGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6RzrPOr86xIM6kz4HOuc6szrTOsSwgzpTOl86czp/OoyDOnM6ZzpTOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFkYWQwMWJiYzYzMjQ2OGJhM2Q1ZmJmNDgxNDNmZGQ2LnNldENvbnRlbnQoaHRtbF8xNmQ1MmY3YTIyYTU0NjBjYTFlZTY1ZDM0YzM3YzY0YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kNjhjN2FhZmNiZDE0YmZlYmQ3YjAxYjJiZGE5NDdiNy5iaW5kUG9wdXAocG9wdXBfMWRhZDAxYmJjNjMyNDY4YmEzZDVmYmY0ODE0M2ZkZDYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTFjNWZkMjEzMjAwNDlhMzg3ZTUxZjg4YWE4MTkzODYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42Mzg2OTQ3NiwyMi43NzA2NTg0OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MWQxMjdkNmYwN2Y0ZDUxOTViNjVkNTUzMGEwOTNmNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85MzdjYzcwZjRjNWE0MDA0ODYzMjc1ZTYyMDUzNGFiMiA9ICQoJzxkaXYgaWQ9Imh0bWxfOTM3Y2M3MGY0YzVhNDAwNDg2MzI3NWU2MjA1MzRhYjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6azr/Phc+Bz4TOrM66zrnOv869LCDOlM6XzpzOn86jIM6RzqHOk86fzqXOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTFkMTI3ZDZmMDdmNGQ1MTk1YjY1ZDU1MzBhMDkzZjcuc2V0Q29udGVudChodG1sXzkzN2NjNzBmNGM1YTQwMDQ4NjMyNzVlNjIwNTM0YWIyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ExYzVmZDIxMzIwMDQ5YTM4N2U1MWY4OGFhODE5Mzg2LmJpbmRQb3B1cChwb3B1cF81MWQxMjdkNmYwN2Y0ZDUxOTViNjVkNTUzMGEwOTNmNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85OTNkMzM5NDdlYmQ0ZTk4YWQ4MjM3ZWI0MGFiMzg0NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYzNjgyOTM4LDIyLjU0NDMyMjk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ0Yjc2Yzk2M2YzYTRmZTBhYWZmMGY4ZDUwNDVlMTA1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFiYjJlNGM0MzBkNDQ2Zjc4OGQzMWU2YTAwOGMyZTczID0gJCgnPGRpdiBpZD0iaHRtbF8xYmIyZTRjNDMwZDQ0NmY3ODhkMzFlNmEwMDhjMmU3MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOsc+Bz4XOrCwgzpTOl86czp/OoyDOm86lzqHOms6VzpnOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80NGI3NmM5NjNmM2E0ZmUwYWFmZjBmOGQ1MDQ1ZTEwNS5zZXRDb250ZW50KGh0bWxfMWJiMmU0YzQzMGQ0NDZmNzg4ZDMxZTZhMDA4YzJlNzMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTkzZDMzOTQ3ZWJkNGU5OGFkODIzN2ViNDBhYjM4NDcuYmluZFBvcHVwKHBvcHVwXzQ0Yjc2Yzk2M2YzYTRmZTBhYWZmMGY4ZDUwNDVlMTA1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNkNmM4NzAwNjg3MjQ5MDU5ZTE1NjU2YzZlZGVmMGJmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjQwMzk5OTMsMjIuNzg3NTI1MThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2Y1MzczMzEzM2Q0NGYwNmFjZGM0YmJjMDc1ZGQwMWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzZkNTBlY2EzYjVmNDFiNTlkNDA3OGYzNzdkMDRkZGUgPSAkKCc8ZGl2IGlkPSJodG1sXzM2ZDUwZWNhM2I1ZjQxYjU5ZDQwNzhmMzc3ZDA0ZGRlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Ol8+BzrHOr86/zr0sIM6UzpfOnM6fzqMgzpzOmc6UzpXOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jZjUzNzMzMTMzZDQ0ZjA2YWNkYzRiYmMwNzVkZDAxYS5zZXRDb250ZW50KGh0bWxfMzZkNTBlY2EzYjVmNDFiNTlkNDA3OGYzNzdkMDRkZGUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2Q2Yzg3MDA2ODcyNDkwNTllMTU2NTZjNmVkZWYwYmYuYmluZFBvcHVwKHBvcHVwX2NmNTM3MzMxMzNkNDRmMDZhY2RjNGJiYzA3NWRkMDFhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQyNDdjZTkwMjlmMTQwZmFiMjM5Mzc4NjlmYjcyZWRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjUxMDA0NzksMjIuNzYwMjEzODVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDJkY2FhMTFiZWNjNDM5MTlhYjY4MDdkMjc4NGMwNDQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGI4ZjY3NDlhNjkwNDAyYWExYjE1M2ZjOGEzODA1MzQgPSAkKCc8ZGl2IGlkPSJodG1sXzBiOGY2NzQ5YTY5MDQwMmFhMWIxNTNmYzhhMzgwNTM0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oic+BzrEsIM6UzpfOnM6fzqMgzpHOoc6Tzp/Opc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMmRjYWExMWJlY2M0MzkxOWFiNjgwN2QyNzg0YzA0NC5zZXRDb250ZW50KGh0bWxfMGI4ZjY3NDlhNjkwNDAyYWExYjE1M2ZjOGEzODA1MzQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDI0N2NlOTAyOWYxNDBmYWIyMzkzNzg2OWZiNzJlZGIuYmluZFBvcHVwKHBvcHVwX2QyZGNhYTExYmVjYzQzOTE5YWI2ODA3ZDI3ODRjMDQ0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ5NTVkZmJlN2QxMzQxYzI5NDAxMTIwZjI3NjczNzg2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjU1NjA1MzIsMjIuNzg4MzU4NjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmExYmIxNjE1MWIyNDYwMDk1MmRmNDk5ZWEyNTRmYTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGI5MTFmN2Y0NTBjNDU2NWJlYTAwNjdhMDk4OWMxZjggPSAkKCc8ZGl2IGlkPSJodG1sX2RiOTExZjdmNDUwYzQ1NjViZWEwMDY3YTA5ODljMWY4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc69z43Phs65zr/OvSwgzpTOl86czp/OoyDOnM6ZzpTOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JhMWJiMTYxNTFiMjQ2MDA5NTJkZjQ5OWVhMjU0ZmE4LnNldENvbnRlbnQoaHRtbF9kYjkxMWY3ZjQ1MGM0NTY1YmVhMDA2N2EwOTg5YzFmOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80OTU1ZGZiZTdkMTM0MWMyOTQwMTEyMGYyNzY3Mzc4Ni5iaW5kUG9wdXAocG9wdXBfYmExYmIxNjE1MWIyNDYwMDk1MmRmNDk5ZWEyNTRmYTgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTM2MTQxN2ViNDAzNGU3MDhiOTkzYjhlMzgwZjhiZjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NTM3NzQyNiwyMi42MDUzOTA1NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hMmYzNWMyN2ViZDA0ZTI5YmQ2MTgxNjNmYjYwYmZkNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNjZmYmNmYTlmMDY0OWU5OWQyMjE5NWM2MjFkY2IyYSA9ICQoJzxkaXYgaWQ9Imh0bWxfYTY2ZmJjZmE5ZjA2NDllOTlkMjIxOTVjNjIxZGNiMmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Sz4HOv8+Nz4PPhM65zr/OvSwgzpTOl86czp/OoyDOms6fzqXOpM6jzp/OoM6fzpTOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2EyZjM1YzI3ZWJkMDRlMjliZDYxODE2M2ZiNjBiZmQ1LnNldENvbnRlbnQoaHRtbF9hNjZmYmNmYTlmMDY0OWU5OWQyMjE5NWM2MjFkY2IyYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMzYxNDE3ZWI0MDM0ZTcwOGI5OTNiOGUzODBmOGJmNy5iaW5kUG9wdXAocG9wdXBfYTJmMzVjMjdlYmQwNGUyOWJkNjE4MTYzZmI2MGJmZDUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGY3ZGIxOGM5ODcyNDhmZjk5YzU5YjE4NGE3M2Q5NzkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NjE3ODEzMSwyMi43NDg1NzcxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85N2IyM2Y0OGNhZDA0YWNiODE4MTE1MzI4YTAzNDE3OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xMzBiMjY1MGJlYWU0YzAzYTMxMTg4Y2MwMzU4MDg1MyA9ICQoJzxkaXYgaWQ9Imh0bWxfMTMwYjI2NTBiZWFlNGMwM2EzMTE4OGNjMDM1ODA4NTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Kzr3Osc+Hzr/PgiwgzpTOl86czp/OoyDOkc6hzpPOn86lzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk3YjIzZjQ4Y2FkMDRhY2I4MTgxMTUzMjhhMDM0MTc5LnNldENvbnRlbnQoaHRtbF8xMzBiMjY1MGJlYWU0YzAzYTMxMTg4Y2MwMzU4MDg1Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZjdkYjE4Yzk4NzI0OGZmOTljNTliMTg0YTczZDk3OS5iaW5kUG9wdXAocG9wdXBfOTdiMjNmNDhjYWQwNGFjYjgxODExNTMyOGEwMzQxNzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmNlZjJiODEzMzYyNDNhMWIyNzBmYTRiZDM3NTY0YTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NjE1ODI5NSwyMi40ODA5NzIyOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83N2RlNjMxYWU3NmU0MWFiOWNhMTNhM2RkYjE2N2YxZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kZWUxZDQwNzlhMmQ0NGM4OTM1NDcyNDFlMmNiODg5NSA9ICQoJzxkaXYgaWQ9Imh0bWxfZGVlMWQ0MDc5YTJkNDRjODkzNTQ3MjQxZTJjYjg4OTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6dzrXOv8+Hz47Pgc65zr/OvSwgzpTOl86czp/OoyDOm86lzqHOms6VzpnOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83N2RlNjMxYWU3NmU0MWFiOWNhMTNhM2RkYjE2N2YxZC5zZXRDb250ZW50KGh0bWxfZGVlMWQ0MDc5YTJkNDRjODkzNTQ3MjQxZTJjYjg4OTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmNlZjJiODEzMzYyNDNhMWIyNzBmYTRiZDM3NTY0YTguYmluZFBvcHVwKHBvcHVwXzc3ZGU2MzFhZTc2ZTQxYWI5Y2ExM2EzZGRiMTY3ZjFkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UxODk3ZmVhMDBkNDQ5MGE5ZWI4MjEyMTAxMjJkNmZkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjY5MjI3NiwyMi43NzI2MTM1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xZWM1OTJjNTc5Mjc0YmZlOTYzN2IwMWRmMjRmMDgwNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NmY3NTU2YWZiNmM0MjY3YmUzODYxY2M3ODhhZTc4NCA9ICQoJzxkaXYgaWQ9Imh0bWxfNjZmNzU1NmFmYjZjNDI2N2JlMzg2MWNjNzg4YWU3ODQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6dzq3Ov869IM6Xz4HOsc6vzr/OvSwgzpTOl86czp/OoyDOnM6lzprOl86dzpHOmc6pzp08L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFlYzU5MmM1NzkyNzRiZmU5NjM3YjAxZGYyNGYwODA0LnNldENvbnRlbnQoaHRtbF82NmY3NTU2YWZiNmM0MjY3YmUzODYxY2M3ODhhZTc4NCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lMTg5N2ZlYTAwZDQ0OTBhOWViODIxMjEwMTIyZDZmZC5iaW5kUG9wdXAocG9wdXBfMWVjNTkyYzU3OTI3NGJmZTk2MzdiMDFkZjI0ZjA4MDQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWU1OTMzNjMxYTM2NDYyMmJlNDg4MTg5NTE2MjM0ZGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NzM2ODMxNywyMy4xMjcwNzEzOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iNzE3ZTFlNDIwZGY0Mzc0OTBjYmFhNTE5ODY3YjJmMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lZmUwNTI2YWY0ZGU0MmEyOThmMGJlZjZlOTdlNWFmOSA9ICQoJzxkaXYgaWQ9Imh0bWxfZWZlMDUyNmFmNGRlNDJhMjk4ZjBiZWY2ZTk3ZTVhZjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6dzq3OsSDOlc+Azq/OtM6xz4XPgc6/z4IsIM6UzpfOnM6fzqMgzpXOoM6ZzpTOkc6lzqHOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNzE3ZTFlNDIwZGY0Mzc0OTBjYmFhNTE5ODY3YjJmMS5zZXRDb250ZW50KGh0bWxfZWZlMDUyNmFmNGRlNDJhMjk4ZjBiZWY2ZTk3ZTVhZjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWU1OTMzNjMxYTM2NDYyMmJlNDg4MTg5NTE2MjM0ZGYuYmluZFBvcHVwKHBvcHVwX2I3MTdlMWU0MjBkZjQzNzQ5MGNiYWE1MTk4NjdiMmYxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U0NTk4YmJlMDE2NDQyNGJiYzg1YTk0M2EzY2Q0ZGUyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjc2MDk3ODcsMjIuOTU1OTA3ODJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTk5NDdiM2RiM2EyNDVmNjg3ZWExNzJhMzY4NTg0OWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmQ1NjU0YWVjYjU5NGRjOWFkOTcyY2IyYTFmZjdkMDggPSAkKCc8ZGl2IGlkPSJodG1sXzJkNTY1NGFlY2I1OTRkYzlhZDk3MmNiMmExZmY3ZDA4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc+BzrHPh869zrHOr86/zr0sIM6UzpfOnM6fzqMgzpzOmc6UzpXOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85OTk0N2IzZGIzYTI0NWY2ODdlYTE3MmEzNjg1ODQ5Zi5zZXRDb250ZW50KGh0bWxfMmQ1NjU0YWVjYjU5NGRjOWFkOTcyY2IyYTFmZjdkMDgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTQ1OThiYmUwMTY0NDI0YmJjODVhOTQzYTNjZDRkZTIuYmluZFBvcHVwKHBvcHVwXzk5OTQ3YjNkYjNhMjQ1ZjY4N2VhMTcyYTM2ODU4NDlmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Q0N2Y3NjliNWY5NzQ1OThiYjllNDUwNTk2NjJhMGI1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjc0ODQ2NjUsMjIuNTA3NjczMjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTZlM2EyMDIwYzMwNDY5ZDgwZGIxZjM1ODdlYzRiZGMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDE2MjYzNDU5MmMzNDg2ZTg2NmJlNTExNGVlODNlN2UgPSAkKCc8ZGl2IGlkPSJodG1sX2QxNjI2MzQ1OTJjMzQ4NmU4NjZiZTUxMTRlZTgzZTdlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6xz4DOsc+Bzq3Ou867zrnOv869LCDOlM6XzpzOn86jIM6bzqXOoc6azpXOmc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE2ZTNhMjAyMGMzMDQ2OWQ4MGRiMWYzNTg3ZWM0YmRjLnNldENvbnRlbnQoaHRtbF9kMTYyNjM0NTkyYzM0ODZlODY2YmU1MTE0ZWU4M2U3ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kNDdmNzY5YjVmOTc0NTk4YmI5ZTQ1MDU5NjYyYTBiNS5iaW5kUG9wdXAocG9wdXBfMTZlM2EyMDIwYzMwNDY5ZDgwZGIxZjM1ODdlYzRiZGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjA3NjJmYTRkMzQ3NDY5MDkwODdjMDUyMzU2ZDRmZjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NzYwNzQ5OCwyMi42MjE2NzE2OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZWNmYTA5NjQ0NzY0MmQ5YjJiNmQ3YzkwYjE0YjYzOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wNGMwNzQ0MDZhMWE0MjAxODkyNjhlNTJjODdlZjI1YSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDRjMDc0NDA2YTFhNDIwMTg5MjY4ZTUyYzg3ZWYyNWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6jz4fOuc69zr/Ph8+Oz4HOuc6/zr0sIM6UzpfOnM6fzqMgzprOn86lzqTOo86fzqDOn86UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZWNmYTA5NjQ0NzY0MmQ5YjJiNmQ3YzkwYjE0YjYzOC5zZXRDb250ZW50KGh0bWxfMDRjMDc0NDA2YTFhNDIwMTg5MjY4ZTUyYzg3ZWYyNWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjA3NjJmYTRkMzQ3NDY5MDkwODdjMDUyMzU2ZDRmZjYuYmluZFBvcHVwKHBvcHVwX2JlY2ZhMDk2NDQ3NjQyZDliMmI2ZDdjOTBiMTRiNjM4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NjMjgzMzg5NjdiYzQxNTNhYmQ2NzZmZGNjNmMwNmI2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjgwNTM0MzYsMjIuNzEzNjk5MzRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmJhOGEzMjI1OWExNDY3ZmI1M2NkNTkwM2FhMWFmYjggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTAxNzZjNzQzNTI2NDdlMDljNWQ5MWVhNzNkNjVhZTcgPSAkKCc8ZGl2IGlkPSJodG1sXzUwMTc2Yzc0MzUyNjQ3ZTA5YzVkOTFlYTczZDY1YWU3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6/z4XPhM+Dzr/PgM+MzrTOuc6/zr0sIM6UzpfOnM6fzqMgzprOn86lzqTOo86fzqDOn86UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82YmE4YTMyMjU5YTE0NjdmYjUzY2Q1OTAzYWExYWZiOC5zZXRDb250ZW50KGh0bWxfNTAxNzZjNzQzNTI2NDdlMDljNWQ5MWVhNzNkNjVhZTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2MyODMzODk2N2JjNDE1M2FiZDY3NmZkY2M2YzA2YjYuYmluZFBvcHVwKHBvcHVwXzZiYThhMzIyNTlhMTQ2N2ZiNTNjZDU5MDNhYTFhZmI4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc3Nzg3MjcxNzg0OTQzNDFiMmQ2OTczMmU1YjdlZTJlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjg5OTk4NjMsMjMuMDcwNDg3OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGUwODYxOWZlNmEzNGQ3MDhjZGM5NTY1NzAzMWZkOTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTYzZDJlM2FmMGE2NDBmOGFhYThlNzk2YTJlNDQwYWUgPSAkKCc8ZGl2IGlkPSJodG1sXzU2M2QyZTNhZjBhNjQwZjhhYWE4ZTc5NmEyZTQ0MGFlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OlM6uzrzOsc65zr3OsSwgzpTOl86czp/OoyDOlc6gzpnOlM6RzqXOoc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RlMDg2MTlmZTZhMzRkNzA4Y2RjOTU2NTcwMzFmZDkwLnNldENvbnRlbnQoaHRtbF81NjNkMmUzYWYwYTY0MGY4YWFhOGU3OTZhMmU0NDBhZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83Nzc4NzI3MTc4NDk0MzQxYjJkNjk3MzJlNWI3ZWUyZS5iaW5kUG9wdXAocG9wdXBfZGUwODYxOWZlNmEzNGQ3MDhjZGM5NTY1NzAzMWZkOTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzcyZGI4YTk2OTMxNDIwY2JkY2QwMzk3ZjU5MmYzNmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42OTY5MTA4NiwyMi40NzEzODc4Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84MTc1ZTE4YmY3OGU0NmJjYWRiYjI1OTFlMWVjNDg1MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wZTI4MzA4OTQ4YTg0NjQxYjBmNzZkNWFkMTc3ZTJkMCA9ICQoJzxkaXYgaWQ9Imh0bWxfMGUyODMwODk0OGE4NDY0MWIwZjc2ZDVhZDE3N2UyZDAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6azrXPhs6xzrvPjM6yz4HPhc+Dzr/OvSwgzpTOl86czp/OoyDOm86lzqHOms6VzpnOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84MTc1ZTE4YmY3OGU0NmJjYWRiYjI1OTFlMWVjNDg1MS5zZXRDb250ZW50KGh0bWxfMGUyODMwODk0OGE4NDY0MWIwZjc2ZDVhZDE3N2UyZDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzcyZGI4YTk2OTMxNDIwY2JkY2QwMzk3ZjU5MmYzNmUuYmluZFBvcHVwKHBvcHVwXzgxNzVlMThiZjc4ZTQ2YmNhZGJiMjU5MWUxZWM0ODUxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRjNWM2ZGQ2YjE0NDQyMjVhNThiMjhlYTk0NjJjMzE3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNzAyMTcxMzMsMjIuNzQ4MjAzMjhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzcyNDc3YmE1NTlmNDNiY2FjZjM5NGZmMTk2ZmQzNGIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGVjZjFkY2VmNDkwNGVlNzlhMzYxNTU5Njk1MTM5YmEgPSAkKCc8ZGl2IGlkPSJodG1sXzBlY2YxZGNlZjQ5MDRlZTc5YTM2MTU1OTY5NTEzOWJhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM6/zr3Osc+Dz4TOt8+BzqzOus65zr/OvSwgzpTOl86czp/OoyDOnM6lzprOl86dzpHOmc6pzp08L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM3MjQ3N2JhNTU5ZjQzYmNhY2YzOTRmZjE5NmZkMzRiLnNldENvbnRlbnQoaHRtbF8wZWNmMWRjZWY0OTA0ZWU3OWEzNjE1NTk2OTUxMzliYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80YzVjNmRkNmIxNDQ0MjI1YTU4YjI4ZWE5NDYyYzMxNy5iaW5kUG9wdXAocG9wdXBfMzcyNDc3YmE1NTlmNDNiY2FjZjM5NGZmMTk2ZmQzNGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTc5M2IzOWVkMjhjNGRhMzgwN2Q4YWE3MTU2ZjhjZGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy43MDE2MDI5NCwyMi41NDk1OTg2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNTE1Y2VkYWUwNjY0MWMyYmM5ZTcyZDhhNjMyMTE4ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNTJjYWJjM2U4YTg0MzY0ODE0NWYxMzMxNTY5MzRlZiA9ICQoJzxkaXYgaWQ9Imh0bWxfZTUyY2FiYzNlOGE4NDM2NDgxNDVmMTMzMTU2OTM0ZWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6bz43Pgc66zrXOuc6xLCDOlM6XzpzOn86jIM6bzqXOoc6azpXOmc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI1MTVjZWRhZTA2NjQxYzJiYzllNzJkOGE2MzIxMThlLnNldENvbnRlbnQoaHRtbF9lNTJjYWJjM2U4YTg0MzY0ODE0NWYxMzMxNTY5MzRlZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNzkzYjM5ZWQyOGM0ZGEzODA3ZDhhYTcxNTZmOGNkYi5iaW5kUG9wdXAocG9wdXBfMjUxNWNlZGFlMDY2NDFjMmJjOWU3MmQ4YTYzMjExOGUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2I2MGUxZTE4YTczNDA2MTlkMDFlMDMxNDIwYzI2ZDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy43MTA1NDA3NywyMi44NDAwOTE3MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83YmMyZDQwMjhlNjc0ZThmOGU5ZGVlYmMzNzY4YjhiZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80OWRhN2QzOGMzYzM0OTdiODQzMDk4YWYzYjhmZTU1MCA9ICQoJzxkaXYgaWQ9Imh0bWxfNDlkYTdkMzhjM2MzNDk3Yjg0MzA5OGFmM2I4ZmU1NTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gz4HPjM+Dz4XOvM69zrEsIM6UzpfOnM6fzqMgzpzOpc6azpfOnc6RzpnOqc6dPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83YmMyZDQwMjhlNjc0ZThmOGU5ZGVlYmMzNzY4YjhiZC5zZXRDb250ZW50KGh0bWxfNDlkYTdkMzhjM2MzNDk3Yjg0MzA5OGFmM2I4ZmU1NTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2I2MGUxZTE4YTczNDA2MTlkMDFlMDMxNDIwYzI2ZDMuYmluZFBvcHVwKHBvcHVwXzdiYzJkNDAyOGU2NzRlOGY4ZTlkZWViYzM3NjhiOGJkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI5ZDhmY2M4ZWIzZTRlY2JhYzdkNWI2ZDU2YmU2NWE4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNzEwMjA1MDgsMjIuODc4Nzg3OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTIyOGFmNjFmZjEyNGI3MGFhNjRjMjZiZDMyMzVjYjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDk1ZGJiOWM3NmZjNGIxNjkyYzk4OTY0NWU1NzNiMDggPSAkKCc8ZGl2IGlkPSJodG1sX2Q5NWRiYjljNzZmYzRiMTY5MmM5ODk2NDVlNTczYjA4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Om86vzrzOvc6xzrksIM6UzpfOnM6fzqMgzpzOpc6azpfOnc6RzpnOqc6dPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xMjI4YWY2MWZmMTI0YjcwYWE2NGMyNmJkMzIzNWNiMS5zZXRDb250ZW50KGh0bWxfZDk1ZGJiOWM3NmZjNGIxNjkyYzk4OTY0NWU1NzNiMDgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjlkOGZjYzhlYjNlNGVjYmFjN2Q1YjZkNTZiZTY1YTguYmluZFBvcHVwKHBvcHVwXzEyMjhhZjYxZmYxMjRiNzBhYTY0YzI2YmQzMjM1Y2IxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY5ZmZkZGY5NTM2NTRjNTJhMzVlNDI5ODA0NjAwM2U5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNzEwNTk0MTgsMjIuNDI0MTg4NjFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWU3ZjkyYzQ2ZDE5NGZiNzg5ODNkZTVhMjMwMzZhM2UgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGZmZWNiNWRjYmRkNGRkYTg3MTU1NmQ4N2FkZTZhNDggPSAkKCc8ZGl2IGlkPSJodG1sXzBmZmVjYjVkY2JkZDRkZGE4NzE1NTZkODdhZGU2YTQ4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Ops+Bzr/Phc+DzrnOv8+Nzr3OsSwgzprOn86Zzp3On86kzpfOpM6RIM6RzpvOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVlN2Y5MmM0NmQxOTRmYjc4OTgzZGU1YTIzMDM2YTNlLnNldENvbnRlbnQoaHRtbF8wZmZlY2I1ZGNiZGQ0ZGRhODcxNTU2ZDg3YWRlNmE0OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82OWZmZGRmOTUzNjU0YzUyYTM1ZTQyOTgwNDYwMDNlOS5iaW5kUG9wdXAocG9wdXBfNWU3ZjkyYzQ2ZDE5NGZiNzg5ODNkZTVhMjMwMzZhM2UpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTRjY2MyNWVkYmViNGNjOGFiZTk5ZmZiZmVkNmI5YjAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy43MTcwNjc3MiwyMi43NDM5MjMxOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80OTQ5MzA1ZjQ3ZDE0NTEzOTJkNDA4YmI5Y2NiZWMwNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81ZTVjYTU2Y2RlZWE0ZmM1YWFjOWI3NzlmMTUxNTU1MSA9ICQoJzxkaXYgaWQ9Imh0bWxfNWU1Y2E1NmNkZWVhNGZjNWFhYzliNzc5ZjE1MTU1NTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6cz4XOus6uzr3Osc65LCDOlM6XzpzOn86jIM6czqXOms6Xzp3Okc6ZzqnOnTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDk0OTMwNWY0N2QxNDUxMzkyZDQwOGJiOWNjYmVjMDQuc2V0Q29udGVudChodG1sXzVlNWNhNTZjZGVlYTRmYzVhYWM5Yjc3OWYxNTE1NTUxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU0Y2NjMjVlZGJlYjRjYzhhYmU5OWZmYmZlZDZiOWIwLmJpbmRQb3B1cChwb3B1cF80OTQ5MzA1ZjQ3ZDE0NTEzOTJkNDA4YmI5Y2NiZWMwNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNmI3YTA5YzlhZGE0ZDhjYTM2YjA1NDI0NTI4MTgwNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjcxNzI1ODQ1LDIyLjU5NTU2NzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODA5NmYxNDljZWEwNDNlYzliZmU0MmViMmRmYWFiZDIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzliMTNiODAzMzljNDEwZTk5YzhhMzk0YWFmMzBmYmEgPSAkKCc8ZGl2IGlkPSJodG1sX2M5YjEzYjgwMzM5YzQxMGU5OWM4YTM5NGFhZjMwZmJhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oo8+Ezq3Pgc69zrEsIM6UzpfOnM6fzqMgzpvOpc6hzprOlc6ZzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODA5NmYxNDljZWEwNDNlYzliZmU0MmViMmRmYWFiZDIuc2V0Q29udGVudChodG1sX2M5YjEzYjgwMzM5YzQxMGU5OWM4YTM5NGFhZjMwZmJhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q2YjdhMDljOWFkYTRkOGNhMzZiMDU0MjQ1MjgxODA2LmJpbmRQb3B1cChwb3B1cF84MDk2ZjE0OWNlYTA0M2VjOWJmZTQyZWIyZGZhYWJkMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iYzJlM2YzYjU3ODE0M2U1OGI3MDNmYTZmMTEwYWJkMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjcyMDU1MDU0LDIyLjcyMzI2Mjc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzIwOTNkNDczNjJkMTQ4NzRiZGI5MGE5ZThlZjFjOWVjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJiZmY2NmZkMWMwZjRiMjJhMjc0NGNiNTYzOTU1NzE4ID0gJCgnPGRpdiBpZD0iaHRtbF8yYmZmNjZmZDFjMGY0YjIyYTI3NDRjYjU2Mzk1NTcxOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqbOr8+Hz4TOuc6/zr0sIM6UzpfOnM6fzqMgzpzOpc6azpfOnc6RzpnOqc6dPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yMDkzZDQ3MzYyZDE0ODc0YmRiOTBhOWU4ZWYxYzllYy5zZXRDb250ZW50KGh0bWxfMmJmZjY2ZmQxYzBmNGIyMmEyNzQ0Y2I1NjM5NTU3MTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmMyZTNmM2I1NzgxNDNlNThiNzAzZmE2ZjExMGFiZDIuYmluZFBvcHVwKHBvcHVwXzIwOTNkNDczNjJkMTQ4NzRiZGI5MGE5ZThlZjFjOWVjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzczOWVkYjgwMmNkMDRlYTRhNmI2M2JmMjI0NjRmOTVkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNzIzODY1NTEsMjIuNjM4NTY2OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmMwZjQ1ZDFhMmVhNDQ0NTk3MTU1OGQ0ZDJhZTc4OWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzgyZTRjNGIyNzdmNGQ4ZThjZGU3ZWY0MGY3NDQzODAgPSAkKCc8ZGl2IGlkPSJodG1sXzc4MmU0YzRiMjc3ZjRkOGU4Y2RlN2VmNDBmNzQ0MzgwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM6xzrvOsc69z4TPgc6tzr3Ouc6/zr0sIM6UzpfOnM6fzqMgzprOn86lzqTOo86fzqDOn86UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mYzBmNDVkMWEyZWE0NDQ1OTcxNTU4ZDRkMmFlNzg5YS5zZXRDb250ZW50KGh0bWxfNzgyZTRjNGIyNzdmNGQ4ZThjZGU3ZWY0MGY3NDQzODApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzM5ZWRiODAyY2QwNGVhNGE2YjYzYmYyMjQ2NGY5NWQuYmluZFBvcHVwKHBvcHVwX2ZjMGY0NWQxYTJlYTQ0NDU5NzE1NThkNGQyYWU3ODlhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM4MGY5YTFhMzYxYzRkYjI5MjFiMTE0OTFjYWZjMTBkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNzI3NTAwOTIsMjIuNjgyMzc0OTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWU1Y2Q0MGEwNzAwNGZmYWJhM2I2YjRiZjJhYzk2NzkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjgyYjgyZGVjODkxNDU3YzhmMThhZWQxZTNmZWYyNDcgPSAkKCc8ZGl2IGlkPSJodG1sXzI4MmI4MmRlYzg5MTQ1N2M4ZjE4YWVkMWUzZmVmMjQ3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM+Az4zPgc+DzrHPgiwgzpTOl86czp/OoyDOnM6lzprOl86dzpHOmc6pzp08L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VlNWNkNDBhMDcwMDRmZmFiYTNiNmI0YmYyYWM5Njc5LnNldENvbnRlbnQoaHRtbF8yODJiODJkZWM4OTE0NTdjOGYxOGFlZDFlM2ZlZjI0Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zODBmOWExYTM2MWM0ZGIyOTIxYjExNDkxY2FmYzEwZC5iaW5kUG9wdXAocG9wdXBfZWU1Y2Q0MGEwNzAwNGZmYWJhM2I2YjRiZjJhYzk2NzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmM0YTBmNjMzYWUzNDZmOGE2NjdhZmU4YzA1NWQ1NDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy43NDkyOTQyOCwyMi40MjQ4OTA1Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lY2QzZWIyZGE4ZDc0YmEwODBmODU2ZjljNDU0MTBmZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YmUxNWIzYWE2MGQ0YzE4YmIxMDVjYjRlNGI3M2E3MiA9ICQoJzxkaXYgaWQ9Imh0bWxfN2JlMTViM2FhNjBkNGMxOGJiMTA1Y2I0ZTRiNzNhNzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6RzrvOrc6xLCDOms6fzpnOnc6fzqTOl86kzpEgzpHOm86VzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWNkM2ViMmRhOGQ3NGJhMDgwZjg1NmY5YzQ1NDEwZmUuc2V0Q29udGVudChodG1sXzdiZTE1YjNhYTYwZDRjMThiYjEwNWNiNGU0YjczYTcyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZjNGEwZjYzM2FlMzQ2ZjhhNjY3YWZlOGMwNTVkNTQ1LmJpbmRQb3B1cChwb3B1cF9lY2QzZWIyZGE4ZDc0YmEwODBmODU2ZjljNDU0MTBmZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84YmQzNDI4MmVlNjk0MmNjYjcxMDAxMWYyNDdiZTA1NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3Ljc3MzA0NDU5LDIyLjU3MjIwNjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmU0YmM4MWI3MDgxNGMwZWI3NjFkZmE2OWQwOWZlZmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDM3MTgxM2VlNzk1NDcwYmJiMmQ3OGIyMGI1YTEzZjAgPSAkKCc8ZGl2IGlkPSJodG1sXzQzNzE4MTNlZTc5NTQ3MGJiYjJkNzhiMjBiNWExM2YwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Ok8+FzrzOvc+Mzr0sIM6UzpfOnM6fzqMgzpvOpc6hzprOlc6ZzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMmU0YmM4MWI3MDgxNGMwZWI3NjFkZmE2OWQwOWZlZmYuc2V0Q29udGVudChodG1sXzQzNzE4MTNlZTc5NTQ3MGJiYjJkNzhiMjBiNWExM2YwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhiZDM0MjgyZWU2OTQyY2NiNzEwMDExZjI0N2JlMDU2LmJpbmRQb3B1cChwb3B1cF8yZTRiYzgxYjcwODE0YzBlYjc2MWRmYTY5ZDA5ZmVmZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYzQyY2Q4OTdmNjE0MDU5YTE5YWExNGIyM2ZkODQ4OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3Ljc4NTI4NTk1LDIyLjQzNDg3MzU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUwYTIxYzMzODNmOTQwODI4NDg1NmE1ODVkNjhhYjg2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2QwOGI0NzY4MzI5MDRmNzY4YjE1ZTc3MWFiOTkyYTI3ID0gJCgnPGRpdiBpZD0iaHRtbF9kMDhiNDc2ODMyOTA0Zjc2OGIxNWU3NzFhYjk5MmEyNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqPOus6/z4TOtc65zr3OriwgzprOn86Zzp3On86kzpfOpM6RIM6RzpvOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUwYTIxYzMzODNmOTQwODI4NDg1NmE1ODVkNjhhYjg2LnNldENvbnRlbnQoaHRtbF9kMDhiNDc2ODMyOTA0Zjc2OGIxNWU3NzFhYjk5MmEyNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wYzQyY2Q4OTdmNjE0MDU5YTE5YWExNGIyM2ZkODQ4OS5iaW5kUG9wdXAocG9wdXBfNTBhMjFjMzM4M2Y5NDA4Mjg0ODU2YTU4NWQ2OGFiODYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzFhYmFkMDEzODJlNGY5MGI2OWRmNjBhZGZiMWM0ZTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy44MTE2MzQwNiwyMi41MjY5NTI3NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MjI5ZThjNDhjYzU0ZGZkYjNhOTQwYTA2ZDZmODZjYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xMzRhNWRkYmY5OWE0ZWU2OWJhYzRkYmZmOGM5NjMxMCA9ICQoJzxkaXYgaWQ9Imh0bWxfMTM0YTVkZGJmOTlhNGVlNjliYWM0ZGJmZjhjOTYzMTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gzrvOsc+EzqzOvc65zr/OvSwgzprOn86Zzp3On86kzpfOpM6RIM6RzpvOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYyMjllOGM0OGNjNTRkZmRiM2E5NDBhMDZkNmY4NmNiLnNldENvbnRlbnQoaHRtbF8xMzRhNWRkYmY5OWE0ZWU2OWJhYzRkYmZmOGM5NjMxMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zMWFiYWQwMTM4MmU0ZjkwYjY5ZGY2MGFkZmIxYzRlNS5iaW5kUG9wdXAocG9wdXBfNjIyOWU4YzQ4Y2M1NGRmZGIzYTk0MGEwNmQ2Zjg2Y2IpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzM5YzE1YTk1MzBjNGIwNGE4ZjQxYmE5M2QwMjIyODQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NDEzNTM2MSwyMi44ODA3NTA2Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83YmUxMWQyODg1MTQ0NWEwYWE2YzkzNDcwYzU3NTAyYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kZWQzZTc4ZTQ5YjI0OWI1OGYzZWYzZjA4ZDFjNmNlYiA9ICQoJzxkaXYgaWQ9Imh0bWxfZGVkM2U3OGU0OWIyNDliNThmM2VmM2YwOGQxYzZjZWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6RzrzOsc+BzrnOsc69z4zPgiwgzpTOl86czp/OoyDOnM6ZzpTOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdiZTExZDI4ODUxNDQ1YTBhYTZjOTM0NzBjNTc1MDJjLnNldENvbnRlbnQoaHRtbF9kZWQzZTc4ZTQ5YjI0OWI1OGYzZWYzZjA4ZDFjNmNlYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMzljMTVhOTUzMGM0YjA0YThmNDFiYTkzZDAyMjI4NC5iaW5kUG9wdXAocG9wdXBfN2JlMTFkMjg4NTE0NDVhMGFhNmM5MzQ3MGM1NzUwMmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWVmNmVhYzFmOWQ2NDRhODkxMDNkNjhiZTQ3Mjc4YWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NzE3OTQ4OSwyMi43NDI3OTIxM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85YjA4YTkxMmI0OGY0NDBmOWE2ZWM3ZmI5NjQwZjE4MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zMzQwMTJmZTZlYzM0MGM4ODJiMWZjODIzMTMwMzJjOCA9ICQoJzxkaXYgaWQ9Imh0bWxfMzM0MDEyZmU2ZWMzNDBjODgyYjFmYzgyMzEzMDMyYzgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6kz4HOr8+Dz4TPgc6xz4TOv869LCDOlM6XzpzOn86jIM6RzqHOk86fzqXOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWIwOGE5MTJiNDhmNDQwZjlhNmVjN2ZiOTY0MGYxODIuc2V0Q29udGVudChodG1sXzMzNDAxMmZlNmVjMzQwYzg4MmIxZmM4MjMxMzAzMmM4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFlZjZlYWMxZjlkNjQ0YTg5MTAzZDY4YmU0NzI3OGFhLmJpbmRQb3B1cChwb3B1cF85YjA4YTkxMmI0OGY0NDBmOWE2ZWM3ZmI5NjQwZjE4Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yMzAxMzU4ZDEyYTk0ODUwYTAwMGQ5YWFiZjFhYjE5MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjY3MzY1MjY1LDIyLjgyNDI4MzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWQ5ZGNmNjkwYjQ3NDBjMzhlOWY0MWZlYzkyOTJjNmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDMzNGQwZDczMDFlNGQ2ZmE1ZWQ2YTc4NjQ4ODcwNmQgPSAkKCc8ZGl2IGlkPSJodG1sX2QzMzRkMGQ3MzAxZTRkNmZhNWVkNmE3ODY0ODg3MDZkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM61z4TPjM+HzrnOv869LCDOlM6XzpzOn86jIM6czpnOlM6VzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNWQ5ZGNmNjkwYjQ3NDBjMzhlOWY0MWZlYzkyOTJjNmIuc2V0Q29udGVudChodG1sX2QzMzRkMGQ3MzAxZTRkNmZhNWVkNmE3ODY0ODg3MDZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIzMDEzNThkMTJhOTQ4NTBhMDAwZDlhYWJmMWFiMTkyLmJpbmRQb3B1cChwb3B1cF81ZDlkY2Y2OTBiNDc0MGMzOGU5ZjQxZmVjOTI5MmM2Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80NTA4NTUyOTNmZTA0NzhmYWNkNjgwYWE2MTdiZjVjNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjY1NzY3NjcsMjIuODIzNDU1ODFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzczYmQ0MTM1ZWQzNDJmNGE2OWRlNjRlYmNkNjczOTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmRkNTMyOWI1OTUzNDM2ZDhiNDBjZDJiOWQzMWYyYWMgPSAkKCc8ZGl2IGlkPSJodG1sX2ZkZDUzMjliNTk1MzQzNmQ4YjQwY2QyYjlkMzFmMmFjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM6szr3Otc+DzrfPgiwgzpTOl86czp/OoyDOnM6ZzpTOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M3M2JkNDEzNWVkMzQyZjRhNjlkZTY0ZWJjZDY3MzkwLnNldENvbnRlbnQoaHRtbF9mZGQ1MzI5YjU5NTM0MzZkOGI0MGNkMmI5ZDMxZjJhYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NTA4NTUyOTNmZTA0NzhmYWNkNjgwYWE2MTdiZjVjNS5iaW5kUG9wdXAocG9wdXBfYzczYmQ0MTM1ZWQzNDJmNGE2OWRlNjRlYmNkNjczOTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTJmMGUyNzFhYzAwNDc4MjgwODgyNjk0ZWFmYjkwZTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NDY5MjY4OCwyMi44MTMwNDU0OTk5OTk5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTVjZjU0MWQ2NzdmNGEzZTg0ZjUzNjA0NzA1NGIzN2YgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDJkYmJjMzdiMTUzNGNiZjg4YTQ1M2Y4MTM3M2E5YjggPSAkKCc8ZGl2IGlkPSJodG1sXzAyZGJiYzM3YjE1MzRjYmY4OGE0NTNmODEzNzNhOWI4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OoM6/z4XOu867zrHOus6vzrTOsSwgzpTOl86czp/OoyDOnM6ZzpTOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E1Y2Y1NDFkNjc3ZjRhM2U4NGY1MzYwNDcwNTRiMzdmLnNldENvbnRlbnQoaHRtbF8wMmRiYmMzN2IxNTM0Y2JmODhhNDUzZjgxMzczYTliOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMmYwZTI3MWFjMDA0NzgyODA4ODI2OTRlYWZiOTBlMi5iaW5kUG9wdXAocG9wdXBfYTVjZjU0MWQ2NzdmNGEzZTg0ZjUzNjA0NzA1NGIzN2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjlkZTM4MDQ0Mjc3NGE0ZDk1MzkzZDQwMmY2MGE2ZjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NzgyMjI2NiwyMi44MzQ3MjQ0M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84Nzk1YmRiNTVhNzQ0ZDAwYmQ2YzA4NGJjNTViNGQzMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zMDA2MWQ3MjEyOWQ0OWY5YWEzZWI1ZTgzYjBhMmNkMyA9ICQoJzxkaXYgaWQ9Imh0bWxfMzAwNjFkNzIxMjlkNDlmOWFhM2ViNWU4M2IwYTJjZDMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6RzrzPhc6zzrTOsc67zq/PhM+DzrEsIM6UzpfOnM6fzqMgzpzOmc6UzpXOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84Nzk1YmRiNTVhNzQ0ZDAwYmQ2YzA4NGJjNTViNGQzMi5zZXRDb250ZW50KGh0bWxfMzAwNjFkNzIxMjlkNDlmOWFhM2ViNWU4M2IwYTJjZDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjlkZTM4MDQ0Mjc3NGE0ZDk1MzkzZDQwMmY2MGE2ZjYuYmluZFBvcHVwKHBvcHVwXzg3OTViZGI1NWE3NDRkMDBiZDZjMDg0YmM1NWI0ZDMyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzlhODE3NWNlNThlNzRkMTFiZTBlN2M1YWQ0MGNiZjA5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjU4NzE0MjksMjIuODQxNTA4ODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTNhY2Q3MjU0MzNkNDljYmIyZWFhNGZiMzllYzgwYzcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWYzNjc0NTFjYTczNGY3MDhjMTdiZWQxNzIwYjFkYmUgPSAkKCc8ZGl2IGlkPSJodG1sX2FmMzY3NDUxY2E3MzRmNzA4YzE3YmVkMTcyMGIxZGJlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM65zrTOrc6xLCDOlM6XzpzOn86jIM6czpnOlM6VzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTNhY2Q3MjU0MzNkNDljYmIyZWFhNGZiMzllYzgwYzcuc2V0Q29udGVudChodG1sX2FmMzY3NDUxY2E3MzRmNzA4YzE3YmVkMTcyMGIxZGJlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzlhODE3NWNlNThlNzRkMTFiZTBlN2M1YWQ0MGNiZjA5LmJpbmRQb3B1cChwb3B1cF9hM2FjZDcyNTQzM2Q0OWNiYjJlYWE0ZmIzOWVjODBjNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZjg2OTFiMjc3YmQ0OGQzODRkNmEzMTIzNWNlMDAyOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYzNDc3NzA3LDIyLjkyNTMwNDQxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYxYTEyNjY2YTE3NTRhMmVhODI4ODEyYTFkMTNmYzJhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk1ODMzMjljMWY4NTRjNjBhYTZlNjM3MDdiMjU4ZTA2ID0gJCgnPGRpdiBpZD0iaHRtbF85NTgzMzI5YzFmODU0YzYwYWE2ZTYzNzA3YjI1OGUwNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpPOus6sz4TOts65zrEsIM6UzpfOnM6fzqMgzpHOo86azpvOl86gzpnOlc6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjFhMTI2NjZhMTc1NGEyZWE4Mjg4MTJhMWQxM2ZjMmEuc2V0Q29udGVudChodG1sXzk1ODMzMjljMWY4NTRjNjBhYTZlNjM3MDdiMjU4ZTA2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VmODY5MWIyNzdiZDQ4ZDM4NGQ2YTMxMjM1Y2UwMDI4LmJpbmRQb3B1cChwb3B1cF82MWExMjY2NmExNzU0YTJlYTgyODgxMmExZDEzZmMyYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hYjM5YTk4ZWM0NzA0NTI2OGQzNGYyMTA5NTQ3MDVjOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUzNTMzNTU0LDIyLjg3ODM1MTIxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IyYzY2M2Y5MzA0NDQ5NWY4NWZmNzI2ZDlhMDI2ZGUwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzliNGVhYzNkYWI5OTQ3YmRhZDhhYjZiOWVmZTAzNDA3ID0gJCgnPGRpdiBpZD0iaHRtbF85YjRlYWMzZGFiOTk0N2JkYWQ4YWI2YjllZmUwMzQwNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOsc67zrvOuc64zq3OsSwgzpTOl86czp/OoyDOkc6jzpnOnc6XzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IyYzY2M2Y5MzA0NDQ5NWY4NWZmNzI2ZDlhMDI2ZGUwLnNldENvbnRlbnQoaHRtbF85YjRlYWMzZGFiOTk0N2JkYWQ4YWI2YjllZmUwMzQwNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hYjM5YTk4ZWM0NzA0NTI2OGQzNGYyMTA5NTQ3MDVjOC5iaW5kUG9wdXAocG9wdXBfYjJjNjYzZjkzMDQ0NDk1Zjg1ZmY3MjZkOWEwMjZkZTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGI0NzM1MWMxMmJlNGY2NDg0MzE2YjllMjQ4ZGJjZjkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41MzY1NTYyNCwyMi45MTA4OTA1OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMjIzYjgzNWI4OTg0OTZjYmExMjZkNjIwMjk3YWQ1YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZWEwMmEzNGI0Zjc0ZTRkYTYxMzJhOTE0MWM3MTFmNCA9ICQoJzxkaXYgaWQ9Imh0bWxfZmVhMDJhMzRiNGY3NGU0ZGE2MTMyYTkxNDFjNzExZjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6dzq3OsSDOnM6xz4HOsc64zq3OsSwgzpTOl86czp/OoyDOkc6jzpnOnc6XzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMyMjNiODM1Yjg5ODQ5NmNiYTEyNmQ2MjAyOTdhZDVjLnNldENvbnRlbnQoaHRtbF9mZWEwMmEzNGI0Zjc0ZTRkYTYxMzJhOTE0MWM3MTFmNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84YjQ3MzUxYzEyYmU0ZjY0ODQzMTZiOWUyNDhkYmNmOS5iaW5kUG9wdXAocG9wdXBfMzIyM2I4MzViODk4NDk2Y2JhMTI2ZDYyMDI5N2FkNWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmJlZWM1YzI0MjdhNDUxYmFiYTUwMmRjNjQ5NjMwNWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NDcwOTYyNSwyMi45MjQzOTQ2MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMDkxZjU2NzA4ZTg0NWIwOTkwOWVmMTFmM2QwNzZhZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMTg1MGNhZTI1Yzc0MTAxYjcyY2RmNWEzMDUwMzEyZSA9ICQoJzxkaXYgaWQ9Imh0bWxfZjE4NTBjYWUyNWM3NDEwMWI3MmNkZjVhMzA1MDMxMmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6czrHPgc6xzrjOrc6xLCDOlM6XzpzOn86jIM6RzqPOmc6dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDA5MWY1NjcwOGU4NDViMDk5MDllZjExZjNkMDc2YWQuc2V0Q29udGVudChodG1sX2YxODUwY2FlMjVjNzQxMDFiNzJjZGY1YTMwNTAzMTJlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZiZWVjNWMyNDI3YTQ1MWJhYmE1MDJkYzY0OTYzMDViLmJpbmRQb3B1cChwb3B1cF9kMDkxZjU2NzA4ZTg0NWIwOTkwOWVmMTFmM2QwNzZhZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNWFjMzhmMWJkZDY0YTQ5OGMxYTM0ZGU3ZWU3OGNkMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUzNDc2MzM0LDIyLjkxNDk0MTc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ3ZThhZDFkM2EwZDRhYzU4OTJhMzNiMDQzMjQ4ZTE1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA3NDkzZTQ0YTlmNzQxMzRhNzlhN2ViYTI5MTQ4MjI5ID0gJCgnPGRpdiBpZD0iaHRtbF8wNzQ5M2U0NGE5Zjc0MTM0YTc5YTdlYmEyOTE0ODIyOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpLOuc6yzqzPgc65zr/OvSwgzpTOl86czp/OoyDOkc6jzpnOnc6XzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ3ZThhZDFkM2EwZDRhYzU4OTJhMzNiMDQzMjQ4ZTE1LnNldENvbnRlbnQoaHRtbF8wNzQ5M2U0NGE5Zjc0MTM0YTc5YTdlYmEyOTE0ODIyOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kNWFjMzhmMWJkZDY0YTQ5OGMxYTM0ZGU3ZWU3OGNkMS5iaW5kUG9wdXAocG9wdXBfNDdlOGFkMWQzYTBkNGFjNTg5MmEzM2IwNDMyNDhlMTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODA0NWM3MTYyNTE3NGZjZThjNGVkMDFhNzAxYjIxZjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NDcxNjExMDAwMDAwMDQsMjIuODU1NDM4MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzIwM2IyZDFmOGIzNGE2N2FlOWRiMmIwOTIzN2M5NzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzY1MTQwOWI4YWJiNDMzNTgzNjIxZThkNmFlMGE5Y2EgPSAkKCc8ZGl2IGlkPSJodG1sXzc2NTE0MDliOGFiYjQzMzU4MzYyMWU4ZDZhZTBhOWNhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc6zzq/OsSDOoM6xz4HOsc+DzrrOtc+Fzq4sIM6UzpfOnM6fzqMgzpHOo86Zzp3Ol86jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83MjAzYjJkMWY4YjM0YTY3YWU5ZGIyYjA5MjM3Yzk3MS5zZXRDb250ZW50KGh0bWxfNzY1MTQwOWI4YWJiNDMzNTgzNjIxZThkNmFlMGE5Y2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODA0NWM3MTYyNTE3NGZjZThjNGVkMDFhNzAxYjIxZjIuYmluZFBvcHVwKHBvcHVwXzcyMDNiMmQxZjhiMzRhNjdhZTlkYjJiMDkyMzdjOTcxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VhZGRkNjI1OTdlYzQ4ZmI5OWRjYmQzZDU2M2NjNGI0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTI2ODg1OTksMjIuODc3ODgwMTAwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUzYTg0N2VjNTE5NjRmZWVhMGIzMjUzYmZmYTYwOWEwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I5MTAyNGQ1YzIwNjRhMTU4NjdhODk4NDE0NWFhOTIzID0gJCgnPGRpdiBpZD0iaHRtbF9iOTEwMjRkNWMyMDY0YTE1ODY3YTg5ODQxNDVhYTkyMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDOsc+BzrHOu86vzrEgzpHPg86vzr3Ot8+CLCDOlM6XzpzOn86jIM6RzqPOmc6dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTNhODQ3ZWM1MTk2NGZlZWEwYjMyNTNiZmZhNjA5YTAuc2V0Q29udGVudChodG1sX2I5MTAyNGQ1YzIwNjRhMTU4NjdhODk4NDE0NWFhOTIzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VhZGRkNjI1OTdlYzQ4ZmI5OWRjYmQzZDU2M2NjNGI0LmJpbmRQb3B1cChwb3B1cF81M2E4NDdlYzUxOTY0ZmVlYTBiMzI1M2JmZmE2MDlhMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kYTVmY2IxOTc5Njk0Y2RjYTI4NGJlOGNhY2ZmOWQ0YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU3NzQxNTQ3LDIyLjg4OTg4MzA0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MwMGRmNDYyN2UwMDQyZWI5NjliNTViZjI0ZjFiNjUzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MzNGE4YmVhOWJmMzRkMzY5NjViYzdmZDEyMzI1NTVkID0gJCgnPGRpdiBpZD0iaHRtbF9jMzRhOGJlYTliZjM0ZDM2OTY1YmM3ZmQxMjMyNTU1ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpHPg8+Az4HPjM6yz4HPhc+DzrcsIM6UzpfOnM6fzqMgzp3Okc6lzqDOm86Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzAwZGY0NjI3ZTAwNDJlYjk2OWI1NWJmMjRmMWI2NTMuc2V0Q29udGVudChodG1sX2MzNGE4YmVhOWJmMzRkMzY5NjViYzdmZDEyMzI1NTVkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RhNWZjYjE5Nzk2OTRjZGNhMjg0YmU4Y2FjZmY5ZDRiLmJpbmRQb3B1cChwb3B1cF9jMDBkZjQ2MjdlMDA0MmViOTY5YjU1YmYyNGYxYjY1Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNTg2NjU3Y2M4MDI0YjMyOTY2NGYyMWE3YTE3YTU2NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU4MjY3MjEyLDIyLjg2NDU2Mjk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY0Y2ZhZjQ4OGNiYzRjYzhiZTlmMzgwNWQ5YzMyZTc5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RkNmIwYTI1M2E0ODQ0MjFhMDJkODEwNDg3MDQ4YzYxID0gJCgnPGRpdiBpZD0iaHRtbF9kZDZiMGEyNTNhNDg0NDIxYTAyZDgxMDQ4NzA0OGM2MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDOsc67zrfOv8+Hz47Pgc6xLCDOlM6XzpzOn86jIM6dzpHOpc6gzpvOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY0Y2ZhZjQ4OGNiYzRjYzhiZTlmMzgwNWQ5YzMyZTc5LnNldENvbnRlbnQoaHRtbF9kZDZiMGEyNTNhNDg0NDIxYTAyZDgxMDQ4NzA0OGM2MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mNTg2NjU3Y2M4MDI0YjMyOTY2NGYyMWE3YTE3YTU2Ny5iaW5kUG9wdXAocG9wdXBfNjRjZmFmNDg4Y2JjNGNjOGJlOWYzODA1ZDljMzJlNzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzRmOTBlNjQ2Y2M3NDIyMjkwNjMwYzhmYTc1MDY3ZjUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41OTg0NDIwOCwyMi44NzI3Nzc5NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83YWMyZWI0MTk4YzQ0YTZjOTRmMTcyMDQ2YmJiOTc4ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNjM4NTE1MTNiYjg0ZGRhYTkxMmYwNzUzZDJlYzE1NCA9ICQoJzxkaXYgaWQ9Imh0bWxfMTYzODUxNTEzYmI4NGRkYWE5MTJmMDc1M2QyZWMxNTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gz4HOv8+Gzq7PhM63z4IgzpfOu86vzrHPgiwgzpTOl86czp/OoyDOnc6VzpHOoyDOpM6ZzqHOpc6dzpjOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83YWMyZWI0MTk4YzQ0YTZjOTRmMTcyMDQ2YmJiOTc4Zi5zZXRDb250ZW50KGh0bWxfMTYzODUxNTEzYmI4NGRkYWE5MTJmMDc1M2QyZWMxNTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzRmOTBlNjQ2Y2M3NDIyMjkwNjMwYzhmYTc1MDY3ZjUuYmluZFBvcHVwKHBvcHVwXzdhYzJlYjQxOThjNDRhNmM5NGYxNzIwNDZiYmI5NzhmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IxNTE0YTc0YWViMTQyZWM4YWQxMjRlMjk4NWU3NDk2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTkxMDk4NzksMjIuODQxMzg0ODldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDE4Zjg2ODIwZGZjNDY5YTg0MGVmZjgyYzZjMDk3NDggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzVhOTZlZmQ4Njc2NDQ2YzlkMDk4N2U0MDVkNmM2YzkgPSAkKCc8ZGl2IGlkPSJodG1sXzc1YTk2ZWZkODY3NjQ0NmM5ZDA5ODdlNDA1ZDZjNmM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OoM6xzr3Osc6zzq/OsSwgzpTOl86czp/OoyDOnc6VzpHOoyDOpM6ZzqHOpc6dzpjOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMThmODY4MjBkZmM0NjlhODQwZWZmODJjNmMwOTc0OC5zZXRDb250ZW50KGh0bWxfNzVhOTZlZmQ4Njc2NDQ2YzlkMDk4N2U0MDVkNmM2YzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjE1MTRhNzRhZWIxNDJlYzhhZDEyNGUyOTg1ZTc0OTYuYmluZFBvcHVwKHBvcHVwXzAxOGY4NjgyMGRmYzQ2OWE4NDBlZmY4MmM2YzA5NzQ4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE4MGViYWViMDIwMjQ4YjJiMTIxNmJkYWRjNDdjNGRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjE2MzU5NzEsMjIuOTAzNTE2NzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzE1ZWEzMzU3MjcyNGFiZmE2MzNjMDJmODdkNGFjNTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmY2ZTc2OWI1NTNiNGU4YmE4NTA5ZGQ2YTY4NjRmYmYgPSAkKCc8ZGl2IGlkPSJodG1sX2ZmNmU3NjliNTUzYjRlOGJhODUwOWRkNmE2ODY0ZmJmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM6/zr3OriDOms6xz4HOsc66zrHOu86sLCDOlM6XzpzOn86jIM6dzpXOkc6jIM6kzpnOoc6lzp3OmM6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMxNWVhMzM1NzI3MjRhYmZhNjMzYzAyZjg3ZDRhYzUyLnNldENvbnRlbnQoaHRtbF9mZjZlNzY5YjU1M2I0ZThiYTg1MDlkZDZhNjg2NGZiZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xODBlYmFlYjAyMDI0OGIyYjEyMTZiZGFkYzQ3YzRkYi5iaW5kUG9wdXAocG9wdXBfMzE1ZWEzMzU3MjcyNGFiZmE2MzNjMDJmODdkNGFjNTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmExMzMzYzIzYTA3NGY1OGEwMGM5YzU3YTU4OTM0YzQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MjY2ODk5MSwyMi44NTkxNzg1NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kOTllNDBlM2EwZmY0NzE5YTJkNTFjNzE2YTU0OTVjZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NzVmNmM1YWNkMWI0MjZhYjRkYWJmYzQxZDgyOWFjNCA9ICQoJzxkaXYgaWQ9Imh0bWxfOTc1ZjZjNWFjZDFiNDI2YWI0ZGFiZmM0MWQ4MjlhYzQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6czr/Ovc6uIM6RzrPOr86/z4UgzpjOtc6/zrTOv8+Dzq/Ov8+FIM+Ezr/PhSDOnc6tzr/PhSwgzpTOl86czp/OoyDOnM6ZzpTOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q5OWU0MGUzYTBmZjQ3MTlhMmQ1MWM3MTZhNTQ5NWNkLnNldENvbnRlbnQoaHRtbF85NzVmNmM1YWNkMWI0MjZhYjRkYWJmYzQxZDgyOWFjNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iYTEzMzNjMjNhMDc0ZjU4YTAwYzljNTdhNTg5MzRjNC5iaW5kUG9wdXAocG9wdXBfZDk5ZTQwZTNhMGZmNDcxOWEyZDUxYzcxNmE1NDk1Y2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzg4MTRkZjg0ODNkNDdkMjk5MGRjZWFiY2IwZjcwZDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MjgzODM2NCwyMi44MjE4NzY1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85OWRmOWY4ODVlNDk0MGQ2OWEyYTk4ZDMzYmQwM2EyMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82OTU1MDNjOTk3MjA0OWRhODIxMDA5MGFmY2E5OTVkNiA9ICQoJzxkaXYgaWQ9Imh0bWxfNjk1NTAzYzk5NzIwNDlkYTgyMTAwOTBhZmNhOTk1ZDYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gzrHOvc6xz4HOr8+EzrfPgiwgzpTOl86czp/OoyDOnM6ZzpTOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk5ZGY5Zjg4NWU0OTQwZDY5YTJhOThkMzNiZDAzYTIzLnNldENvbnRlbnQoaHRtbF82OTU1MDNjOTk3MjA0OWRhODIxMDA5MGFmY2E5OTVkNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jODgxNGRmODQ4M2Q0N2QyOTkwZGNlYWJjYjBmNzBkMS5iaW5kUG9wdXAocG9wdXBfOTlkZjlmODg1ZTQ5NDBkNjlhMmE5OGQzM2JkMDNhMjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTdhYjUyNzg2ODQ5NDIyYzlkMjdjYjgyMTZhZTFhOGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MTc1Njg5NywyMi43OTc1NTk3NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kOGUwY2NmNTlkYjk0NTg3ODI5ZjBkMDZmMDIwM2U3MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80YjE0ZjkzMmVlMWQ0ZmRmOGRhZGJjNDdmODU0NGY1NCA9ICQoJzxkaXYgaWQ9Imh0bWxfNGIxNGY5MzJlZTFkNGZkZjhkYWRiYzQ3Zjg1NDRmNTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Rz4HOs86/zrvOuc66z4zOvSwgzpTOl86czp/OoyDOnM6ZzpTOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q4ZTBjY2Y1OWRiOTQ1ODc4MjlmMGQwNmYwMjAzZTczLnNldENvbnRlbnQoaHRtbF80YjE0ZjkzMmVlMWQ0ZmRmOGRhZGJjNDdmODU0NGY1NCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85N2FiNTI3ODY4NDk0MjJjOWQyN2NiODIxNmFlMWE4YS5iaW5kUG9wdXAocG9wdXBfZDhlMGNjZjU5ZGI5NDU4NzgyOWYwZDA2ZjAyMDNlNzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTI2ODQ0OGRkNjMzNDQ3YmEwNGY5NmJlM2EzZWY5M2QgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41OTQ1ODU0MiwyMi43OTkyODU4OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81N2NiZTAxYThjNzA0NGE4YTg4MWNhNzQ5MzhhOWU2MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jN2NhMDBiOGM0Mzk0NTEwOWQxNTZjM2QzMDdhNWE2ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzdjYTAwYjhjNDM5NDUxMDlkMTU2YzNkMzA3YTVhNmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6kzq/Pgc+Fzr3PgiwgzpTOl86czp/OoyDOnc6VzpHOoyDOpM6ZzqHOpc6dzpjOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81N2NiZTAxYThjNzA0NGE4YTg4MWNhNzQ5MzhhOWU2MS5zZXRDb250ZW50KGh0bWxfYzdjYTAwYjhjNDM5NDUxMDlkMTU2YzNkMzA3YTVhNmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTI2ODQ0OGRkNjMzNDQ3YmEwNGY5NmJlM2EzZWY5M2QuYmluZFBvcHVwKHBvcHVwXzU3Y2JlMDFhOGM3MDQ0YThhODgxY2E3NDkzOGE5ZTYxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIyNjcxN2I2NDhmNTQwZDdiNDVjYjQyODI3Yjk3MTAyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTg2MjA0NTMsMjIuODA4MTk1MTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzBmODYzYmJjNjFlNGU5YWFjNTE3NWMxMmYyZjNkMDIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWNmNWVjNTg2MGU1NDJhZjhiNGU1YWY2YzkxZWUxMWQgPSAkKCc8ZGl2IGlkPSJodG1sXzFjZjVlYzU4NjBlNTQyYWY4YjRlNWFmNmM5MWVlMTFkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6xz4DOv860zq/Pg8+Ez4HOuc6xz4IsIM6UzpfOnM6fzqMgzp3Olc6RzqMgzqTOmc6hzqXOnc6YzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzBmODYzYmJjNjFlNGU5YWFjNTE3NWMxMmYyZjNkMDIuc2V0Q29udGVudChodG1sXzFjZjVlYzU4NjBlNTQyYWY4YjRlNWFmNmM5MWVlMTFkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIyNjcxN2I2NDhmNTQwZDdiNDVjYjQyODI3Yjk3MTAyLmJpbmRQb3B1cChwb3B1cF83MGY4NjNiYmM2MWU0ZTlhYWM1MTc1YzEyZjJmM2QwMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kOGY2OTk0YjcyNGU0MTUzYWUyYjYwMTgzYjhkNjcwNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUxNzk1NTc4LDIyLjg1NzE0MzM5OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ZjAzNjkzOTBiNzQ0NTdhODAxOWI0NGU1MTBlMTc3ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MDBlMDU2Y2ZmNTQ0MDQ2OWE4Y2ZiOTgwYzZkMDI1NSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDAwZTA1NmNmZjU0NDA0NjlhOGNmYjk4MGM2ZDAyNTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6kzr/Ou8+Mzr0sIM6UzpfOnM6fzqMgzpHOo86Zzp3Ol86jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZjAzNjkzOTBiNzQ0NTdhODAxOWI0NGU1MTBlMTc3Zi5zZXRDb250ZW50KGh0bWxfNDAwZTA1NmNmZjU0NDA0NjlhOGNmYjk4MGM2ZDAyNTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDhmNjk5NGI3MjRlNDE1M2FlMmI2MDE4M2I4ZDY3MDYuYmluZFBvcHVwKHBvcHVwXzdmMDM2OTM5MGI3NDQ1N2E4MDE5YjQ0ZTUxMGUxNzdmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFlMGExNTdiNWFlNDRhZTA5MjZlNDVjZjM2YzdiMjhhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTAwNjcxMzksMjIuODYzMDgyODldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzYwZTNlYTRlNGZlNDczOWEzMzQwODYyYmJiZWZhMDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2RjZjY5NWY4YzZhNGQxOThlNGE3MWNlMmM5Y2FhZmEgPSAkKCc8ZGl2IGlkPSJodG1sXzdkY2Y2OTVmOGM2YTRkMTk4ZTRhNzFjZTJjOWNhYWZhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OlM6xz4POus6xzrvOtc65z4wsIM6UzpfOnM6fzqMgzpHOo86Zzp3Ol86jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NjBlM2VhNGU0ZmU0NzM5YTMzNDA4NjJiYmJlZmEwNy5zZXRDb250ZW50KGh0bWxfN2RjZjY5NWY4YzZhNGQxOThlNGE3MWNlMmM5Y2FhZmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWUwYTE1N2I1YWU0NGFlMDkyNmU0NWNmMzZjN2IyOGEuYmluZFBvcHVwKHBvcHVwXzc2MGUzZWE0ZTRmZTQ3MzlhMzM0MDg2MmJiYmVmYTA3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QyZDBmYzMzNWJiNTRlMzViMjdjY2M4Y2U3ZjVmYjk3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTUzODEwMTIsMjIuOTQ2MDI1ODVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWQyZjM1N2QwYWE3NDQ5MTk1ZDNjN2RlNzYyZDZiZWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTU4OGYyMWYxODJjNGYzYTkxOWVhMGNkMzBiOTFhMDMgPSAkKCc8ZGl2IGlkPSJodG1sXzU1ODhmMjFmMTgyYzRmM2E5MTllYTBjZDMwYjkxYTAzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Cts6zzrnOv8+CIM6Rzr3PhM+Ozr3Ouc6/z4IsIM6UzpfOnM6fzqMgzpHOo86Zzp3Ol86jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZDJmMzU3ZDBhYTc0NDkxOTVkM2M3ZGU3NjJkNmJlZC5zZXRDb250ZW50KGh0bWxfNTU4OGYyMWYxODJjNGYzYTkxOWVhMGNkMzBiOTFhMDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDJkMGZjMzM1YmI1NGUzNWIyN2NjYzhjZTdmNWZiOTcuYmluZFBvcHVwKHBvcHVwXzVkMmYzNTdkMGFhNzQ0OTE5NWQzYzdkZTc2MmQ2YmVkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MzMWQxNWMwYzRkZjQxMTdhMTMzM2M5NjczZTVlNDcwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjE1NjUzOTksMjIuOTgxOTU2NDhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzM2MjEzMjEwMjhkNDU4M2I1OTAwNTYyMjI4Yjk0MWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjI3NzY3ZmNjZThiNDgyN2JkYTNjODM0MzRlYWY1NDQgPSAkKCc8ZGl2IGlkPSJodG1sX2YyNzc2N2ZjY2U4YjQ4MjdiZGEzYzgzNDM0ZWFmNTQ0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oo8+EzrHOvM6xz4TOsc6vzrnOus6xLCDOlM6XzpzOn86jIM6RzqPOms6bzpfOoM6ZzpXOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MzNjIxMzIxMDI4ZDQ1ODNiNTkwMDU2MjIyOGI5NDFmLnNldENvbnRlbnQoaHRtbF9mMjc3NjdmY2NlOGI0ODI3YmRhM2M4MzQzNGVhZjU0NCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMzFkMTVjMGM0ZGY0MTE3YTEzMzNjOTY3M2U1ZTQ3MC5iaW5kUG9wdXAocG9wdXBfYzM2MjEzMjEwMjhkNDU4M2I1OTAwNTYyMjI4Yjk0MWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWJhOTQzMjg0MmNjNDdmYmI5YWYxN2M2MGY2OGNjZWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MDY3NjE5MywyMi45NzkyNzI4NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84MGM1OTBmNWQxNzE0MTJhYjVlZDNlNjU1NzljODgzNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81YTFlOTE3MTVkNWM0ZDZjYmQ3Y2FiMTMxOTcwNDEyMyA9ICQoJzxkaXYgaWQ9Imh0bWxfNWExZTkxNzE1ZDVjNGQ2Y2JkN2NhYjEzMTk3MDQxMjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6nzr/Phc+EzrHOu86xzq/Ouc66zrEsIM6UzpfOnM6fzqMgzpHOo86azpvOl86gzpnOlc6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODBjNTkwZjVkMTcxNDEyYWI1ZWQzZTY1NTc5Yzg4MzYuc2V0Q29udGVudChodG1sXzVhMWU5MTcxNWQ1YzRkNmNiZDdjYWIxMzE5NzA0MTIzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzViYTk0MzI4NDJjYzQ3ZmJiOWFmMTdjNjBmNjhjY2ViLmJpbmRQb3B1cChwb3B1cF84MGM1OTBmNWQxNzE0MTJhYjVlZDNlNjU1NzljODgzNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zMjJmNDFlNTI3OWU0MGNmYTBhMzEzNDRjZDVlMWQ1MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU5NjQ1ODQ0LDIyLjk3MjcyMzAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZkNjFkNjI2MjRhNDQ1YzFhOTZjMDRiMjY3OTdhNWNkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JmOTc4N2ZiYzNmYjQxM2VhZGNkMTczYWVmM2QyYjkyID0gJCgnPGRpdiBpZD0iaHRtbF9iZjk3ODdmYmMzZmI0MTNlYWRjZDE3M2FlZjNkMmI5MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpPOuc6xzr3Ovc6/z4XOu86xzq/Ouc66zrEsIM6UzpfOnM6fzqMgzpHOo86azpvOl86gzpnOlc6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmQ2MWQ2MjYyNGE0NDVjMWE5NmMwNGIyNjc5N2E1Y2Quc2V0Q29udGVudChodG1sX2JmOTc4N2ZiYzNmYjQxM2VhZGNkMTczYWVmM2QyYjkyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMyMmY0MWU1Mjc5ZTQwY2ZhMGEzMTM0NGNkNWUxZDUyLmJpbmRQb3B1cChwb3B1cF9mZDYxZDYyNjI0YTQ0NWMxYTk2YzA0YjI2Nzk3YTVjZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81YjkwNTUzOWRkYzk0MjFhYjJiOTVlODg5NTg3MGNhMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU5MTYyMTQsMjIuOTgyNzE1NjFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmExYzZlNDRkMDEyNDQyMzgyYzMzNmI5NjFjNWNkY2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzI0Y2IzYjUyYWY4NGY0OGI4ZjE3MDA3YWJmYzQwNzggPSAkKCc8ZGl2IGlkPSJodG1sX2MyNGNiM2I1MmFmODRmNDhiOGYxNzAwN2FiZmM0MDc4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6/zrrOus65zr3OrM60zrXPgiwgzpTOl86czp/OoyDOkc6jzprOm86XzqDOmc6VzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yYTFjNmU0NGQwMTI0NDIzODJjMzM2Yjk2MWM1Y2RjYi5zZXRDb250ZW50KGh0bWxfYzI0Y2IzYjUyYWY4NGY0OGI4ZjE3MDA3YWJmYzQwNzgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWI5MDU1MzlkZGM5NDIxYWIyYjk1ZTg4OTU4NzBjYTIuYmluZFBvcHVwKHBvcHVwXzJhMWM2ZTQ0ZDAxMjQ0MjM4MmMzMzZiOTYxYzVjZGNiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2JkM2RmYjE0MGM2MDQxZWJiZTM1NTA3M2RhYjkxYjkyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjE0Nzc2NjEsMjMuMDE2Mzk1NTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTFjMjc0MWIwZTIzNDYyMWI4ZDk5ODA4MWQzYTg2ZWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGUzODA1ODVhMGU5NDEyNGJiYTBlMDNiMzI1MGIxMGYgPSAkKCc8ZGl2IGlkPSJodG1sXzhlMzgwNTg1YTBlOTQxMjRiYmEwZTAzYjMyNTBiMTBmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oo8+AzrfOu861zq/OsSwgzpTOl86czp/OoyDOkc6jzprOm86XzqDOmc6VzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MWMyNzQxYjBlMjM0NjIxYjhkOTk4MDgxZDNhODZlZi5zZXRDb250ZW50KGh0bWxfOGUzODA1ODVhMGU5NDEyNGJiYTBlMDNiMzI1MGIxMGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmQzZGZiMTQwYzYwNDFlYmJlMzU1MDczZGFiOTFiOTIuYmluZFBvcHVwKHBvcHVwXzUxYzI3NDFiMGUyMzQ2MjFiOGQ5OTgwODFkM2E4NmVmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZhNjE2YTJhMzJmMTQ5NTQ4ZTY3OWJkY2E4YjBmNGRmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTk3MjIxMzcsMjMuMDA0NTQzM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMWQ4ZjYwZmY2ZGM0ZDg0YjY2NjRhNzY0MmI0YmEyZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kODEzMWI2MmEzZDk0YThkOTE0NTA1ZTJlY2VhOTE1NCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDgxMzFiNjJhM2Q5NGE4ZDkxNDUwNWUyZWNlYTkxNTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6nzqzOvc65IM6czrXPgc66zr/Pjc+BzrcsIM6UzpfOnM6fzqMgzpHOo86azpvOl86gzpnOlc6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDFkOGY2MGZmNmRjNGQ4NGI2NjY0YTc2NDJiNGJhMmUuc2V0Q29udGVudChodG1sX2Q4MTMxYjYyYTNkOTRhOGQ5MTQ1MDVlMmVjZWE5MTU0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZhNjE2YTJhMzJmMTQ5NTQ4ZTY3OWJkY2E4YjBmNGRmLmJpbmRQb3B1cChwb3B1cF9kMWQ4ZjYwZmY2ZGM0ZDg0YjY2NjRhNzY0MmI0YmEyZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mZmQ3MGQ4NzIyYjk0M2NhYTA2MzkzZmNhMGVjNTJjOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU5NDY1Nzg5OTk5OTk5NCwyMy4wNzYyOTAxM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZTQxZDU1ZjAyMGE0NWI1YTY4OGIwZjgxYWUzOTBhOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MDM5OGRkZDAzMjM0ZjhjOTFiMTE0YzgyMWFmNTM4YiA9ICQoJzxkaXYgaWQ9Imh0bWxfNDAzOThkZGQwMzIzNGY4YzkxYjExNGM4MjFhZjUzOGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Rz4POus67zrfPgM65zrXOr86/IM6Vz4DOuc60zrHPjc+Bzr/PhSwgzpTOl86czp/OoyDOkc6jzprOm86XzqDOmc6VzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82ZTQxZDU1ZjAyMGE0NWI1YTY4OGIwZjgxYWUzOTBhOC5zZXRDb250ZW50KGh0bWxfNDAzOThkZGQwMzIzNGY4YzkxYjExNGM4MjFhZjUzOGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmZkNzBkODcyMmI5NDNjYWEwNjM5M2ZjYTBlYzUyYzkuYmluZFBvcHVwKHBvcHVwXzZlNDFkNTVmMDIwYTQ1YjVhNjg4YjBmODFhZTM5MGE4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FhY2IyNzg5MjUzYzQ1NDI5NGU4NDQxZTM3ODYwMjliID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjI4Njk2NDQsMjMuMDg4ODkzODldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDJiZWEyOTNlNGVlNDRlZWE1ZGE5NjBiMmNjZDBkMjkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjQ0YWFmNzU4OWNiNDFjNTlmNDBhOGM3OWU4MTBhYjggPSAkKCc8ZGl2IGlkPSJodG1sX2I0NGFhZjc1ODljYjQxYzU5ZjQwYThjNzllODEwYWI4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Cts6zzrnOv8+CIM6Rzr3OtM+Bzq3Osc+CLCDOlM6XzpzOn86jIM6RzqPOms6bzpfOoM6ZzpXOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QyYmVhMjkzZTRlZTQ0ZWVhNWRhOTYwYjJjY2QwZDI5LnNldENvbnRlbnQoaHRtbF9iNDRhYWY3NTg5Y2I0MWM1OWY0MGE4Yzc5ZTgxMGFiOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hYWNiMjc4OTI1M2M0NTQyOTRlODQ0MWUzNzg2MDI5Yi5iaW5kUG9wdXAocG9wdXBfZDJiZWEyOTNlNGVlNDRlZWE1ZGE5NjBiMmNjZDBkMjkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWMzMGJhOTdmN2I5NGM1Nzg1YzUxYzJiZGU3ZTI0YWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42Mjc0MDcwNywyMy4xMzQ0NTA5MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hN2QxNmY1NGQ5Nzk0ZTlmODI3MjA0MDI3NWIxNDMxYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84Y2NlMGI5MTEzNWE0NDdjOWU4NDc0NGY0NWIyNTU4YyA9ICQoJzxkaXYgaWQ9Imh0bWxfOGNjZTBiOTExMzVhNDQ3YzllODQ3NDRmNDViMjU1OGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gzrHOvc+Mz4HOsc68zrEsIM6UzpfOnM6fzqMgzpXOoM6ZzpTOkc6lzqHOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hN2QxNmY1NGQ5Nzk0ZTlmODI3MjA0MDI3NWIxNDMxYy5zZXRDb250ZW50KGh0bWxfOGNjZTBiOTExMzVhNDQ3YzllODQ3NDRmNDViMjU1OGMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWMzMGJhOTdmN2I5NGM1Nzg1YzUxYzJiZGU3ZTI0YWEuYmluZFBvcHVwKHBvcHVwX2E3ZDE2ZjU0ZDk3OTRlOWY4MjcyMDQwMjc1YjE0MzFjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E2ZWQ4YWEwNWZmMjQ5ZGJiNzMxMGE1YjBlMDUwMWZmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjMxOTY5NDUsMjMuMTI0MzIyODldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGY1MGNmNTNjNmM2NGY2NDkzMzdlNGQyYTJhZGVlMGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzgzMzM1MWE2NjdjNDY4NDllMWE2NGNlOGFiZGY2OWUgPSAkKCc8ZGl2IGlkPSJodG1sXzM4MzMzNTFhNjY3YzQ2ODQ5ZTFhNjRjZThhYmRmNjllIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Olc+AzqzOvc+JIM6Vz4DOr860zrHPhc+Bzr/PgiwgzpTOl86czp/OoyDOlc6gzpnOlM6RzqXOoc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RmNTBjZjUzYzZjNjRmNjQ5MzM3ZTRkMmEyYWRlZTBhLnNldENvbnRlbnQoaHRtbF8zODMzMzUxYTY2N2M0Njg0OWUxYTY0Y2U4YWJkZjY5ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNmVkOGFhMDVmZjI0OWRiYjczMTBhNWIwZTA1MDFmZi5iaW5kUG9wdXAocG9wdXBfZGY1MGNmNTNjNmM2NGY2NDkzMzdlNGQyYTJhZGVlMGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODA0ODc4OWFjYjYyNDAzNmE0ZWYxODQ2M2U2MzJlNWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MTEwMDM4OCwyMy4xNjA5NzI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FmNTM3ZDA1Mzg4YjQ0NDg5YTcxODE1OGFmZWJhMTI3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZhM2MyNjgzOTRmNDQ2ZGRhYzBmMWMxNjhlNGQ0MGU0ID0gJCgnPGRpdiBpZD0iaHRtbF82YTNjMjY4Mzk0ZjQ0NmRkYWMwZjFjMTY4ZTRkNDBlNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDOsc69zrHOs86vzrEsIM6UzpfOnM6fzqMgzpXOoM6ZzpTOkc6lzqHOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZjUzN2QwNTM4OGI0NDQ4OWE3MTgxNThhZmViYTEyNy5zZXRDb250ZW50KGh0bWxfNmEzYzI2ODM5NGY0NDZkZGFjMGYxYzE2OGU0ZDQwZTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODA0ODc4OWFjYjYyNDAzNmE0ZWYxODQ2M2U2MzJlNWQuYmluZFBvcHVwKHBvcHVwX2FmNTM3ZDA1Mzg4YjQ0NDg5YTcxODE1OGFmZWJhMTI3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzgwYjQ3ZjBkMjQ2MTQ5YzVhOGZiYjY0MWRhMzI4OGY4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNzA0NjUwODgsMjMuMTUxNDE2NzhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzAwYTUyYjhmNjAwNDJiNDkxNWUzMjBkYWE3MmRiZWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmYzOGQyZGM2YTQ0NDY2YWJjODQ1YzQyZmQyMDQyYjcgPSAkKCc8ZGl2IGlkPSJodG1sXzJmMzhkMmRjNmE0NDQ2NmFiYzg0NWM0MmZkMjA0MmI3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Ok86xzrvOsc69zrHOr865zrrOsSwgzpTOl86czp/OoyDOlc6gzpnOlM6RzqXOoc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MwMGE1MmI4ZjYwMDQyYjQ5MTVlMzIwZGFhNzJkYmVlLnNldENvbnRlbnQoaHRtbF8yZjM4ZDJkYzZhNDQ0NjZhYmM4NDVjNDJmZDIwNDJiNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MGI0N2YwZDI0NjE0OWM1YThmYmI2NDFkYTMyODhmOC5iaW5kUG9wdXAocG9wdXBfYzAwYTUyYjhmNjAwNDJiNDkxNWUzMjBkYWE3MmRiZWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTAwYzQ5M2U2N2E5NDU0Y2JjNTAzODdhNmFjMmI1N2MgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NzE4NDA2NywyMy4wODA3OTUyOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNGRhMmU4OGRhOTI0ODZkYWU1M2E0ZDhhNDJiMjkwNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80ZTBmYTYxODRkNDM0MjJlODdlYTNmMjBhM2JhMDA4MiA9ICQoJzxkaXYgaWQ9Imh0bWxfNGUwZmE2MTg0ZDQzNDIyZTg3ZWEzZjIwYTNiYTAwODIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6czr/Ovc6uIM6kzrHOvs65zrHPgc+Hz47OvSwgzpTOl86czp/OoyDOlc6gzpnOlM6RzqXOoc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE0ZGEyZTg4ZGE5MjQ4NmRhZTUzYTRkOGE0MmIyOTA0LnNldENvbnRlbnQoaHRtbF80ZTBmYTYxODRkNDM0MjJlODdlYTNmMjBhM2JhMDA4Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MDBjNDkzZTY3YTk0NTRjYmM1MDM4N2E2YWMyYjU3Yy5iaW5kUG9wdXAocG9wdXBfMTRkYTJlODhkYTkyNDg2ZGFlNTNhNGQ4YTQyYjI5MDQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGM1YTQzYzg3NWQ4NDNkYzhmNGYyNmIyY2Q4OGRmNTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42ODA5NzY4NywyMy4wNzMyNjUwOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lOWYzMTZiOGIzY2E0ZTMzYmQ2YjI2ZTFkOTgxZGRjMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lOThiYzZjNTRlZTk0Yzc4YjM2NGJlNjQyYWZmZmM2OSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTk4YmM2YzU0ZWU5NGM3OGIzNjRiZTY0MmFmZmZjNjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6dzq3OsSDOlM6uzrzOsc65zr3OsSwgzpTOl86czp/OoyDOlc6gzpnOlM6RzqXOoc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U5ZjMxNmI4YjNjYTRlMzNiZDZiMjZlMWQ5ODFkZGMxLnNldENvbnRlbnQoaHRtbF9lOThiYzZjNTRlZTk0Yzc4YjM2NGJlNjQyYWZmZmM2OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80YzVhNDNjODc1ZDg0M2RjOGY0ZjI2YjJjZDg4ZGY1Ny5iaW5kUG9wdXAocG9wdXBfZTlmMzE2YjhiM2NhNGUzM2JkNmIyNmUxZDk4MWRkYzEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjA1YTM3OGZiNDZmNGFmMDg2MTFiNTZiYTZlZTA2YWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41Njc4NDgyMSwyMy4xNDQzMTU3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNzg0MmNhNThkNDA0NDgxOWI2ZTJjYTc4YjQ5YjFiMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZTlkMGFlNTM4Y2E0NDNjOGUwYWU1OGQyMjUwZGNhZiA9ICQoJzxkaXYgaWQ9Imh0bWxfOWU5ZDBhZTUzOGNhNDQzYzhlMGFlNThkMjI1MGRjYWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Vzr7Ov8+Hzq4sIM6UzpfOnM6fzqMgzpXOoM6ZzpTOkc6lzqHOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zNzg0MmNhNThkNDA0NDgxOWI2ZTJjYTc4YjQ5YjFiMi5zZXRDb250ZW50KGh0bWxfOWU5ZDBhZTUzOGNhNDQzYzhlMGFlNThkMjI1MGRjYWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjA1YTM3OGZiNDZmNGFmMDg2MTFiNTZiYTZlZTA2YWUuYmluZFBvcHVwKHBvcHVwXzM3ODQyY2E1OGQ0MDQ0ODE5YjZlMmNhNzhiNDliMWIyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2EwMDIzMmU3NTdmOTRkZGZhMmQzZjgwNjUzODZmY2ZlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTkwNzgyMTcsMjMuMTU3NTAzMTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTdiYmZlZjdiNGFmNGI4MjkzNWMyNjNlNzY5YjAxMDEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDViOWYwYzRhZGY4NDliYjg5OTI4YTdlODQ4NGYwNTQgPSAkKCc8ZGl2IGlkPSJodG1sX2Q1YjlmMGM0YWRmODQ5YmI4OTkyOGE3ZTg0ODRmMDU0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6/zrvOuc6szrrOuc6/zr0sIM6UzpfOnM6fzqMgzpXOoM6ZzpTOkc6lzqHOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lN2JiZmVmN2I0YWY0YjgyOTM1YzI2M2U3NjliMDEwMS5zZXRDb250ZW50KGh0bWxfZDViOWYwYzRhZGY4NDliYjg5OTI4YTdlODQ4NGYwNTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTAwMjMyZTc1N2Y5NGRkZmEyZDNmODA2NTM4NmZjZmUuYmluZFBvcHVwKHBvcHVwX2U3YmJmZWY3YjRhZjRiODI5MzVjMjYzZTc2OWIwMTAxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2YyOWUyMmRhNGFhYjRlYTRiY2VlMWIxZTI3Y2I2N2Q1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTUyNTU4OSwyMy4xMzUwNzI3MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yOTY1MmY4ZDJjMjA0MzhkOTAyMGY1NTk2NWQ3MTQ1YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNzcxZjgxOWE0YmE0ODI3ODdmOWM3YTE1NTY1MzA0NyA9ICQoJzxkaXYgaWQ9Imh0bWxfMTc3MWY4MTlhNGJhNDgyNzg3ZjljN2ExNTU2NTMwNDciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6jz4TOsc+Fz4HPjM+CLCDOlM6XzpzOn86jIM6VzqDOmc6UzpHOpc6hzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjk2NTJmOGQyYzIwNDM4ZDkwMjBmNTU5NjVkNzE0NWMuc2V0Q29udGVudChodG1sXzE3NzFmODE5YTRiYTQ4Mjc4N2Y5YzdhMTU1NjUzMDQ3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2YyOWUyMmRhNGFhYjRlYTRiY2VlMWIxZTI3Y2I2N2Q1LmJpbmRQb3B1cChwb3B1cF8yOTY1MmY4ZDJjMjA0MzhkOTAyMGY1NTk2NWQ3MTQ1Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zMDU5NWM5YWU5YmI0YjhiYTNjNzY0ZmYxMmFiNTdjNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUzNTI4OTc2LDIzLjEzNzI4NTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBiMmFhM2NlOTAzMjQ5Mjk5M2U4OWU2NGFjM2RkZTQ5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2UxOWYyNWQ0NDQwZTRkOGM4ZjRjYjZmMWQ4NTlkYjk1ID0gJCgnPGRpdiBpZD0iaHRtbF9lMTlmMjVkNDQ0MGU0ZDhjOGY0Y2I2ZjFkODU5ZGI5NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpLOv864zq/Ous65zr/OvSwgzpTOl86czp/OoyDOlc6gzpnOlM6RzqXOoc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBiMmFhM2NlOTAzMjQ5Mjk5M2U4OWU2NGFjM2RkZTQ5LnNldENvbnRlbnQoaHRtbF9lMTlmMjVkNDQ0MGU0ZDhjOGY0Y2I2ZjFkODU5ZGI5NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zMDU5NWM5YWU5YmI0YjhiYTNjNzY0ZmYxMmFiNTdjNy5iaW5kUG9wdXAocG9wdXBfMGIyYWEzY2U5MDMyNDkyOTkzZTg5ZTY0YWMzZGRlNDkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDc5MTc1ODZiNzE2NDYzZjk4ZjYyYTBmOWIwZDY2YzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NDE5MjM1MiwyMy4xMTM1NjE2M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYTIzYjQ5NmQwOGY0YzBlYjMxZGY1MmExY2M1NGEzMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NGJhMGEwYTQ2ZWI0NDI5OTAzOGRiMzBlNzM4OTk2MSA9ICQoJzxkaXYgaWQ9Imh0bWxfOTRiYTBhMGE0NmViNDQyOTkwMzhkYjMwZTczODk5NjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6czrHPhM6xz4HOrM6zzrrOsSwgzpTOl86czp/OoyDOlc6gzpnOlM6RzqXOoc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FhMjNiNDk2ZDA4ZjRjMGViMzFkZjUyYTFjYzU0YTMwLnNldENvbnRlbnQoaHRtbF85NGJhMGEwYTQ2ZWI0NDI5OTAzOGRiMzBlNzM4OTk2MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NzkxNzU4NmI3MTY0NjNmOThmNjJhMGY5YjBkNjZjMC5iaW5kUG9wdXAocG9wdXBfYWEyM2I0OTZkMDhmNGMwZWIzMWRmNTJhMWNjNTRhMzApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGU5NmNiYzE3MDUyNGUzOWIxNTE4YjU4ZjY2NTA5YTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41MDQ0Mjg4NiwyMy4wNTk2MDI3NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zZjBkZWExMWMzMDg0MjRmYmU4MzdhZDNlNjhjOWFjZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yODk2Njg0ZGIzYWE0ZTBiODQxODhmYWJlNmQyZDg2NiA9ICQoJzxkaXYgaWQ9Imh0bWxfMjg5NjY4NGRiM2FhNGUwYjg0MTg4ZmFiZTZkMmQ4NjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPsK2zr3PiSDOms6xz4HOvc61zrbOsc6vzrnOus6xLCDOlM6XzpzOn86jIM6RzqPOmc6dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2YwZGVhMTFjMzA4NDI0ZmJlODM3YWQzZTY4YzlhY2Yuc2V0Q29udGVudChodG1sXzI4OTY2ODRkYjNhYTRlMGI4NDE4OGZhYmU2ZDJkODY2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBlOTZjYmMxNzA1MjRlMzliMTUxOGI1OGY2NjUwOWE2LmJpbmRQb3B1cChwb3B1cF8zZjBkZWExMWMzMDg0MjRmYmU4MzdhZDNlNjhjOWFjZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NmRlNjFhODMxZWU0MjY1OWZkYjc4M2UzMjJlOWU5MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUxMTUyODAyLDIzLjA5NjgwMzY3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2VkYzk4NmMzNzQyZDRmZDRhMDQzNTEwNzRmODViNWM2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2IzOTQxZWQxZTc3MTQ2OTI4NTJlZWQ3ZWI5YWU5OGY1ID0gJCgnPGRpdiBpZD0iaHRtbF9iMzk0MWVkMWU3NzE0NjkyODUyZWVkN2ViOWFlOThmNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOsc69zrHPgM6vz4TPg86xLCDOlM6XzpzOn86jIM6RzqPOmc6dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWRjOTg2YzM3NDJkNGZkNGEwNDM1MTA3NGY4NWI1YzYuc2V0Q29udGVudChodG1sX2IzOTQxZWQxZTc3MTQ2OTI4NTJlZWQ3ZWI5YWU5OGY1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk2ZGU2MWE4MzFlZTQyNjU5ZmRiNzgzZTMyMmU5ZTkyLmJpbmRQb3B1cChwb3B1cF9lZGM5ODZjMzc0MmQ0ZmQ0YTA0MzUxMDc0Zjg1YjVjNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYTBjNzU5NjQ4MzM0OTA4YTlmNmY1NzBiNWU3YTk0MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUyMDIwNjQ1LDIzLjExNjM4ODMyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUxYTE5MDMyMTY1MzQwMTc5ZWNlNzI5YTRjMDZhMjQ1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NjYjY4YzQ1NmY0MTQ3NzFiYzQ5NTVkZTExYWVlNTA5ID0gJCgnPGRpdiBpZD0iaHRtbF9jY2I2OGM0NTZmNDE0NzcxYmM0OTU1ZGUxMWFlZTUwOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqPPhM6xz4XPgc6/z4DPjM60zrnOv869LCDOlM6XzpzOn86jIM6RzqPOmc6dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTFhMTkwMzIxNjUzNDAxNzllY2U3MjlhNGMwNmEyNDUuc2V0Q29udGVudChodG1sX2NjYjY4YzQ1NmY0MTQ3NzFiYzQ5NTVkZTExYWVlNTA5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBhMGM3NTk2NDgzMzQ5MDhhOWY2ZjU3MGI1ZTdhOTQyLmJpbmRQb3B1cChwb3B1cF81MWExOTAzMjE2NTM0MDE3OWVjZTcyOWE0YzA2YTI0NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZGY3NGRkYjVlYjM0MDU2YjA1ZDYzYjVmMWNjMjFmNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU2OTk4NDQ0LDIzLjA3MTIyODAzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA4NjQ0M2I0MjcwZDRkY2E4YmZhNTI5ZTA1NmM4N2I5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FhZjEzYzQ4NDg4MTQxYjBhNjdhMDYzZmU1ZTBhNDJmID0gJCgnPGRpdiBpZD0iaHRtbF9hYWYxM2M0ODQ4ODE0MWIwYTY3YTA2M2ZlNWUwYTQyZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpHOtM6szrzOuc6/zr0sIM6UzpfOnM6fzqMgzpHOo86azpvOl86gzpnOlc6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDg2NDQzYjQyNzBkNGRjYThiZmE1MjllMDU2Yzg3Yjkuc2V0Q29udGVudChodG1sX2FhZjEzYzQ4NDg4MTQxYjBhNjdhMDYzZmU1ZTBhNDJmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJkZjc0ZGRiNWViMzQwNTZiMDVkNjNiNWYxY2MyMWY3LmJpbmRQb3B1cChwb3B1cF8wODY0NDNiNDI3MGQ0ZGNhOGJmYTUyOWUwNTZjODdiOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ZmI2ZjQxODYxMTk0YWFmOTAwNmE5NDA4ZGRiNGM5MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU4OTMwNTg4LDIzLjEwNTQ0MDE0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQzMjI5MzQ4NzQ1ZjRhYzA5NmUzYThhNjZiZDU0ODMwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI1YTNhOGZiMzhhYzQ4YWFhMzNkZjcxZDk4MzU4MTdlID0gJCgnPGRpdiBpZD0iaHRtbF8yNWEzYThmYjM4YWM0OGFhYTMzZGY3MWQ5ODM1ODE3ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpTOt868zr/Pg865zqwsIM6UzpfOnM6fzqMgzpHOo86azpvOl86gzpnOlc6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDMyMjkzNDg3NDVmNGFjMDk2ZTNhOGE2NmJkNTQ4MzAuc2V0Q29udGVudChodG1sXzI1YTNhOGZiMzhhYzQ4YWFhMzNkZjcxZDk4MzU4MTdlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdmYjZmNDE4NjExOTRhYWY5MDA2YTk0MDhkZGI0YzkyLmJpbmRQb3B1cChwb3B1cF80MzIyOTM0ODc0NWY0YWMwOTZlM2E4YTY2YmQ1NDgzMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ODIwZjdiZWNkMDI0MTZjYTNmNTIwNjBmZDc5NDc0MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU0NTMxNDc5LDIyLjk5MDcwMzU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMzMTk3OGE5OWZhMjRmYzQ5YmY0N2ZmZTJmMWE1NzFkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E0M2FlNjVlOTczNDQwMTI4MDVhYjVjMGIxZDdjMDhmID0gJCgnPGRpdiBpZD0iaHRtbF9hNDNhZTY1ZTk3MzQ0MDEyODA1YWI1YzBiMWQ3YzA4ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbOs865zr/PgiDOnc65zrrPjM67zrHOv8+CLCDOlM6XzpzOn86jIM6RzqPOms6bzpfOoM6ZzpXOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMzMTk3OGE5OWZhMjRmYzQ5YmY0N2ZmZTJmMWE1NzFkLnNldENvbnRlbnQoaHRtbF9hNDNhZTY1ZTk3MzQ0MDEyODA1YWI1YzBiMWQ3YzA4Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80ODIwZjdiZWNkMDI0MTZjYTNmNTIwNjBmZDc5NDc0MC5iaW5kUG9wdXAocG9wdXBfMzMxOTc4YTk5ZmEyNGZjNDliZjQ3ZmZlMmYxYTU3MWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWMzNDc1MTQ3YmZmNDlmMWIxYjYxZWU2YmU1MTdkMjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40OTI4NDc0NCwyMy4wMjU2NjcxOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mM2NjNTIzYjdjY2E0Njc2OTk5ZTM1YjA4MjYzYWYyYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wODQzZmZlYTZkM2E0NjVjOWRkOWU2ZWI1NTMyYmQ3YSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDg0M2ZmZWE2ZDNhNDY1YzlkZDllNmViNTUzMmJkN2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6azrHPhM+DzrnOs865zrHOvc69zrHOr865zrrOsSwgzpTOl86czp/OoyDOkc6jzpnOnc6XzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2YzY2M1MjNiN2NjYTQ2NzY5OTllMzViMDgyNjNhZjJjLnNldENvbnRlbnQoaHRtbF8wODQzZmZlYTZkM2E0NjVjOWRkOWU2ZWI1NTMyYmQ3YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85YzM0NzUxNDdiZmY0OWYxYjFiNjFlZTZiZTUxN2QyMi5iaW5kUG9wdXAocG9wdXBfZjNjYzUyM2I3Y2NhNDY3Njk5OWUzNWIwODI2M2FmMmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzNkNTI5YjUyYmQ3NGM2ZjgxOGI3MWI5Y2M3NzA3NGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41MjcyNTIyMDAwMDAwMSwyMi45NTk1MzU2MDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzM5MDBmMjc5NjU5NDBlYjhmNTI0MjY5N2VkMjVhOGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWM4NWEzNWFiZTM0NGE1ZDkxMzk3OGU1MDU5MmZlZTAgPSAkKCc8ZGl2IGlkPSJodG1sXzVjODVhMzVhYmUzNDRhNWQ5MTM5NzhlNTA1OTJmZWUwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6szr3PhM65zrEsIM6UzpfOnM6fzqMgzpHOo86Zzp3Ol86jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83MzkwMGYyNzk2NTk0MGViOGY1MjQyNjk3ZWQyNWE4YS5zZXRDb250ZW50KGh0bWxfNWM4NWEzNWFiZTM0NGE1ZDkxMzk3OGU1MDU5MmZlZTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzNkNTI5YjUyYmQ3NGM2ZjgxOGI3MWI5Y2M3NzA3NGYuYmluZFBvcHVwKHBvcHVwXzczOTAwZjI3OTY1OTQwZWI4ZjUyNDI2OTdlZDI1YThhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E2ZmU4NWNiNzM4OTQ1NzFhODhmYmY0ZmY0YzgwOWQ1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDM0MjE5MzYsMjIuOTkxNDg1NjAwMDAwMDA0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkxYjU3Y2M1Njg4MjQ5MWFhMWQ2ZjdiZGFhYTZlYTExID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhiNGFjNTJlNDIwNzRiZGY4ZWFjMTAxYzg5Y2U2Y2RmID0gJCgnPGRpdiBpZD0iaHRtbF84YjRhYzUyZTQyMDc0YmRmOGVhYzEwMWM4OWNlNmNkZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqjOt867zq4sIM6UzpfOnM6fzqMgzpHOo86Zzp3Ol86jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MWI1N2NjNTY4ODI0OTFhYTFkNmY3YmRhYWE2ZWExMS5zZXRDb250ZW50KGh0bWxfOGI0YWM1MmU0MjA3NGJkZjhlYWMxMDFjODljZTZjZGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTZmZTg1Y2I3Mzg5NDU3MWE4OGZiZjRmZjRjODA5ZDUuYmluZFBvcHVwKHBvcHVwXzkxYjU3Y2M1Njg4MjQ5MWFhMWQ2ZjdiZGFhYTZlYTExKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ0NThmNGYzMTdiZjQwMzk4MzI4YmRjZWQzZWFkMGJhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDExMTM2NjMsMjMuMzMyMTY0NzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjY4YTIyODFmODY2NDUwY2E3NDcyYWIxYTkzYmYzYWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjkwZWQ1ZjFhY2YwNGJmNTg2YTY3MjIzNzJhNWVmOGIgPSAkKCc8ZGl2IGlkPSJodG1sXzI5MGVkNWYxYWNmMDRiZjU4NmE2NzIyMzcyYTVlZjhiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc66z4TOriDOjs60z4HOsc+CLCDOlM6XzpzOn86jIM6VzqHOnM6Zzp/Onc6XzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y2OGEyMjgxZjg2NjQ1MGNhNzQ3MmFiMWE5M2JmM2FlLnNldENvbnRlbnQoaHRtbF8yOTBlZDVmMWFjZjA0YmY1ODZhNjcyMjM3MmE1ZWY4Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NDU4ZjRmMzE3YmY0MDM5ODMyOGJkY2VkM2VhZDBiYS5iaW5kUG9wdXAocG9wdXBfZjY4YTIyODFmODY2NDUwY2E3NDcyYWIxYTkzYmYzYWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmE0NzVhZmY5YzMxNDY1YTkzODJlNWRkZDEyODc5YmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40MTI0NzU1OSwyMy4zNDY1MDQyMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wMjI2YTZjZGEyOTI0YTljYjBlZjFjZjJjZDM0MjExZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81NmIxZmQ0M2VhZDM0NGE4OTIxNDdhOWFiOWZlMDU3ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNTZiMWZkNDNlYWQzNDRhODkyMTQ3YTlhYjlmZTA1N2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6RzrPOr86xIM6RzrnOus6xz4TOtc+Bzq/Ovc63LCDOlM6XzpzOn86jIM6VzqHOnM6Zzp/Onc6XzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzAyMjZhNmNkYTI5MjRhOWNiMGVmMWNmMmNkMzQyMTFlLnNldENvbnRlbnQoaHRtbF81NmIxZmQ0M2VhZDM0NGE4OTIxNDdhOWFiOWZlMDU3ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iYTQ3NWFmZjljMzE0NjVhOTM4MmU1ZGRkMTI4NzliYS5iaW5kUG9wdXAocG9wdXBfMDIyNmE2Y2RhMjkyNGE5Y2IwZWYxY2YyY2QzNDIxMWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmFhODZkZTE4NWJiNDYzNTllMjI2YmZiYjc3ZmI4ZDIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40MTE4MzA5LDIzLjM1NDE4NzAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA0ZWIyOTJjMzFjMjRmNzc4NjVlYzQ1NmM5ODU1YmUxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBkYzA3ZmRhZjZmODQ1NDQ5NjdlZjNhODI4NWNjYzYyID0gJCgnPGRpdiBpZD0iaHRtbF8wZGMwN2ZkYWY2Zjg0NTQ0OTY3ZWYzYTgyODVjY2M2MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDOu86tz4DOuc6/zr0sIM6UzpfOnM6fzqMgzpXOoc6czpnOn86dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDRlYjI5MmMzMWMyNGY3Nzg2NWVjNDU2Yzk4NTViZTEuc2V0Q29udGVudChodG1sXzBkYzA3ZmRhZjZmODQ1NDQ5NjdlZjNhODI4NWNjYzYyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZhYTg2ZGUxODViYjQ2MzU5ZTIyNmJmYmI3N2ZiOGQyLmJpbmRQb3B1cChwb3B1cF8wNGViMjkyYzMxYzI0Zjc3ODY1ZWM0NTZjOTg1NWJlMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80YmZjZGM5MmEwZGE0ZmM1YWZkNzIzZDgyMzQ5OWNiZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQxMDI2Njg4LDIzLjM3MjQ5NzU2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YzNTQ1MGZlZmVlNTRlMGE5YjA2ODhlZjFlMTZjMmNmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U2ODVhZTI4OGQ1MjRiMTdiZDU1MjQxM2ViNDg4OWE1ID0gJCgnPGRpdiBpZD0iaHRtbF9lNjg1YWUyODhkNTI0YjE3YmQ1NTI0MTNlYjQ4ODlhNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDOt86zzqzOtM65zrEsIM6UzpfOnM6fzqMgzpXOoc6czpnOn86dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjM1NDUwZmVmZWU1NGUwYTliMDY4OGVmMWUxNmMyY2Yuc2V0Q29udGVudChodG1sX2U2ODVhZTI4OGQ1MjRiMTdiZDU1MjQxM2ViNDg4OWE1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRiZmNkYzkyYTBkYTRmYzVhZmQ3MjNkODIzNDk5Y2JmLmJpbmRQb3B1cChwb3B1cF9mMzU0NTBmZWZlZTU0ZTBhOWIwNjg4ZWYxZTE2YzJjZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNzk4ODQyYjU5ZWE0ZWViYWFjYzg1MjZmNDViNzEzNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQwOTg1MTA3LDIzLjM4MDUzNzAzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI5ZGFmM2RkZDIyODQ2MTQ4NGM5OTQyNDlmYmIyNWFkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc2OTQwMzQ2MGRiZDQ3ZWM5OWUyNGFmZTdhZGE4ZjU4ID0gJCgnPGRpdiBpZD0iaHRtbF83Njk0MDM0NjBkYmQ0N2VjOTllMjRhZmU3YWRhOGY1OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqPPic67zrfOvc6sz4HOuc6/zr0sIM6UzpfOnM6fzqMgzpXOoc6czpnOn86dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjlkYWYzZGRkMjI4NDYxNDg0Yzk5NDI0OWZiYjI1YWQuc2V0Q29udGVudChodG1sXzc2OTQwMzQ2MGRiZDQ3ZWM5OWUyNGFmZTdhZGE4ZjU4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I3OTg4NDJiNTllYTRlZWJhYWNjODUyNmY0NWI3MTM0LmJpbmRQb3B1cChwb3B1cF8yOWRhZjNkZGQyMjg0NjE0ODRjOTk0MjQ5ZmJiMjVhZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yOGY5ZTllOTI1ODc0MDg3ODMyZGViMTI1MTRjNWMzMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQxMzYwMDkyLDIzLjM5NzI5NjkxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdmYjgwYmEzYTM5YjQzMDZhYTQwMWIyMTBiYTQ1M2QzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNiMTE5OWMyN2RjNjRjYzQ4YTU0ZDE2MGZkOWRmMDY3ID0gJCgnPGRpdiBpZD0iaHRtbF8zYjExOTljMjdkYzY0Y2M0OGE1NGQxNjBmZDlkZjA2NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpzOtc+Ez4zPh865zr/OvSwgzpTOl86czp/OoyDOlc6hzpzOmc6fzp3Ol86jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZmI4MGJhM2EzOWI0MzA2YWE0MDFiMjEwYmE0NTNkMy5zZXRDb250ZW50KGh0bWxfM2IxMTk5YzI3ZGM2NGNjNDhhNTRkMTYwZmQ5ZGYwNjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjhmOWU5ZTkyNTg3NDA4NzgzMmRlYjEyNTE0YzVjMzEuYmluZFBvcHVwKHBvcHVwXzdmYjgwYmEzYTM5YjQzMDZhYTQwMWIyMTBiYTQ1M2QzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMzNTYzZGM1OGQ4YjQ4YjZhYjA1MTE0NWQ2MTEwOWI1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDAyMDUzODMsMjMuMjgwNzI3MzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTRmNzMwNGFiMTJkNGVmNDgyMTgzYTczZDIzZTk3YTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDZmMDk4NThmZjhkNGI5MGEzOTJjNmEwNzMwOGM0MmUgPSAkKCc8ZGl2IGlkPSJodG1sXzQ2ZjA5ODU4ZmY4ZDRiOTBhMzkyYzZhMDczMDhjNDJlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc+HzrvOsc60zq/PhM+DzrEsIM6UzpfOnM6fzqMgzpXOoc6czpnOn86dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTRmNzMwNGFiMTJkNGVmNDgyMTgzYTczZDIzZTk3YTUuc2V0Q29udGVudChodG1sXzQ2ZjA5ODU4ZmY4ZDRiOTBhMzkyYzZhMDczMDhjNDJlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMzNTYzZGM1OGQ4YjQ4YjZhYjA1MTE0NWQ2MTEwOWI1LmJpbmRQb3B1cChwb3B1cF81NGY3MzA0YWIxMmQ0ZWY0ODIxODNhNzNkMjNlOTdhNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85YmE2NzZiZDA1MDA0Njc1OWI4YTBlNTIzNzIyMzEzYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjM1MTYwODI4LDIzLjI0MDYyOTE5OTk5OTk5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84NmFmMzFjY2VmYmU0Yzc4OTBmYTczNDE5ZTYyZmMwMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNjRiNmZhNGYyZTc0MGRjODk3YzBkMGQ4YTFmZDhjYSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTY0YjZmYTRmMmU3NDBkYzg5N2MwZDBkOGExZmQ4Y2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPsK2zrPOuc6/zrkgzpHOvc6sz4HOs8+Fz4HOv865LCDOlM6XzpzOn86jIM6VzqHOnM6Zzp/Onc6XzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg2YWYzMWNjZWZiZTRjNzg5MGZhNzM0MTllNjJmYzAzLnNldENvbnRlbnQoaHRtbF9lNjRiNmZhNGYyZTc0MGRjODk3YzBkMGQ4YTFmZDhjYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85YmE2NzZiZDA1MDA0Njc1OWI4YTBlNTIzNzIyMzEzYy5iaW5kUG9wdXAocG9wdXBfODZhZjMxY2NlZmJlNGM3ODkwZmE3MzQxOWU2MmZjMDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmJhNjdmNDA3NTA4NDgyNDg1NjUxZjJlMWIyMDk3MTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4zMzk3MDY0MiwyMy4xOTI1NTYzOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jZTA4NTAxY2NhMDE0YjY2ODI4NmEwN2I2ZGRiNzlkOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jYzA3MWZmNWFjMzg0YTEzYTQ5MTY2OTc1ZWZmZTcwZiA9ICQoJzxkaXYgaWQ9Imh0bWxfY2MwNzFmZjVhYzM4NGExM2E0OTE2Njk3NWVmZmU3MGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gzrXPhM+Bzr/OuM6szrvOsc+Dz4POsSwgzpTOl86czp/OoyDOms6hzpHOnc6ZzpTOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NlMDg1MDFjY2EwMTRiNjY4Mjg2YTA3YjZkZGI3OWQ4LnNldENvbnRlbnQoaHRtbF9jYzA3MWZmNWFjMzg0YTEzYTQ5MTY2OTc1ZWZmZTcwZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYmE2N2Y0MDc1MDg0ODI0ODU2NTFmMmUxYjIwOTcxMi5iaW5kUG9wdXAocG9wdXBfY2UwODUwMWNjYTAxNGI2NjgyODZhMDdiNmRkYjc5ZDgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWViY2Q1ZDQwMzc3NDEzMzg2NDQ1MTM3MGQ0Yzk5ZjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4zMDMyMjY0NywyMy4xOTU0NDk4M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yYjliZjRhMGI5Yjc0ODI5OGM1YmZlMTgyOTcyNDFkOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNTc5MTdmZjUyMzE0MDEwYTM1ZGZhOGM5NDAzZjE1MiA9ICQoJzxkaXYgaWQ9Imh0bWxfMTU3OTE3ZmY1MjMxNDAxMGEzNWRmYThjOTQwM2YxNTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6azr/Phc69zr/Pjc+AzrksIM6UzpfOnM6fzqMgzprOoc6Rzp3Omc6UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yYjliZjRhMGI5Yjc0ODI5OGM1YmZlMTgyOTcyNDFkOC5zZXRDb250ZW50KGh0bWxfMTU3OTE3ZmY1MjMxNDAxMGEzNWRmYThjOTQwM2YxNTIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWViY2Q1ZDQwMzc3NDEzMzg2NDQ1MTM3MGQ0Yzk5ZjguYmluZFBvcHVwKHBvcHVwXzJiOWJmNGEwYjliNzQ4Mjk4YzViZmUxODI5NzI0MWQ4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUzMzNjYmNhMTI0ZDQ5Yjc5NTdiNjNhZjZhZDViYmEzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuMzUxNDA2MSwyMy4wOTI3NzM0NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hNDdhNmM3ZDY4ZmQ0N2QwYjdiZGU2MTk0OTkyZWY0OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NGE4ZTRiZmIzZTY0ZWY5YjQ3N2NjMzlkYTAwYjg0MiA9ICQoJzxkaXYgaWQ9Imh0bWxfOTRhOGU0YmZiM2U2NGVmOWI0NzdjYzM5ZGEwMGI4NDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPsK2zrPOuc6/z4Igzp3Ouc66z4zOu86xzr/PgiwgzpTOl86czp/OoyDOms6hzpHOnc6ZzpTOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E0N2E2YzdkNjhmZDQ3ZDBiN2JkZTYxOTQ5OTJlZjQ4LnNldENvbnRlbnQoaHRtbF85NGE4ZTRiZmIzZTY0ZWY5YjQ3N2NjMzlkYTAwYjg0Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MzMzY2JjYTEyNGQ0OWI3OTU3YjYzYWY2YWQ1YmJhMy5iaW5kUG9wdXAocG9wdXBfYTQ3YTZjN2Q2OGZkNDdkMGI3YmRlNjE5NDk5MmVmNDgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2UyYmU0NzQyZjJlNGU3MTkyMDljMzg2NzdkZDQyODQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4zODQzNDYwMSwyMy4wODE2ODk4M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yOTkyMjY5OGIxOGU0YjM0OTY0MWNlMTc3OTg3ODQ3YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wYjZhOWEwODI2ZmM0ZDEzYjNmNjVjMGUzZmY3ZmMyNyA9ICQoJzxkaXYgaWQ9Imh0bWxfMGI2YTlhMDgyNmZjNGQxM2IzZjY1YzBlM2ZmN2ZjMjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Yz4XOvc6vLCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjk5MjI2OThiMThlNGIzNDk2NDFjZTE3Nzk4Nzg0N2Iuc2V0Q29udGVudChodG1sXzBiNmE5YTA4MjZmYzRkMTNiM2Y2NWMwZTNmZjdmYzI3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNlMmJlNDc0MmYyZTRlNzE5MjA5YzM4Njc3ZGQ0Mjg0LmJpbmRQb3B1cChwb3B1cF8yOTkyMjY5OGIxOGU0YjM0OTY0MWNlMTc3OTg3ODQ3Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZDY2ODZmNmQzYjE0NzE2YTgyYWJjNzljMzUxMjA2YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjM4NjgyMTc1LDIzLjA5ODczMDA5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ4YTkzMGE3YmUxNTQ4MDNiOWE5OGRiOWE2YjM1ZDkyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA4NTBkYzI1ZDU1MjRhNmY4OWJhNDhmNzEzMjg0N2RlID0gJCgnPGRpdiBpZD0iaHRtbF8wODUwZGMyNWQ1NTI0YTZmODliYTQ4ZjcxMzI4NDdkZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpvOrM66zrrOtc+CLCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDhhOTMwYTdiZTE1NDgwM2I5YTk4ZGI5YTZiMzVkOTIuc2V0Q29udGVudChodG1sXzA4NTBkYzI1ZDU1MjRhNmY4OWJhNDhmNzEzMjg0N2RlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FkNjY4NmY2ZDNiMTQ3MTZhODJhYmM3OWMzNTEyMDZiLmJpbmRQb3B1cChwb3B1cF80OGE5MzBhN2JlMTU0ODAzYjlhOThkYjlhNmIzNWQ5Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kM2Q4OWEwY2E2MWQ0ODhkOTcxOTAzZDcxYjZjNjEyZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjM5MzM2MDE0LDIzLjEwNzkwNjM0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYyNGE3Zjk4Njc2ZTRmNzFhZmJiYjdlYTg0OWY0YTlhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhmNzYzZWZkOGQ0YTQ4NzQ4MzJlYTVkNTUyM2I3NGVlID0gJCgnPGRpdiBpZD0iaHRtbF84Zjc2M2VmZDhkNGE0ODc0ODMyZWE1ZDU1MjNiNzRlZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpTOv8+Bzr/Pjc+GzrksIM6UzpfOnM6fzqMgzprOoc6Rzp3Omc6UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MjRhN2Y5ODY3NmU0ZjcxYWZiYmI3ZWE4NDlmNGE5YS5zZXRDb250ZW50KGh0bWxfOGY3NjNlZmQ4ZDRhNDg3NDgzMmVhNWQ1NTIzYjc0ZWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDNkODlhMGNhNjFkNDg4ZDk3MTkwM2Q3MWI2YzYxMmYuYmluZFBvcHVwKHBvcHVwXzYyNGE3Zjk4Njc2ZTRmNzFhZmJiYjdlYTg0OWY0YTlhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ4ZDYzMmNhODVhYjQzZTA4OTlhOTdlM2Q3MGE3YjhiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDA5Njk0NjcsMjMuMTQ5NzU5MjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2ZlYzExYzQyOTRlNDJjOTg2YjNkOWM5NDg3NDk5MmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzlmZWZlMGU5ZjQ0NDI4M2ExNGM1NzkwYjBjNDQ4ZWUgPSAkKCc8ZGl2IGlkPSJodG1sXzM5ZmVmZTBlOWY0NDQyODNhMTRjNTc5MGIwYzQ0OGVlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6szrzPgM6/z4IsIM6UzpfOnM6fzqMgzprOoc6Rzp3Omc6UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zZmVjMTFjNDI5NGU0MmM5ODZiM2Q5Yzk0ODc0OTkyYy5zZXRDb250ZW50KGh0bWxfMzlmZWZlMGU5ZjQ0NDI4M2ExNGM1NzkwYjBjNDQ4ZWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDhkNjMyY2E4NWFiNDNlMDg5OWE5N2UzZDcwYTdiOGIuYmluZFBvcHVwKHBvcHVwXzNmZWMxMWM0Mjk0ZTQyYzk4NmIzZDljOTQ4NzQ5OTJjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UwNDc1M2YwMDFiNzQ4YjdiYjRjOGVhMGM4NjYzNzc2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDE4OTE4NjEsMjMuMTUxMTYzMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85YjBjNmIxN2IyM2E0MmVmODhlNzZjYzlhNDNmZDA3ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZDFlOWUxNTZlZjM0YzE5YjI0ZTlkN2UyYzBkY2MwMCA9ICQoJzxkaXYgaWQ9Imh0bWxfZmQxZTllMTU2ZWYzNGMxOWIyNGU5ZDdlMmMwZGNjMDAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6SzrvOsc+Hzr/PgM6/z4XOu86tzrnOus6xLCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWIwYzZiMTdiMjNhNDJlZjg4ZTc2Y2M5YTQzZmQwN2Quc2V0Q29udGVudChodG1sX2ZkMWU5ZTE1NmVmMzRjMTliMjRlOWQ3ZTJjMGRjYzAwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2UwNDc1M2YwMDFiNzQ4YjdiYjRjOGVhMGM4NjYzNzc2LmJpbmRQb3B1cChwb3B1cF85YjBjNmIxN2IyM2E0MmVmODhlNzZjYzlhNDNmZDA3ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yNDRmYmNjYzQ2NmI0MDgyOTY3YmRhZTg4MDIyNmE5NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQwNjg1NjU0LDIzLjEzNzUyMTc0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2UzMTkwMjY4OTMwZjRjMTliOGM1YmEzY2VkNzY1MjBlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc5NDBhZDc1ZWFhYTQzNTc5YjdkMDZiNDYxNTE1OGMwID0gJCgnPGRpdiBpZD0iaHRtbF83OTQwYWQ3NWVhYWE0MzU3OWI3ZDA2YjQ2MTUxNThjMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOrM68z4DOv8+CLCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTMxOTAyNjg5MzBmNGMxOWI4YzViYTNjZWQ3NjUyMGUuc2V0Q29udGVudChodG1sXzc5NDBhZDc1ZWFhYTQzNTc5YjdkMDZiNDYxNTE1OGMwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI0NGZiY2NjNDY2YjQwODI5NjdiZGFlODgwMjI2YTk3LmJpbmRQb3B1cChwb3B1cF9lMzE5MDI2ODkzMGY0YzE5YjhjNWJhM2NlZDc2NTIwZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NjU0NWVmYWZmYTA0ZjIyOWZiNTU3YjNlMDAyMDJkNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjM5NDc0ODY5LDIzLjEwNzYxMjYxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QyMzk3NDU3ZjQ2NTQyYjM5MmY3ODJlOWMwNTJkZjIzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ5OGUzMzA3YmY5YzRlYTQ4NGNlYWFiMjY1ZWYyOGNhID0gJCgnPGRpdiBpZD0iaHRtbF80OThlMzMwN2JmOWM0ZWE0ODRjZWFhYjI2NWVmMjhjYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpTOv8+Bzr/Pjc+GzrksIM6UzpfOnM6fzqMgzprOoc6Rzp3Omc6UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMjM5NzQ1N2Y0NjU0MmIzOTJmNzgyZTljMDUyZGYyMy5zZXRDb250ZW50KGh0bWxfNDk4ZTMzMDdiZjljNGVhNDg0Y2VhYWIyNjVlZjI4Y2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTY1NDVlZmFmZmEwNGYyMjlmYjU1N2IzZTAwMjAyZDYuYmluZFBvcHVwKHBvcHVwX2QyMzk3NDU3ZjQ2NTQyYjM5MmY3ODJlOWMwNTJkZjIzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2RlMWMyNjIxZjA0NzQ0MGM5YzI3YmQ4MTEzNzcxYTI4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDIyMTM0NCwyMy4xMTgwMjEwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mZjgzNGQ1NWY5NWU0MDY1YmEyNDU3MmNmZDI5N2QwYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNWNjY2ZhYTNlY2E0YTFkYWM2NTgyYmM2Njc1ODE1OCA9ICQoJzxkaXYgaWQ9Imh0bWxfZTVjY2NmYWEzZWNhNGExZGFjNjU4MmJjNjY3NTgxNTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6azr/Pgc+Jzr3Or8+CLCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmY4MzRkNTVmOTVlNDA2NWJhMjQ1NzJjZmQyOTdkMGIuc2V0Q29udGVudChodG1sX2U1Y2NjZmFhM2VjYTRhMWRhYzY1ODJiYzY2NzU4MTU4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RlMWMyNjIxZjA0NzQ0MGM5YzI3YmQ4MTEzNzcxYTI4LmJpbmRQb3B1cChwb3B1cF9mZjgzNGQ1NWY5NWU0MDY1YmEyNDU3MmNmZDI5N2QwYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZDhlYzEyYzJhMmU0OWM0YjMyY2I4ZmY4YzY2NDNjNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQyNTgzNDY2LDIzLjEzMzYyODg1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZjMmNlMTIyMmVkNjQyMTA4NTk4MTA2ZmM5YmVlNGQ4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzUwMjUzNTQyMTdiNzQxYmY4NDhmZGUwNDc2ODBkNjAwID0gJCgnPGRpdiBpZD0iaHRtbF81MDI1MzU0MjE3Yjc0MWJmODQ4ZmRlMDQ3NjgwZDYwMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDOsc+BzrHOu86vzrEgzqbOv8+Nz4HOvc+Jzr0sIM6UzpfOnM6fzqMgzprOoc6Rzp3Omc6UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82YzJjZTEyMjJlZDY0MjEwODU5ODEwNmZjOWJlZTRkOC5zZXRDb250ZW50KGh0bWxfNTAyNTM1NDIxN2I3NDFiZjg0OGZkZTA0NzY4MGQ2MDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWQ4ZWMxMmMyYTJlNDljNGIzMmNiOGZmOGM2NjQzYzQuYmluZFBvcHVwKHBvcHVwXzZjMmNlMTIyMmVkNjQyMTA4NTk4MTA2ZmM5YmVlNGQ4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc3YjZjZjNlZTZiMzRlODFiMjg0YTNlNjIyMjgxMzYwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuMzMyMTIyNzk5OTk5OTksMjMuMTMyNTQ1NDddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDA4MDdlYmJiZTRlNDUzYzlmOGRjYWVkNTY5NTkzZWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2IyN2RhYTZjNDExNGFkMDg1YzFhMDdhMGM1ZDIzNjYgPSAkKCc8ZGl2IGlkPSJodG1sXzdiMjdkYWE2YzQxMTRhZDA4NWMxYTA3YTBjNWQyMzY2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oks61z4HOss61z4HOv8+NzrTOsSwgzpTOl86czp/OoyDOms6hzpHOnc6ZzpTOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QwODA3ZWJiYmU0ZTQ1M2M5ZjhkY2FlZDU2OTU5M2VmLnNldENvbnRlbnQoaHRtbF83YjI3ZGFhNmM0MTE0YWQwODVjMWEwN2EwYzVkMjM2Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83N2I2Y2YzZWU2YjM0ZTgxYjI4NGEzZTYyMjI4MTM2MC5iaW5kUG9wdXAocG9wdXBfZDA4MDdlYmJiZTRlNDUzYzlmOGRjYWVkNTY5NTkzZWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzkwYWI5OGJiMjgzNDY3MTk0YWYxZmExM2E2OTI4ODEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4zMDQzMjEyOSwyMy4xNDI0NjU1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNjQ4ODlmMzFhZTA0MjRjYmRlNWUwODkzYjkxOWZjZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NWFkZTRkNjQ4M2M0NjM0YTMwMDQxZDQ4MzRmYWY0NSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzVhZGU0ZDY0ODNjNDYzNGEzMDA0MWQ0ODM0ZmFmNDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6nzrnOvc6vz4TPg86xLCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTY0ODg5ZjMxYWUwNDI0Y2JkZTVlMDg5M2I5MTlmY2Quc2V0Q29udGVudChodG1sXzc1YWRlNGQ2NDgzYzQ2MzRhMzAwNDFkNDgzNGZhZjQ1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM5MGFiOThiYjI4MzQ2NzE5NGFmMWZhMTNhNjkyODgxLmJpbmRQb3B1cChwb3B1cF8xNjQ4ODlmMzFhZTA0MjRjYmRlNWUwODkzYjkxOWZjZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yNjQzODIwMGQ3MGE0NmJhYmJjMzNkODAyNmQ1Mjg5OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjI5Mzk3OTY0LDIzLjE2MTEzODUzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJjMzEyZjcyNzIxNzQwMDM4NzU5MWU4OTYzNWQwMjM1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZjMDlhOTA3ZDA3ZTQ3MGQ4Mjc3YWI1MmVmYzFkYjU5ID0gJCgnPGRpdiBpZD0iaHRtbF82YzA5YTkwN2QwN2U0NzBkODI3N2FiNTJlZmMxZGI1OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprPjM+Dz4TOsSwgzpTOl86czp/OoyDOms6hzpHOnc6ZzpTOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJjMzEyZjcyNzIxNzQwMDM4NzU5MWU4OTYzNWQwMjM1LnNldENvbnRlbnQoaHRtbF82YzA5YTkwN2QwN2U0NzBkODI3N2FiNTJlZmMxZGI1OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yNjQzODIwMGQ3MGE0NmJhYmJjMzNkODAyNmQ1Mjg5OC5iaW5kUG9wdXAocG9wdXBfMmMzMTJmNzI3MjE3NDAwMzg3NTkxZTg5NjM1ZDAyMzUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjdlYzFiNTUzZTQ3NDZmMjg4NmVlZjYzMjM3NmM4MzEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4yOTMyNjI0OCwyMy4xODk0NDU0OTk5OTk5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzQyMmQzZjNhNzViNGFhZTgxYzU3ZGM5OWQ5ZmFiYjUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjc0YzgwODU5YzRjNDRmNTg1NTBmYjdmZWM2NzMyZDAgPSAkKCc8ZGl2IGlkPSJodG1sXzY3NGM4MDg1OWM0YzQ0ZjU4NTUwZmI3ZmVjNjczMmQwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Cts6zzrnOv8+CIM6RzrnOvM65zrvOuc6xzr3PjM+CLCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzQyMmQzZjNhNzViNGFhZTgxYzU3ZGM5OWQ5ZmFiYjUuc2V0Q29udGVudChodG1sXzY3NGM4MDg1OWM0YzQ0ZjU4NTUwZmI3ZmVjNjczMmQwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY3ZWMxYjU1M2U0NzQ2ZjI4ODZlZWY2MzIzNzZjODMxLmJpbmRQb3B1cChwb3B1cF9jNDIyZDNmM2E3NWI0YWFlODFjNTdkYzk5ZDlmYWJiNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NTNiZjNhNGFjYzk0YTU5OWY5OWNhOWYzZWJjMDMyYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQ1NDI1NDE1LDIzLjIzNDMzMzA0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MwYzk1ZDdhMzFlMjRkOWJiODNmYzhlYzRlZmVmNzhkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E5ZmQ5MWJjODMyNzQ2MWE5NTJiMjZjMzAzNzZhNTMzID0gJCgnPGRpdiBpZD0iaHRtbF9hOWZkOTFiYzgzMjc0NjFhOTUyYjI2YzMwMzc2YTUzMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpvOv8+FzrrOsc6Qz4TOuc6/zr0sIM6UzpfOnM6fzqMgzprOoc6Rzp3Omc6UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jMGM5NWQ3YTMxZTI0ZDliYjgzZmM4ZWM0ZWZlZjc4ZC5zZXRDb250ZW50KGh0bWxfYTlmZDkxYmM4MzI3NDYxYTk1MmIyNmMzMDM3NmE1MzMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzUzYmYzYTRhY2M5NGE1OTlmOTljYTlmM2ViYzAzMmEuYmluZFBvcHVwKHBvcHVwX2MwYzk1ZDdhMzFlMjRkOWJiODNmYzhlYzRlZmVmNzhkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJkZjIzODg2ZTFmMTQ5ZWRhYjEyMWEwMGQyYWY2OGY2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTE1MDUyOCwyMy4yMDQ2MTA4Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZGNmYjEzN2FlM2E0ZjRkYTUxMzI4ODgwMzE0YjhkMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jNmIwZjI3YmYyNDM0YjI2OWQwZDIxNDBlMDNkZjQ2YSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzZiMGYyN2JmMjQzNGIyNjlkMGQyMTQwZTAzZGY0NmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6hzqzOtM6/zr0sIM6UzpfOnM6fzqMgzprOoc6Rzp3Omc6UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZGNmYjEzN2FlM2E0ZjRkYTUxMzI4ODgwMzE0YjhkMC5zZXRDb250ZW50KGh0bWxfYzZiMGYyN2JmMjQzNGIyNjlkMGQyMTQwZTAzZGY0NmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmRmMjM4ODZlMWYxNDllZGFiMTIxYTAwZDJhZjY4ZjYuYmluZFBvcHVwKHBvcHVwXzlkY2ZiMTM3YWUzYTRmNGRhNTEzMjg4ODAzMTRiOGQwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBmZmVlYjRmZjJjMzRjNTA4MzAxNWFhNjgyZWFkM2IyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTA0NDc4NDUsMjMuMTc1MjQzMzhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmUzNWY4MzA3ODgwNGYzZWI1MGU3NmVjZDQxNTY4YmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTBjNzA5MjJjMDZjNDAwOThiMWI4ZjUwN2IyOGQzZmMgPSAkKCc8ZGl2IGlkPSJodG1sXzEwYzcwOTIyYzA2YzQwMDk4YjFiOGY1MDdiMjhkM2ZjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OoM61zrvOtc6uLCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmUzNWY4MzA3ODgwNGYzZWI1MGU3NmVjZDQxNTY4YmMuc2V0Q29udGVudChodG1sXzEwYzcwOTIyYzA2YzQwMDk4YjFiOGY1MDdiMjhkM2ZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBmZmVlYjRmZjJjMzRjNTA4MzAxNWFhNjgyZWFkM2IyLmJpbmRQb3B1cChwb3B1cF9iZTM1ZjgzMDc4ODA0ZjNlYjUwZTc2ZWNkNDE1NjhiYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MDBiMmE2NjQzNDA0MzEzYjg0NWFlYTc5OGM5MTIxYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQ0NzcxOTU3LDIzLjEyNDEwNzM2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg2Nzg0YWJhZjQxMDQ4YWFiZjhkNDg4OWE4MDkyOTY0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgwOTViMzU4MGRmNjQ0ZGY5NGM2NWY3OGViYjU0ZTkwID0gJCgnPGRpdiBpZD0iaHRtbF84MDk1YjM1ODBkZjY0NGRmOTRjNjVmNzhlYmI1NGU5MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqPOsc67zqzOvc+EzrnOv869LCDOlM6XzpzOn86jIM6azqHOkc6dzpnOlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODY3ODRhYmFmNDEwNDhhYWJmOGQ0ODg5YTgwOTI5NjQuc2V0Q29udGVudChodG1sXzgwOTViMzU4MGRmNjQ0ZGY5NGM2NWY3OGViYjU0ZTkwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYwMGIyYTY2NDM0MDQzMTNiODQ1YWVhNzk4YzkxMjFhLmJpbmRQb3B1cChwb3B1cF84Njc4NGFiYWY0MTA0OGFhYmY4ZDQ4ODlhODA5Mjk2NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82ZjQ1YmI0YjhiYWE0ZWNjYmE5NGMwNjBhNDJmNWY5NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQ1MzIwMTI5LDIzLjEwMDg4MTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzAyMDNlMGVlZmY3NzQ2ODlhNzM4M2MyODFjYjY3Y2UzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2IyMzQzZWEzMjNkMDQ0YTk5ZWJkZGUzMGIyZGQ5MzBmID0gJCgnPGRpdiBpZD0iaHRtbF9iMjM0M2VhMzIzZDA0NGE5OWViZGRlMzBiMmRkOTMwZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbOs865zr/PgiDOmc+JzqzOvc69zrfPgiwgzpTOl86czp/OoyDOms6hzpHOnc6ZzpTOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzAyMDNlMGVlZmY3NzQ2ODlhNzM4M2MyODFjYjY3Y2UzLnNldENvbnRlbnQoaHRtbF9iMjM0M2VhMzIzZDA0NGE5OWViZGRlMzBiMmRkOTMwZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82ZjQ1YmI0YjhiYWE0ZWNjYmE5NGMwNjBhNDJmNWY5NS5iaW5kUG9wdXAocG9wdXBfMDIwM2UwZWVmZjc3NDY4OWE3MzgzYzI4MWNiNjdjZTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTZmMmE4MWY2N2ZhNDYzZGJjNDc3ZTdmOTE0ZGM3Y2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NjcwNzAwMSwyMi43ODk1MDExOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mYjM4ZDMyN2FlNjA0Y2MxOWUwYjAxNGU1MjQ5NDhhMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kNTNkZmQ0ZWE1MmY0NWNiYTUzODdhZjFmYzJhMTc3YSA9ICQoJzxkaXYgaWQ9Imh0bWxfZDUzZGZkNGVhNTJmNDVjYmE1Mzg3YWYxZmMyYTE3N2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6cz4DOv8+Nz4HPhM62zrnOv869LCDOlM6XzpzOn86jIM6dzpHOpc6gzpvOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZiMzhkMzI3YWU2MDRjYzE5ZTBiMDE0ZTUyNDk0OGEwLnNldENvbnRlbnQoaHRtbF9kNTNkZmQ0ZWE1MmY0NWNiYTUzODdhZjFmYzJhMTc3YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNmYyYTgxZjY3ZmE0NjNkYmM0NzdlN2Y5MTRkYzdjYS5iaW5kUG9wdXAocG9wdXBfZmIzOGQzMjdhZTYwNGNjMTllMGIwMTRlNTI0OTQ4YTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODBhNWFjYTQyM2E2NGVmN2E4YzVkNTMxNTFlMjYyNTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NzUwNTAzNSwyMi43Mjk0NTQwNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNDYyNzM5YzMwNmQ0OGVkODBlMDM5ZmNlMjNiZjQyMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMGQ0OGM2NmQwOGY0NGZkYWRhY2E2MGMxMjkzNThmNyA9ICQoJzxkaXYgaWQ9Imh0bWxfMDBkNDhjNjZkMDhmNDRmZGFkYWNhNjBjMTI5MzU4ZjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6kzrfOvM6tzr3Ouc6/zr0sIM6UzpfOnM6fzqMgzpHOoc6Tzp/Opc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xNDYyNzM5YzMwNmQ0OGVkODBlMDM5ZmNlMjNiZjQyMC5zZXRDb250ZW50KGh0bWxfMDBkNDhjNjZkMDhmNDRmZGFkYWNhNjBjMTI5MzU4ZjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODBhNWFjYTQyM2E2NGVmN2E4YzVkNTMxNTFlMjYyNTUuYmluZFBvcHVwKHBvcHVwXzE0NjI3MzljMzA2ZDQ4ZWQ4MGUwMzlmY2UyM2JmNDIwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VkOTY2ZmJkZWIxYzRmYmQ4ZTIzZWJkZjAyOWEzMmI1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjExMzk2NzksMjIuNjcyNzkyNDNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjRlOTE1NDllMTc4NDEyMThlY2Q5ZWE1NGE3YzYyMDQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTQ3NGUxYjhjZTU0NDJiM2E4MWRkMjc3MWRhNjJlOTUgPSAkKCc8ZGl2IGlkPSJodG1sX2U0NzRlMWI4Y2U1NDQyYjNhODFkZDI3NzFkYTYyZTk1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms+MzrrOu86xLCDOlM6XzpzOn86jIM6RzqHOk86fzqXOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjRlOTE1NDllMTc4NDEyMThlY2Q5ZWE1NGE3YzYyMDQuc2V0Q29udGVudChodG1sX2U0NzRlMWI4Y2U1NDQyYjNhODFkZDI3NzFkYTYyZTk1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VkOTY2ZmJkZWIxYzRmYmQ4ZTIzZWJkZjAyOWEzMmI1LmJpbmRQb3B1cChwb3B1cF9iNGU5MTU0OWUxNzg0MTIxOGVjZDllYTU0YTdjNjIwNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNDEzZThhYzE2MmM0YmYwYWEwOTMwNTcxZjZjNzFlMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjY0MjIxMTkxLDIyLjY4NTI3MjIyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE0MDVjZWE3ODI4MjQ0NDU4MGI5MGVmMmI5YzM2Yjg1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE3MzU2NzVkM2M4MjRjODQ5MDRhZmVkODc0ZmViOTdjID0gJCgnPGRpdiBpZD0iaHRtbF8xNzM1Njc1ZDNjODI0Yzg0OTA0YWZlZDg3NGZlYjk3YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbOus6/zrLOsSwgzpTOl86czp/OoyDOkc6hzpPOn86lzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE0MDVjZWE3ODI4MjQ0NDU4MGI5MGVmMmI5YzM2Yjg1LnNldENvbnRlbnQoaHRtbF8xNzM1Njc1ZDNjODI0Yzg0OTA0YWZlZDg3NGZlYjk3Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iNDEzZThhYzE2MmM0YmYwYWEwOTMwNTcxZjZjNzFlMy5iaW5kUG9wdXAocG9wdXBfMTQwNWNlYTc4MjgyNDQ0NTgwYjkwZWYyYjljMzZiODUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGExZmMwNWQ3ZTYxNGUzOTk1ZjI0ZDhjODZmZTZmNjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NjU4NDM5NiwyMi43MDQ5NTQxNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYzAwMGEwZGMzZDA0Nzc5ODQwMmM0OTY0ODQ3MjViYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82MjRmZjE0MTE4NzI0YjQ2YTU5ZGQ2NmYzNTQ3MWY2NCA9ICQoJzxkaXYgaWQ9Imh0bWxfNjI0ZmYxNDExODcyNGI0NmE1OWRkNjZmMzU0NzFmNjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6RzrPOr86xIM6RzrnOus6xz4TOtc+Bzq/Ovc63LCDOlM6XzpzOn86jIM6azp/Opc6kzqPOn86gzp/OlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2MwMDBhMGRjM2QwNDc3OTg0MDJjNDk2NDg0NzI1YmMuc2V0Q29udGVudChodG1sXzYyNGZmMTQxMTg3MjRiNDZhNTlkZDY2ZjM1NDcxZjY0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhhMWZjMDVkN2U2MTRlMzk5NWYyNGQ4Yzg2ZmU2ZjYyLmJpbmRQb3B1cChwb3B1cF9jYzAwMGEwZGMzZDA0Nzc5ODQwMmM0OTY0ODQ3MjViYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZTYxOGVjNGNlNjE0NmNjYTYyYWI3N2EzYWZiZDZlOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjY2Mjc5NjAyLDIyLjY5ODU5ODg2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNkMjlhOTQ3YTBkZDQ1NWRhN2VhMTc1MjM1ODQzMTJmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYxOGI2YzlmYjhkMjQ0ZTk5ZDMzZDhmZWE3YjVjZTczID0gJCgnPGRpdiBpZD0iaHRtbF82MThiNmM5ZmI4ZDI0NGU5OWQzM2Q4ZmVhN2I1Y2U3MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpHOtc+Bzr/OtM+Bz4zOvM65zr8sIM6UzpfOnM6fzqMgzprOn86lzqTOo86fzqDOn86UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zZDI5YTk0N2EwZGQ0NTVkYTdlYTE3NTIzNTg0MzEyZi5zZXRDb250ZW50KGh0bWxfNjE4YjZjOWZiOGQyNDRlOTlkMzNkOGZlYTdiNWNlNzMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2U2MThlYzRjZTYxNDZjY2E2MmFiNzdhM2FmYmQ2ZTguYmluZFBvcHVwKHBvcHVwXzNkMjlhOTQ3YTBkZDQ1NWRhN2VhMTc1MjM1ODQzMTJmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VmMWJjNzkwMmU3ODRkNDFiZjc4ZGUyN2Y0NzUyYzgyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjc5MTMwNTUsMjIuNjg1MTM2ODAwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzgzZGNkNDJhNTQ4ODRkYWY4NGY4OGZjNzUwMmE2YmI1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNlZDlkMmU4OWFiOTRmZTFiZTYxNzliYzkwNjMyY2FmID0gJCgnPGRpdiBpZD0iaHRtbF8zZWQ5ZDJlODlhYjk0ZmUxYmU2MTc5YmM5MDYzMmNhZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDOsc69z4zPgc6xzrzOsSwgzpTOl86czp/OoyDOms6fzqXOpM6jzp/OoM6fzpTOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzgzZGNkNDJhNTQ4ODRkYWY4NGY4OGZjNzUwMmE2YmI1LnNldENvbnRlbnQoaHRtbF8zZWQ5ZDJlODlhYjk0ZmUxYmU2MTc5YmM5MDYzMmNhZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lZjFiYzc5MDJlNzg0ZDQxYmY3OGRlMjdmNDc1MmM4Mi5iaW5kUG9wdXAocG9wdXBfODNkY2Q0MmE1NDg4NGRhZjg0Zjg4ZmM3NTAyYTZiYjUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmVkMDVlNTI4M2RmNDhmYjg5NTk5YWUwMDViMTlkNmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42ODM5ODI4NSwyMi42Njg1NjAwM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYjY1MTNkZTI5Mjc0MGIwODFjYTFjOTNjZDQxMjAwMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MzE0YTc0ODRkNWQ0NjU0YmZkYjE2ZWMxODJkZGE4NyA9ICQoJzxkaXYgaWQ9Imh0bWxfNTMxNGE3NDg0ZDVkNDY1NGJmZGIxNmVjMTgyZGRhODciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6jz43Ovc6/z4HOvywgzpTOl86czp/OoyDOms6fzqXOpM6jzp/OoM6fzpTOmc6fzqU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FiNjUxM2RlMjkyNzQwYjA4MWNhMWM5M2NkNDEyMDAyLnNldENvbnRlbnQoaHRtbF81MzE0YTc0ODRkNWQ0NjU0YmZkYjE2ZWMxODJkZGE4Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mZWQwNWU1MjgzZGY0OGZiODk1OTlhZTAwNWIxOWQ2ZC5iaW5kUG9wdXAocG9wdXBfYWI2NTEzZGUyOTI3NDBiMDgxY2ExYzkzY2Q0MTIwMDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjY0MjM1NDg1OWMyNDNlMDhiNTkxOWQ1YzU2MTQyZTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MTc1NjEzNCwyMi41NTA0NTUwOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNzUzYzc3MDQzYjY0MjEyODUxMTBlM2IyY2VmODgxNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NWQ1MDIyYzE5YzE0Y2RiYTI5ZDM3YjUzMTRjOWM5YSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzVkNTAyMmMxOWMxNGNkYmEyOWQzN2I1MzE0YzljOWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Rz4HOr86xLCDOlM6XzpzOn86jIM6bzqXOoc6azpXOmc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI3NTNjNzcwNDNiNjQyMTI4NTExMGUzYjJjZWY4ODE1LnNldENvbnRlbnQoaHRtbF83NWQ1MDIyYzE5YzE0Y2RiYTI5ZDM3YjUzMTRjOWM5YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yNjQyMzU0ODU5YzI0M2UwOGI1OTE5ZDVjNTYxNDJlMC5iaW5kUG9wdXAocG9wdXBfMjc1M2M3NzA0M2I2NDIxMjg1MTEwZTNiMmNlZjg4MTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTA0MTJkNjgzNjhlNGEyNjg0ODg1NmYxNzRkMjkwYjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MzQ4NDE5MiwyMi41Nzg0NzIxNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wYTE4ZDIzMDY1ZGI0NGYwYjcwYTY0N2VlYTJjYTRmNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80OWVhZjQ4NDYyOWQ0YzNmOGFkMzlmZTA4MWYxMTQ1NyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDllYWY0ODQ2MjlkNGMzZjhhZDM5ZmUwODFmMTE0NTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6TzrHOu86xzr3Osc6vzrnOus6xLCDOlM6XzpzOn86jIM6bzqXOoc6azpXOmc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBhMThkMjMwNjVkYjQ0ZjBiNzBhNjQ3ZWVhMmNhNGY0LnNldENvbnRlbnQoaHRtbF80OWVhZjQ4NDYyOWQ0YzNmOGFkMzlmZTA4MWYxMTQ1Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MDQxMmQ2ODM2OGU0YTI2ODQ4ODU2ZjE3NGQyOTBiNy5iaW5kUG9wdXAocG9wdXBfMGExOGQyMzA2NWRiNDRmMGI3MGE2NDdlZWEyY2E0ZjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTkxYTk4MmI2OWRiNGNkZjgxY2IxM2JkMGJlYWZlYWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MjAzMjcsMjIuNTg0ODY1NTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzhmZDFjYzI0NmM4NDQxNjg3MWM5YjRjYWE2OWIzNzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDYyMzI2OTZmYTQxNGVmYTk3YzNhOWQ3MmI1MDFhMTUgPSAkKCc8ZGl2IGlkPSJodG1sXzQ2MjMyNjk2ZmE0MTRlZmE5N2MzYTlkNzJiNTAxYTE1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oo8+Ez4HOsc6yzq4gzqHOrM+HzrcsIM6UzpfOnM6fzqMgzpvOpc6hzprOlc6ZzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzhmZDFjYzI0NmM4NDQxNjg3MWM5YjRjYWE2OWIzNzEuc2V0Q29udGVudChodG1sXzQ2MjMyNjk2ZmE0MTRlZmE5N2MzYTlkNzJiNTAxYTE1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E5MWE5ODJiNjlkYjRjZGY4MWNiMTNiZDBiZWFmZWFlLmJpbmRQb3B1cChwb3B1cF83OGZkMWNjMjQ2Yzg0NDE2ODcxYzliNGNhYTY5YjM3MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kZjlkOWVjODhkZmI0MDkyOGQwYjViY2ZmMjNhYTYzNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYzNTM0MTY0LDIyLjU5NjA5NDEzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U4NGJlNDA2YzE4ZTQ5YTE4ZTc5NTcxZjBjZWZiNzFmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzcwZWI3YWQ0M2MxMzQ4MTI4YjhjZDc2NzBkMzY3MGZhID0gJCgnPGRpdiBpZD0iaHRtbF83MGViN2FkNDNjMTM0ODEyOGI4Y2Q3NjcwZDM2NzBmYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpPOsc67zqzPhM65zr/OvSwgzpTOl86czp/OoyDOm86lzqHOms6VzpnOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lODRiZTQwNmMxOGU0OWExOGU3OTU3MWYwY2VmYjcxZi5zZXRDb250ZW50KGh0bWxfNzBlYjdhZDQzYzEzNDgxMjhiOGNkNzY3MGQzNjcwZmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGY5ZDllYzg4ZGZiNDA5MjhkMGI1YmNmZjIzYWE2MzUuYmluZFBvcHVwKHBvcHVwX2U4NGJlNDA2YzE4ZTQ5YTE4ZTc5NTcxZjBjZWZiNzFmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkxNjBjODQxYjk3NTQwM2E4ZjlmY2ZiNDkzZDY1ZGVhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjIwNTE3NzMsMjIuNTk2MTU3MDddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzJjZTc0MTNmMDNiNDAwMGFmNDA0NTQ0OTAyOGUwZjUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODMyYWE5MDRiZWYxNGNmOTgwZjI2MTRjMTY4NjA3ZGIgPSAkKCc8ZGl2IGlkPSJodG1sXzgzMmFhOTA0YmVmMTRjZjk4MGYyNjE0YzE2ODYwN2RiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc6zz4HOuc67zq/PhM+DzrEsIM6UzpfOnM6fzqMgzpvOpc6hzprOlc6ZzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzJjZTc0MTNmMDNiNDAwMGFmNDA0NTQ0OTAyOGUwZjUuc2V0Q29udGVudChodG1sXzgzMmFhOTA0YmVmMTRjZjk4MGYyNjE0YzE2ODYwN2RiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzkxNjBjODQxYjk3NTQwM2E4ZjlmY2ZiNDkzZDY1ZGVhLmJpbmRQb3B1cChwb3B1cF83MmNlNzQxM2YwM2I0MDAwYWY0MDQ1NDQ5MDI4ZTBmNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hODA1MmNhZTEzNGU0MTI1YmU1M2ZiYjBlNTRlMGY0OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYxNTI1NzI2LDIyLjYwNjExMzQzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q5YWQ5NzMxZGYzMTRkMzFhZDhjOTYwNmYxNjI3YzY5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E2YjI5NDViZWNkZjQ2NmZhYjZiOWRlMzRjMDIzNTc2ID0gJCgnPGRpdiBpZD0iaHRtbF9hNmIyOTQ1YmVjZGY0NjZmYWI2YjlkZTM0YzAyMzU3NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqPPgM6xzr3Osc6vzrnOus6xLCDOlM6XzpzOn86jIM6bzqXOoc6azpXOmc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q5YWQ5NzMxZGYzMTRkMzFhZDhjOTYwNmYxNjI3YzY5LnNldENvbnRlbnQoaHRtbF9hNmIyOTQ1YmVjZGY0NjZmYWI2YjlkZTM0YzAyMzU3Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hODA1MmNhZTEzNGU0MTI1YmU1M2ZiYjBlNTRlMGY0OC5iaW5kUG9wdXAocG9wdXBfZDlhZDk3MzFkZjMxNGQzMWFkOGM5NjA2ZjE2MjdjNjkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmM2YjQ4MzA4MTc4NDNhNWIwN2I4MjI5Zjg3NzkyMDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42Mjk5ODU4MSwyMi42NTIxMDkxNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMjRlM2Q1MDI2MTQ0OWY0OWRlM2NlMmQ0Njc2Zjk5MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lYzRjZTJkYjY2MWQ0ZGU1YjExYTEzYWU1OWM3NDlmNyA9ICQoJzxkaXYgaWQ9Imh0bWxfZWM0Y2UyZGI2NjFkNGRlNWIxMWExM2FlNTljNzQ5ZjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6nzr/Pjc69zrcsIM6UzpfOnM6fzqMgzpvOpc6hzprOlc6ZzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTI0ZTNkNTAyNjE0NDlmNDlkZTNjZTJkNDY3NmY5OTEuc2V0Q29udGVudChodG1sX2VjNGNlMmRiNjYxZDRkZTViMTFhMTNhZTU5Yzc0OWY3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJjNmI0ODMwODE3ODQzYTViMDdiODIyOWY4Nzc5MjAwLmJpbmRQb3B1cChwb3B1cF9lMjRlM2Q1MDI2MTQ0OWY0OWRlM2NlMmQ0Njc2Zjk5MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mMTkwYmE4YTZhZjY0NzNlYjNjMWFiNGMzNDIyMzFlNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYyMjY4MDY2LDIyLjYyNDI1MDQxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U4NjA3N2QzZGVmMzRhMjQ4NzI2N2I3OWI1M2RhZWVlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YxMjQ4ZGJiYzBkOTQ4NWM4ZjY3MzAwNGJmNWNhMjIxID0gJCgnPGRpdiBpZD0iaHRtbF9mMTI0OGRiYmMwZDk0ODVjOGY2NzMwMDRiZjVjYTIyMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpzPgM6/zrbOuc6/zr3Otc67zrHOr865zrrOsSwgzpTOl86czp/OoyDOm86lzqHOms6VzpnOkc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lODYwNzdkM2RlZjM0YTI0ODcyNjdiNzliNTNkYWVlZS5zZXRDb250ZW50KGh0bWxfZjEyNDhkYmJjMGQ5NDg1YzhmNjczMDA0YmY1Y2EyMjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjE5MGJhOGE2YWY2NDczZWIzYzFhYjRjMzQyMjMxZTYuYmluZFBvcHVwKHBvcHVwX2U4NjA3N2QzZGVmMzRhMjQ4NzI2N2I3OWI1M2RhZWVlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U0ZTAzNWZkMGJhMDQ2MjU5YzM4NTlhNmRhNDkwNTIxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjQyOTYzNDEsMjIuNTczMjgwMzNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjI1MjBjZWMwYjA5NDU3ZmFlMDg3NWZhY2Y3NGU0NmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWY3OGM3YTVhZTdmNDU1ZTgxN2UwYjJlZmM2MzRkZmUgPSAkKCc8ZGl2IGlkPSJodG1sX2FmNzhjN2E1YWU3ZjQ1NWU4MTdlMGIyZWZjNjM0ZGZlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Cts6zzrnOv8+CIM6TzrXPjs+BzrPOuc6/z4IsIM6UzpfOnM6fzqMgzpvOpc6hzprOlc6ZzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjI1MjBjZWMwYjA5NDU3ZmFlMDg3NWZhY2Y3NGU0NmIuc2V0Q29udGVudChodG1sX2FmNzhjN2E1YWU3ZjQ1NWU4MTdlMGIyZWZjNjM0ZGZlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U0ZTAzNWZkMGJhMDQ2MjU5YzM4NTlhNmRhNDkwNTIxLmJpbmRQb3B1cChwb3B1cF8yMjUyMGNlYzBiMDk0NTdmYWUwODc1ZmFjZjc0ZTQ2Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hNjIyZmZjYTcxNjE0NjM4ODkxNzQ1YjE5YmIyOTYzMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYwMjk2NjMxLDIyLjU1OTA4Nzc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA3M2NmYjUwM2JjMTRkNTk5M2VhYTY5ZDAwNTQxMWJlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkxNzk0MTZiZjFmMzQwYjE5YTMyOTQ1YTM5ODkxYzg4ID0gJCgnPGRpdiBpZD0iaHRtbF85MTc5NDE2YmYxZjM0MGIxOWEzMjk0NWEzOTg5MWM4OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpzOtc+BzrrOv8+Nz4HOuc6/zr0sIM6UzpfOnM6fzqMgzpvOpc6hzprOlc6ZzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDczY2ZiNTAzYmMxNGQ1OTkzZWFhNjlkMDA1NDExYmUuc2V0Q29udGVudChodG1sXzkxNzk0MTZiZjFmMzQwYjE5YTMyOTQ1YTM5ODkxYzg4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E2MjJmZmNhNzE2MTQ2Mzg4OTE3NDViMTliYjI5NjMxLmJpbmRQb3B1cChwb3B1cF8wNzNjZmI1MDNiYzE0ZDU5OTNlYWE2OWQwMDU0MTFiZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MGUyYjE4NDczMzg0MTA3YTAxN2Q0MzA3MWU3MzdkNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYwNjM5NTcyLDIyLjYxNjEzODQ2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcyMDY2OGExMWQyNDRiZjhhZDQxMWE1MmQ3NzQwZTVkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MzODI3ZjcyZDVlZTRhODA5M2E3YjZhZGE0MDM0Y2I4ID0gJCgnPGRpdiBpZD0iaHRtbF9jMzgyN2Y3MmQ1ZWU0YTgwOTNhN2I2YWRhNDAzNGNiOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbOs865zr/PgiDOo8+Ezq3Phs6xzr3Ov8+CLCDOlM6XzpzOn86jIM6bzqXOoc6azpXOmc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzcyMDY2OGExMWQyNDRiZjhhZDQxMWE1MmQ3NzQwZTVkLnNldENvbnRlbnQoaHRtbF9jMzgyN2Y3MmQ1ZWU0YTgwOTNhN2I2YWRhNDAzNGNiOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MGUyYjE4NDczMzg0MTA3YTAxN2Q0MzA3MWU3MzdkNS5iaW5kUG9wdXAocG9wdXBfNzIwNjY4YTExZDI0NGJmOGFkNDExYTUyZDc3NDBlNWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWU0Nzk4NWVhZTdjNDg0M2I4ZmY2YzNhMDQxMTAyOGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41OTE5MzAzOSwyMi41MjM2NDE1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81NDFmYThhYzc5YjI0ZjBhYmI0NGRlMWFhNjZhZTlhYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MWVjZDVlNjA3ODg0OTE3YTk0ZDg1NTU5ZDdmYjZjYyA9ICQoJzxkaXYgaWQ9Imh0bWxfODFlY2Q1ZTYwNzg4NDkxN2E5NGQ4NTU1OWQ3ZmI2Y2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6kzr/Phc+Bzr3Or866zrnOv869LCDOlM6XzpzOn86jIM6RzqHOk86fzqXOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTQxZmE4YWM3OWIyNGYwYWJiNDRkZTFhYTY2YWU5YWMuc2V0Q29udGVudChodG1sXzgxZWNkNWU2MDc4ODQ5MTdhOTRkODU1NTlkN2ZiNmNjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFlNDc5ODVlYWU3YzQ4NDNiOGZmNmMzYTA0MTEwMjhlLmJpbmRQb3B1cChwb3B1cF81NDFmYThhYzc5YjI0ZjBhYmI0NGRlMWFhNjZhZTlhYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84Y2NhODNkNTVkZjM0ZmNiOWNiM2VlNGE4ZDJmNjEyMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU3MzgzMzQ3LDIyLjU2NDQwOTI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY3OTg5ZGM5OWQ1ZTQ0ZmQ5N2Y2YTJhMzdiZjU0YzVkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y3ZjRhMTdiZmQ2YjRkMjhiZTYzNWNkOTYwZTMxY2I5ID0gJCgnPGRpdiBpZD0iaHRtbF9mN2Y0YTE3YmZkNmI0ZDI4YmU2MzVjZDk2MGUzMWNiOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprPgc+NzrEgzpLPgc+Nz4POtywgzpTOl86czp/OoyDOkc6hzpPOn86lzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY3OTg5ZGM5OWQ1ZTQ0ZmQ5N2Y2YTJhMzdiZjU0YzVkLnNldENvbnRlbnQoaHRtbF9mN2Y0YTE3YmZkNmI0ZDI4YmU2MzVjZDk2MGUzMWNiOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84Y2NhODNkNTVkZjM0ZmNiOWNiM2VlNGE4ZDJmNjEyMy5iaW5kUG9wdXAocG9wdXBfNjc5ODlkYzk5ZDVlNDRmZDk3ZjZhMmEzN2JmNTRjNWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzc2ZmRhNTEzNDVkNGY0MzgyZGQyMGVjZjRjODkwYTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41ODY0NDQ4NSwyMi42MTkxMTIwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNGY2YThkNDY4YTI0NTI1OWM5MGRiODE4MDNhYjMxYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jZTAwYWM0MWQwODk0M2EwOWNmZDZhYjE0MzA3YjM1ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfY2UwMGFjNDFkMDg5NDNhMDljZmQ2YWIxNDMwN2IzNWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Wz4zOs866zrEsIM6UzpfOnM6fzqMgzpHOoc6Tzp/Opc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNGY2YThkNDY4YTI0NTI1OWM5MGRiODE4MDNhYjMxYS5zZXRDb250ZW50KGh0bWxfY2UwMGFjNDFkMDg5NDNhMDljZmQ2YWIxNDMwN2IzNWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzc2ZmRhNTEzNDVkNGY0MzgyZGQyMGVjZjRjODkwYTcuYmluZFBvcHVwKHBvcHVwX2Q0ZjZhOGQ0NjhhMjQ1MjU5YzkwZGI4MTgwM2FiMzFhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzlkMTg1MjdlYTk5NTQ1Mzg4MDBjZjY5YmY1MWFjZWUwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTkwMTc5NDQsMjIuNjU5OTY3NDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDM1NWQxODllYzM3NDJjNDgyOTIwOTI1YTQ2ZDFhM2QgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjNkYWIxM2VmY2ZiNGJkMTgwMzk1ODQxZDQ3NmVkNjUgPSAkKCc8ZGl2IGlkPSJodG1sX2IzZGFiMTNlZmNmYjRiZDE4MDM5NTg0MWQ0NzZlZDY1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Olc67zrvOt869zrnOus+Mzr0sIM6UzpfOnM6fzqMgzpHOoc6Tzp/Opc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80MzU1ZDE4OWVjMzc0MmM0ODI5MjA5MjVhNDZkMWEzZC5zZXRDb250ZW50KGh0bWxfYjNkYWIxM2VmY2ZiNGJkMTgwMzk1ODQxZDQ3NmVkNjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWQxODUyN2VhOTk1NDUzODgwMGNmNjliZjUxYWNlZTAuYmluZFBvcHVwKHBvcHVwXzQzNTVkMTg5ZWMzNzQyYzQ4MjkyMDkyNWE0NmQxYTNkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI4YTY2NWY2MDlhZTRkOWNhZGYxOGU3YTA0MTlkZDhhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTE3Mzk1MDIsMjIuNjkzODg1OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wYzM5OWQ2OTY5MjQ0Yjg5ODM3NGVlMjU4OWJkZmY5ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MjViN2M5YTAyMGM0ODViYjQ0OTdjNTU2M2JmZTM4ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfODI1YjdjOWEwMjBjNDg1YmI0NDk3YzU1NjNiZmUzOGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6jz4DOt867zrnPic+EzqzOus63z4IsIM6UzpfOnM6fzqMgzpvOlc6hzp3Okc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wYzM5OWQ2OTY5MjQ0Yjg5ODM3NGVlMjU4OWJkZmY5Zi5zZXRDb250ZW50KGh0bWxfODI1YjdjOWEwMjBjNDg1YmI0NDk3YzU1NjNiZmUzOGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjhhNjY1ZjYwOWFlNGQ5Y2FkZjE4ZTdhMDQxOWRkOGEuYmluZFBvcHVwKHBvcHVwXzBjMzk5ZDY5NjkyNDRiODk4Mzc0ZWUyNTg5YmRmZjlmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU0NDkzNmRhMjAzYjQwYTY4YjE4YTllNDdlZjY5ODA0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTMxMDU1NDUsMjIuNzEwNTMxMjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGM3NTE3YmZmM2JhNGVmMDlmMmY3ZjE0YWFkOGRlYzYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2Q2MGJjNzVkZDY3NGI3ZGI3OThjN2I2Nzk2OTdlODggPSAkKCc8ZGl2IGlkPSJodG1sXzdkNjBiYzc1ZGQ2NzRiN2RiNzk4YzdiNjc5Njk3ZTg4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6/z4XOs86xzq/Ouc66zrEsIM6UzpfOnM6fzqMgzpvOlc6hzp3Okc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80Yzc1MTdiZmYzYmE0ZWYwOWYyZjdmMTRhYWQ4ZGVjNi5zZXRDb250ZW50KGh0bWxfN2Q2MGJjNzVkZDY3NGI3ZGI3OThjN2I2Nzk2OTdlODgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTQ0OTM2ZGEyMDNiNDBhNjhiMThhOWU0N2VmNjk4MDQuYmluZFBvcHVwKHBvcHVwXzRjNzUxN2JmZjNiYTRlZjA5ZjJmN2YxNGFhZDhkZWM2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhmOThkZmQyMmM0ZTRjZTRhMWE5ZGUyNTk2MDcyOTk1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTUwMTcwOSwyMi43MTU3MzYzOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MzA4NjlmNTg5NmQ0MjRkYWEzMWE2Y2MyZDQ4ODE2YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MTcyZTY0YTZiMWM0ZDRiODMxY2M2MDliZWQzYTU0MiA9ICQoJzxkaXYgaWQ9Imh0bWxfNTE3MmU2NGE2YjFjNGQ0YjgzMWNjNjA5YmVkM2E1NDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6cz43Ou86/zrksIM6UzpfOnM6fzqMgzpvOlc6hzp3Okc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80MzA4NjlmNTg5NmQ0MjRkYWEzMWE2Y2MyZDQ4ODE2YS5zZXRDb250ZW50KGh0bWxfNTE3MmU2NGE2YjFjNGQ0YjgzMWNjNjA5YmVkM2E1NDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGY5OGRmZDIyYzRlNGNlNGExYTlkZTI1OTYwNzI5OTUuYmluZFBvcHVwKHBvcHVwXzQzMDg2OWY1ODk2ZDQyNGRhYTMxYTZjYzJkNDg4MTZhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ3ODc1YjMzMzZhYTQxY2E5YTEyYzM5YzQwMDg1OTJlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTMwNjI0MzksMjIuNjkyMjEzMDZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2YyYzA3YTcwOTg1NGQ3Y2I2MGM0MzMxYzRjYTdjOTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmU4MThhZTExMGNlNGI0OWEwMmIxOTk1OGI4YzllMzcgPSAkKCc8ZGl2IGlkPSJodG1sXzJlODE4YWUxMTBjZTRiNDlhMDJiMTk5NThiOGM5ZTM3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6xzrvOsc68zqzOus65zr/OvSwgzpTOl86czp/OoyDOm86VzqHOnc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NmMmMwN2E3MDk4NTRkN2NiNjBjNDMzMWM0Y2E3Yzk1LnNldENvbnRlbnQoaHRtbF8yZTgxOGFlMTEwY2U0YjQ5YTAyYjE5OTU4YjhjOWUzNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80Nzg3NWIzMzM2YWE0MWNhOWExMmMzOWM0MDA4NTkyZS5iaW5kUG9wdXAocG9wdXBfY2YyYzA3YTcwOTg1NGQ3Y2I2MGM0MzMxYzRjYTdjOTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWUzNTlmODYwODc2NDk4N2E3MDMxZDk0NjllZDNlN2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NzM0NTIsMjIuNjkzNjc5ODFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzVjOTlkZGU3MzdhNDhjNGIyNzgzM2ZjMWNiZTYzY2QgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTU0ZTNkZjgxNDk4NGZmY2ExNGRlMWY5ZDg2MWE3NTggPSAkKCc8ZGl2IGlkPSJodG1sXzk1NGUzZGY4MTQ5ODRmZmNhMTRkZTFmOWQ4NjFhNzU4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OlM65z4fOrM67zrnOsSwgzpTOl86czp/OoyDOm86VzqHOnc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM1Yzk5ZGRlNzM3YTQ4YzRiMjc4MzNmYzFjYmU2M2NkLnNldENvbnRlbnQoaHRtbF85NTRlM2RmODE0OTg0ZmZjYTE0ZGUxZjlkODYxYTc1OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lZTM1OWY4NjA4NzY0OTg3YTcwMzFkOTQ2OWVkM2U3ZS5iaW5kUG9wdXAocG9wdXBfMzVjOTlkZGU3MzdhNDhjNGIyNzgzM2ZjMWNiZTYzY2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODI2MGZkM2JhYzMzNDI0ZGJiMTk0NGJiY2VmODA0OTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NjU1Mjg4NywyMi43MTc1NDQ1Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMTI2NGZlZTg5MDY0NTcwYTc3ZThmNjQ4OTM3N2U5NCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83ZWZjMjI0NTRlZmI0YjE1OWQ5ZjZlYjYzM2I5NDU2ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfN2VmYzIyNDU0ZWZiNGIxNTlkOWY2ZWI2MzNiOTQ1NmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6RzrvOvM+Fz4HPjM+CLCDOlM6XzpzOn86jIM6bzpXOoc6dzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTEyNjRmZWU4OTA2NDU3MGE3N2U4ZjY0ODkzNzdlOTQuc2V0Q29udGVudChodG1sXzdlZmMyMjQ1NGVmYjRiMTU5ZDlmNmViNjMzYjk0NTZmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzgyNjBmZDNiYWMzMzQyNGRiYjE5NDRiYmNlZjgwNDk4LmJpbmRQb3B1cChwb3B1cF9lMTI2NGZlZTg5MDY0NTcwYTc3ZThmNjQ4OTM3N2U5NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZTcwZTZiNTZmZmU0ZGM2OWIyN2Y2ZGJjOTE1OGE3YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU4OTk1MDU2LDIyLjcwNTQ1MTk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RmMTc2ZWU1NDVkMTQyZWY4NmMwMDAwZTRkODI0YTJhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc2YjMyYTM5MmRhMTQ4YTE4NWJlNmNhOGJmOTQ2NDI1ID0gJCgnPGRpdiBpZD0iaHRtbF83NmIzMmEzOTJkYTE0OGExODViZTZjYThiZjk0NjQyNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpzOsc6zzr/Pjc67zrEsIM6UzpfOnM6fzqMgzpHOoc6Tzp/Opc6jPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kZjE3NmVlNTQ1ZDE0MmVmODZjMDAwMGU0ZDgyNGEyYS5zZXRDb250ZW50KGh0bWxfNzZiMzJhMzkyZGExNDhhMTg1YmU2Y2E4YmY5NDY0MjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2U3MGU2YjU2ZmZlNGRjNjliMjdmNmRiYzkxNThhN2MuYmluZFBvcHVwKHBvcHVwX2RmMTc2ZWU1NDVkMTQyZWY4NmMwMDAwZTRkODI0YTJhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzlkNjU1YmM3ZGMxNDRhYmNhMTZjMjJjZjIxNDA2MmRlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjgzMzE5MDksMjIuNTk3NDk5ODVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWViMGY3YzVjMDU2NGJhN2EwMTkyYjU2NGJhMThkMGUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWQxNjEzMDBiZjBkNDAxMzg5ZWVlNTQxOTViYzE2ZDkgPSAkKCc8ZGl2IGlkPSJodG1sX2FkMTYxMzAwYmYwZDQwMTM4OWVlZTU0MTk1YmMxNmQ5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Op86tzrvOvM63z4IsIM6UzpfOnM6fzqMgzprOn86lzqTOo86fzqDOn86UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xZWIwZjdjNWMwNTY0YmE3YTAxOTJiNTY0YmExOGQwZS5zZXRDb250ZW50KGh0bWxfYWQxNjEzMDBiZjBkNDAxMzg5ZWVlNTQxOTViYzE2ZDkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWQ2NTViYzdkYzE0NGFiY2ExNmMyMmNmMjE0MDYyZGUuYmluZFBvcHVwKHBvcHVwXzFlYjBmN2M1YzA1NjRiYTdhMDE5MmI1NjRiYTE4ZDBlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzExZGU4MTFkNTI3MjQ0NDA4YjE1OGEzN2I0ODVmYTdhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjYxMjczOTYsMjIuNjYwOTM4MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDIyN2UxMmYwNGZiNGU2ZGFiODk5NDE0Y2FiNWMzMmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzA0Zjc1ZTk3MzMwNDcwMGEyYTE2ZmJmM2Q5Mjc3NjEgPSAkKCc8ZGl2IGlkPSJodG1sX2MwNGY3NWU5NzMzMDQ3MDBhMmExNmZiZjNkOTI3NzYxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oo8+EzrHOuM6xzq/Ouc66zrEsIM6UzpfOnM6fzqMgzprOn86lzqTOo86fzqDOn86UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMjI3ZTEyZjA0ZmI0ZTZkYWI4OTk0MTRjYWI1YzMyYi5zZXRDb250ZW50KGh0bWxfYzA0Zjc1ZTk3MzMwNDcwMGEyYTE2ZmJmM2Q5Mjc3NjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTFkZTgxMWQ1MjcyNDQ0MDhiMTU4YTM3YjQ4NWZhN2EuYmluZFBvcHVwKHBvcHVwXzAyMjdlMTJmMDRmYjRlNmRhYjg5OTQxNGNhYjVjMzJiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Q1YTZlYTE5MjU0YjQyYWY5OGI5MGRjYWU5ZDAzOTJmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjY1Nzc5MTEsMjIuNjU4NDY4MjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDNlMjhmYWI1OGZmNGE3MGI2MjMyMGM2NzMxNjAwODQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmQ3NGUxNjM1NzZjNDIyZmFkNDA0ZDQ0NGY0MDRhNTEgPSAkKCc8ZGl2IGlkPSJodG1sX2ZkNzRlMTYzNTc2YzQyMmZhZDQwNGQ0NDRmNDA0YTUxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Op86xzr3PhM6szrrOuc6xLCDOlM6XzpzOn86jIM6azp/Opc6kzqPOn86gzp/OlM6Zzp/OpTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDNlMjhmYWI1OGZmNGE3MGI2MjMyMGM2NzMxNjAwODQuc2V0Q29udGVudChodG1sX2ZkNzRlMTYzNTc2YzQyMmZhZDQwNGQ0NDRmNDA0YTUxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q1YTZlYTE5MjU0YjQyYWY5OGI5MGRjYWU5ZDAzOTJmLmJpbmRQb3B1cChwb3B1cF8wM2UyOGZhYjU4ZmY0YTcwYjYyMzIwYzY3MzE2MDA4NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83M2U5YTFjZDIyYzA0MDRkOTU5ODBlYjI3NTkyZmQ1YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQ4Njg3MzYzLDIyLjY2MTU5NDM5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMwNTMyMzQ2MDM4NTQwODhiYjdiM2MwNzhjYjQ2NzkyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q2NzAwODU3MzkyNTQ0OWE5MDllNTY3YmUyYmFiOGNlID0gJCgnPGRpdiBpZD0iaHRtbF9kNjcwMDg1NzM5MjU0NDlhOTA5ZTU2N2JlMmJhYjhjZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpLOtc67zrHOvc65zrTOuc6sLCDOlM6XzpzOn86jIM6bzpXOoc6dzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzA1MzIzNDYwMzg1NDA4OGJiN2IzYzA3OGNiNDY3OTIuc2V0Q29udGVudChodG1sX2Q2NzAwODU3MzkyNTQ0OWE5MDllNTY3YmUyYmFiOGNlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzczZTlhMWNkMjJjMDQwNGQ5NTk4MGViMjc1OTJmZDVjLmJpbmRQb3B1cChwb3B1cF8zMDUzMjM0NjAzODU0MDg4YmI3YjNjMDc4Y2I0Njc5Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xY2FiZmNjYmIwODI0NWM1OTY1MjA1ZjExYzhjZGVmMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3Ljc2MzMzNjE4LDIyLjQ3NzY2NDk1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JmMWUyM2NhZjY3NTRjNjhhYmVlYTU0YWM0ZjhhYTkwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YzNDYyM2Q3M2Y1YjRjMWNhYTA5YWI3MzFjNmMxZDgzID0gJCgnPGRpdiBpZD0iaHRtbF9mMzQ2MjNkNzNmNWI0YzFjYWEwOWFiNzMxYzZjMWQ4MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbOs865zr/PgiDOnc65zrrPjM67zrHOv8+CLCDOms6fzpnOnc6fzqTOl86kzpEgzpHOm86VzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmYxZTIzY2FmNjc1NGM2OGFiZWVhNTRhYzRmOGFhOTAuc2V0Q29udGVudChodG1sX2YzNDYyM2Q3M2Y1YjRjMWNhYTA5YWI3MzFjNmMxZDgzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFjYWJmY2NiYjA4MjQ1YzU5NjUyMDVmMTFjOGNkZWYyLmJpbmRQb3B1cChwb3B1cF9iZjFlMjNjYWY2NzU0YzY4YWJlZWE1NGFjNGY4YWE5MCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jODAwN2I1YzkyYzA0ZDdlOWMyMTVjMWYzZGU2MTlkNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3Ljc0MjU3NjYsMjIuNDc0NDg3MzAwMDAwMDAzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzAzZWE3Yzc2M2ViNzQ1OGFiYzgzYmY2OWMwZmQ5YTAxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkzYzZkM2U4NGNjZDRlNGRiODJkM2JiYzQ2MmJhNDllID0gJCgnPGRpdiBpZD0iaHRtbF85M2M2ZDNlODRjY2Q0ZTRkYjgyZDNiYmM0NjJiYTQ5ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpXOvs6/z4fOriwgzprOn86Zzp3On86kzpfOpM6RIM6RzpvOlc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzAzZWE3Yzc2M2ViNzQ1OGFiYzgzYmY2OWMwZmQ5YTAxLnNldENvbnRlbnQoaHRtbF85M2M2ZDNlODRjY2Q0ZTRkYjgyZDNiYmM0NjJiYTQ5ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jODAwN2I1YzkyYzA0ZDdlOWMyMTVjMWYzZGU2MTlkNy5iaW5kUG9wdXAocG9wdXBfMDNlYTdjNzYzZWI3NDU4YWJjODNiZjY5YzBmZDlhMDEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGY5MWI2MjQ2MDM5NDJlYmFmNDI2ZmU3NDc3MzgwMzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy43MzI1MTcyNCwyMi41Mzg4Mjk4MDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGE4M2M5YWYyZjU4NDA1NWEzYzZlZGM5ZmY5NWJlYTkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjM2ODNiNGI1NjNiNGMzMzgyM2ViZTBlYTRlZTAwMTkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTUxOGZmZWEzZWYwNDVjZDkxZWE0M2Q3NzkwZmEwODQgPSAkKCc8ZGl2IGlkPSJodG1sXzU1MThmZmVhM2VmMDQ1Y2Q5MWVhNDNkNzc5MGZhMDg0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OpM+DzrnPgc6vz4PPhM+BzrEsIM6UzpfOnM6fzqMgzpvOpc6hzprOlc6ZzpHOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjM2ODNiNGI1NjNiNGMzMzgyM2ViZTBlYTRlZTAwMTkuc2V0Q29udGVudChodG1sXzU1MThmZmVhM2VmMDQ1Y2Q5MWVhNDNkNzc5MGZhMDg0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhmOTFiNjI0NjAzOTQyZWJhZjQyNmZlNzQ3NzM4MDMzLmJpbmRQb3B1cChwb3B1cF9iMzY4M2I0YjU2M2I0YzMzODIzZWJlMGVhNGVlMDAxOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80YmM3ZGMxMDViMzI0Njc1YjFmNThmOWU5OTUzZjhhNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjcyMzA1Njc5LDIyLjUwMjEyMDk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzFiOTViYjVkMGNmYzQ3MDJhNDhkOGZhNzIwMTViMjk4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdjYWZjYThlMjMxZDRlYzY4ODRmZWIyZGUzNWJmMjJjID0gJCgnPGRpdiBpZD0iaHRtbF83Y2FmY2E4ZTIzMWQ0ZWM2ODg0ZmViMmRlMzViZjIyYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpTOv8+NzrrOsSDOks+Bz43Pg863LCDOlM6XzpzOn86jIM6bzqXOoc6azpXOmc6RzqM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFiOTViYjVkMGNmYzQ3MDJhNDhkOGZhNzIwMTViMjk4LnNldENvbnRlbnQoaHRtbF83Y2FmY2E4ZTIzMWQ0ZWM2ODg0ZmViMmRlMzViZjIyYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80YmM3ZGMxMDViMzI0Njc1YjFmNThmOWU5OTUzZjhhNS5iaW5kUG9wdXAocG9wdXBfMWI5NWJiNWQwY2ZjNDcwMmE0OGQ4ZmE3MjAxNWIyOTgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjJkNjVjM2JkMmM2NDlkMTkyOWYxMDFiMGQwM2ZkMGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40OTIzNzQ0MiwyMi45MTg3MzE2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80YTgzYzlhZjJmNTg0MDU1YTNjNmVkYzlmZjk1YmVhOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zYThjYjNlNmVhYjI0MjM2OGI0ZTM3MTA0YzA4YjczMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NmVmYTQ2YjQ2NGY0Mjc5OWVlZThiNjNkNDk3NjBiYSA9ICQoJzxkaXYgaWQ9Imh0bWxfNjZlZmE0NmI0NjRmNDI3OTllZWU4YjYzZDQ5NzYwYmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gzrvOsc+EzrXOuc6sLCDOlM6XzpzOn86jIM6RzqPOmc6dzpfOozwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2E4Y2IzZTZlYWIyNDIzNjhiNGUzNzEwNGMwOGI3MzAuc2V0Q29udGVudChodG1sXzY2ZWZhNDZiNDY0ZjQyNzk5ZWVlOGI2M2Q0OTc2MGJhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIyZDY1YzNiZDJjNjQ5ZDE5MjlmMTAxYjBkMDNmZDBiLmJpbmRQb3B1cChwb3B1cF8zYThjYjNlNmVhYjI0MjM2OGI0ZTM3MTA0YzA4YjczMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNTYwOGU2YTEwMGU0MGM0OTQzNDBlYzM2YzhhMWY5NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjI5OTE3MTQ1LDIzLjEzNzYyNDc0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRhODNjOWFmMmY1ODQwNTVhM2M2ZWRjOWZmOTViZWE5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q0NmI5YWM0ZDJjNDQ4NmU4MGViY2E0YzYxYTVkYzgxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNlOWE2MzI0MmU5MDRmOGZhNTQ1MGFkMzM4N2Q5YjBhID0gJCgnPGRpdiBpZD0iaHRtbF8zZTlhNjMyNDJlOTA0ZjhmYTU0NTBhZDMzODdkOWIwYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqfOt869zq/PhM+DzrEsIM6UzpfOnM6fzqMgzprOoc6Rzp3Omc6UzpnOn86lPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNDZiOWFjNGQyYzQ0ODZlODBlYmNhNGM2MWE1ZGM4MS5zZXRDb250ZW50KGh0bWxfM2U5YTYzMjQyZTkwNGY4ZmE1NDUwYWQzMzg3ZDliMGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjU2MDhlNmExMDBlNDBjNDk0MzQwZWMzNmM4YTFmOTQuYmluZFBvcHVwKHBvcHVwX2Q0NmI5YWM0ZDJjNDQ4NmU4MGViY2E0YzYxYTVkYzgxKTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4=" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



##### After that, the Foursquare API is going to be utilized in order to explore what "happens" inside the estates and then segment them accordingly.


```python
CLIENT_ID = '1FYZCKDYVNSH2YMVLILPYVRFMGBV42DOCKXBQ5WURQIWNRVX' # Foursquare ID
CLIENT_SECRET = 'HHZ3POK3ZVVJM55G1LCK2SOMWG4YSKH1EJWMTVGO2WWXTEJL' # Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('My credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
```

    My credentails:
    CLIENT_ID: 1FYZCKDYVNSH2YMVLILPYVRFMGBV42DOCKXBQ5WURQIWNRVX
    CLIENT_SECRET:HHZ3POK3ZVVJM55G1LCK2SOMWG4YSKH1EJWMTVGO2WWXTEJL


##### Now we are going to create a function to get the top X venues that exist in every estate in Argolis, Greece within a radius of Y meters.


```python
def getNearbyVenues(names, latitudes, longitudes, venues_number, radius):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            venues_number)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Estate', 
                  'Estate Latitude', 
                  'Estate Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```

##### Now let's use it.


```python
argolis_venues = getNearbyVenues(names=df_argolis['Estate'],
                                 latitudes=df_argolis['Latitude'],
                                 longitudes=df_argolis['Longitude'],
                                 venues_number=100,
                                 radius=1000
                                  )
```

    Πορτοχέλιον
    Κρανίδιον
    Ερμιόνη
    Θερμησία
    Κοιλάς
    Φούρνοι
    Ηλιόκαστρον
    Δίδυμα
    Ανδρίτσα
    Ίρια
    Καρνεζαίικα
    Κορωνήσι
    Κιβέριον
    Αχλαδόκαμπος
    Δρέπανον
    Ασίνη
    Λευκάκια
    Τραχειά
    Σκαφιδάκιον
    Ναύπλιον
    ¶ρια
    Πυργιώτικα
    Κρυονέριον
    Νέα Κίος
    Αρκαδικόν
    Κεφαλάριον
    ¶γιος Αδριανός
    Μετόχιον
    Νέα Τίρυνς
    Λυγούριον
    Φρέγκαινα
    Νέον Ροεινόν
    Δαλαμανάρα
    Λάλουκας
    Πυργέλλα
    Αρχαία Επίδαυρος
    ¶ργος
    Αγία Τριάδα
    Κουρτάκιον
    Καρυά
    Ηραίον
    Ήρα
    Ανύφιον
    Βρούστιον
    Ίναχος
    Νεοχώριον
    Νέον Ηραίον
    Νέα Επίδαυρος
    Αραχναίον
    Καπαρέλλιον
    Σχινοχώριον
    Κουτσοπόδιον
    Δήμαινα
    Κεφαλόβρυσον
    Μοναστηράκιον
    Λύρκεια
    Πρόσυμνα
    Λίμναι
    Φρουσιούνα
    Μυκήναι
    Στέρνα
    Φίχτιον
    Μαλαντρένιον
    Μπόρσας
    Αλέα
    Γυμνόν
    Σκοτεινή
    Πλατάνιον
    Αμαριανός
    Τρίστρατον
    Μετόχιον
    Μάνεσης
    Πουλλακίδα
    Αμυγδαλίτσα
    Μιδέα
    Γκάτζια
    Καλλιθέα
    Νέα Μαραθέα
    Μαραθέα
    Βιβάριον
    Αγία Παρασκευή
    Παραλία Ασίνης
    Ασπρόβρυση
    Παληοχώρα
    Προφήτης Ηλίας
    Παναγία
    Μονή Καρακαλά
    Μονή Αγίου Θεοδοσίου του Νέου
    Παναρίτης
    Αργολικόν
    Τίρυνς
    Καποδίστριας
    Τολόν
    Δασκαλειό
    ¶γιος Αντώνιος
    Σταματαίικα
    Χουταλαίικα
    Γιαννουλαίικα
    Κοκκινάδες
    Σπηλεία
    Χάνι Μερκούρη
    Ασκληπιείο Επιδαύρου
    ¶γιος Ανδρέας
    Πανόραμα
    Επάνω Επίδαυρος
    Παναγία
    Γαλαναίικα
    Μονή Ταξιαρχών
    Νέα Δήμαινα
    Εξοχή
    Κολιάκιον
    Σταυρός
    Βοθίκιον
    Ματαράγκα
    ¶νω Καρνεζαίικα
    Καναπίτσα
    Σταυροπόδιον
    Αδάμιον
    Δημοσιά
    ¶γιος Νικόλαος
    Κατσιγιανναίικα
    Κάντια
    Ψηλή
    Ακτή Ύδρας
    Αγία Αικατερίνη
    Πλέπιον
    Πηγάδια
    Σωληνάριον
    Μετόχιον
    Αχλαδίτσα
    ¶γιοι Ανάργυροι
    Πετροθάλασσα
    Κουνούπι
    ¶γιος Νικόλαος
    Θυνί
    Λάκκες
    Δορούφι
    Κάμπος
    Βλαχοπουλέικα
    Κάμπος
    Δορούφι
    Κορωνίς
    Παραλία Φούρνων
    Βερβερούδα
    Χινίτσα
    Κόστα
    ¶γιος Αιμιλιανός
    Λουκαΐτιον
    Ράδον
    Πελεή
    Σαλάντιον
    ¶γιος Ιωάννης
    Μπούρτζιον
    Τημένιον
    Κόκλα
    ¶κοβα
    Αγία Αικατερίνη
    Αεροδρόμιο
    Πανόραμα
    Σύνορο
    Αρία
    Γαλαναίικα
    Στραβή Ράχη
    Γαλάτιον
    Αγριλίτσα
    Σπαναίικα
    Χούνη
    Μποζιονελαίικα
    ¶γιος Γεώργιος
    Μερκούριον
    ¶γιος Στέφανος
    Τουρνίκιον
    Κρύα Βρύση
    Ζόγκα
    Ελληνικόν
    Σπηλιωτάκης
    Κουγαίικα
    Μύλοι
    Καλαμάκιον
    Διχάλια
    Αλμυρός
    Μαγούλα
    Χέλμης
    Σταθαίικα
    Χαντάκια
    Βελανιδιά
    ¶γιος Νικόλαος
    Εξοχή
    Τσιρίστρα
    Δούκα Βρύση
    Πλατειά
    Χηνίτσα



```python
argolis_venues.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Estate Latitude</th>
      <th>Estate Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Πορτοχέλιον</td>
      <td>37.32468</td>
      <td>23.140156</td>
      <td>Solo Gelato</td>
      <td>37.324733</td>
      <td>23.143357</td>
      <td>Ice Cream Shop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Πορτοχέλιον</td>
      <td>37.32468</td>
      <td>23.140156</td>
      <td>The Drunken Clam</td>
      <td>37.323660</td>
      <td>23.143820</td>
      <td>Seafood Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Πορτοχέλιον</td>
      <td>37.32468</td>
      <td>23.140156</td>
      <td>Drougas Bakery</td>
      <td>37.324806</td>
      <td>23.143294</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Πορτοχέλιον</td>
      <td>37.32468</td>
      <td>23.140156</td>
      <td>veranda del vino</td>
      <td>37.323142</td>
      <td>23.143885</td>
      <td>Wine Bar</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Πορτοχέλιον</td>
      <td>37.32468</td>
      <td>23.140156</td>
      <td>To Go Cafe</td>
      <td>37.324371</td>
      <td>23.143678</td>
      <td>Café</td>
    </tr>
  </tbody>
</table>
</div>



##### How many venues were returned for each estate?


```python
argolis_venues.groupby('Estate').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate Latitude</th>
      <th>Estate Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
    <tr>
      <th>Estate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>¶γιοι Ανάργυροι</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>¶γιος Αδριανός</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>¶γιος Αιμιλιανός</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>¶γιος Ανδρέας</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>¶γιος Ιωάννης</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Φρουσιούνα</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Χάνι Μερκούρη</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Χηνίτσα</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Χινίτσα</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Χουταλαίικα</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>117 rows × 6 columns</p>
</div>




```python
print('There are {} uniques venue categories.'.format(len(argolis_venues['Venue Category'].unique())))
```

    There are 111 uniques venue categories.


##### It is time to analyze each neighborhood. Let's create dummy variables for the analysis to be more interpretable.


```python
# one hot encoding
argolis_onehot = pd.get_dummies(argolis_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
argolis_onehot['Estate'] = argolis_venues['Estate'] 

# move neighborhood column to the first column
fixed_columns = [argolis_onehot.columns[-1]] + list(argolis_onehot.columns[:-1])
argolis_onehot = argolis_onehot[fixed_columns]

print("The new dataframe's shape: ", argolis_onehot.shape)
print('')
argolis_onehot.head()
```

    The new dataframe's shape:  (786, 112)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Antique Shop</th>
      <th>Art Gallery</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Workshop</th>
      <th>BBQ Joint</th>
      <th>Bakery</th>
      <th>Bar</th>
      <th>Basketball Court</th>
      <th>Basketball Stadium</th>
      <th>...</th>
      <th>Taverna</th>
      <th>Tea Room</th>
      <th>Theater</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Tunnel</th>
      <th>Vacation Rental</th>
      <th>Warehouse Store</th>
      <th>Waterfront</th>
      <th>Wine Bar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Πορτοχέλιον</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Πορτοχέλιον</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Πορτοχέλιον</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Πορτοχέλιον</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Πορτοχέλιον</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 112 columns</p>
</div>



##### Let's group rows by estate and by taking the mean of the frequency of occurrence of each venue category.


```python
argolis_grouped = argolis_onehot.groupby('Estate').mean().reset_index()
argolis_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Antique Shop</th>
      <th>Art Gallery</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Workshop</th>
      <th>BBQ Joint</th>
      <th>Bakery</th>
      <th>Bar</th>
      <th>Basketball Court</th>
      <th>Basketball Stadium</th>
      <th>...</th>
      <th>Taverna</th>
      <th>Tea Room</th>
      <th>Theater</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Tunnel</th>
      <th>Vacation Rental</th>
      <th>Warehouse Store</th>
      <th>Waterfront</th>
      <th>Wine Bar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>¶γιοι Ανάργυροι</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>¶γιος Αδριανός</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>¶γιος Αιμιλιανός</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>¶γιος Ανδρέας</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>¶γιος Ιωάννης</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 112 columns</p>
</div>



##### Let's print each estate along with the top 5 most common venues.


```python
num_top_venues = 5

for hood in argolis_grouped['Estate']:
    print("----"+hood+"----")
    temp = argolis_grouped[argolis_grouped['Estate'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
```

    ----¶γιοι Ανάργυροι----
              venue  freq
    0         Beach  0.50
    1          Pool  0.25
    2         Hotel  0.25
    3  Antique Shop  0.00
    4        Museum  0.00
    
    
    ----¶γιος Αδριανός----
                  venue  freq
    0  Greek Restaurant   1.0
    1      Antique Shop   0.0
    2            Museum   0.0
    3              Pool   0.0
    4             Plaza   0.0
    
    
    ----¶γιος Αιμιλιανός----
                  venue  freq
    0             Beach   0.6
    1  Greek Restaurant   0.2
    2             Hotel   0.2
    3            Museum   0.0
    4              Pool   0.0
    
    
    ----¶γιος Ανδρέας----
                 venue  freq
    0  Nature Preserve   1.0
    1     Antique Shop   0.0
    2           Museum   0.0
    3             Pool   0.0
    4            Plaza   0.0
    
    
    ----¶γιος Ιωάννης----
              venue  freq
    0         Beach   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----¶γιος Νικόλαος----
              venue  freq
    0         Beach   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----¶ργος----
                    venue  freq
    0                Café  0.24
    1  Italian Restaurant  0.10
    2           Nightclub  0.10
    3   Mobile Phone Shop  0.05
    4        Dessert Shop  0.05
    
    
    ----¶ρια----
                  venue  freq
    0       Coffee Shop   0.2
    1       Event Space   0.2
    2         Cafeteria   0.2
    3       Supermarket   0.2
    4  Greek Restaurant   0.2
    
    
    ----Ήρα----
            venue  freq
    0     Kafenio  0.25
    1       Plaza  0.25
    2  Shoe Store  0.25
    3  Food Truck  0.25
    4        Port  0.00
    
    
    ----Ίναχος----
           venue  freq
    0    Kafenio   0.5
    1    Brewery   0.5
    2  Racetrack   0.0
    3       Pool   0.0
    4      Plaza   0.0
    
    
    ----Ίρια----
                  venue  freq
    0  Greek Restaurant  0.25
    1              Farm  0.25
    2              Café  0.25
    3           Taverna  0.25
    4            Museum  0.00
    
    
    ----Αγία Αικατερίνη----
                  venue  freq
    0            Resort  0.25
    1           Theater  0.12
    2  Business Service  0.12
    3               Spa  0.12
    4           Taverna  0.12
    
    
    ----Αγία Παρασκευή----
                         venue  freq
    0         Greek Restaurant   0.2
    1                   Bakery   0.2
    2                   Ouzeri   0.2
    3        French Restaurant   0.2
    4  Grilled Meat Restaurant   0.2
    
    
    ----Αγία Τριάδα----
                    venue  freq
    0                 Bar  0.33
    1  Basketball Stadium  0.17
    2        Liquor Store  0.17
    3    Basketball Court  0.17
    4                Café  0.17
    
    
    ----Αγριλίτσα----
              venue  freq
    0          Farm   1.0
    1  Antique Shop   0.0
    2     Juice Bar   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Αδάμιον----
                   venue  freq
    0  German Restaurant  0.33
    1             Bakery  0.33
    2            Taverna  0.33
    3       Antique Shop  0.00
    4               Pool  0.00
    
    
    ----Ακτή Ύδρας----
                    venue  freq
    0  Italian Restaurant  0.17
    1                 Spa  0.17
    2              Resort  0.17
    3        Fish Taverna  0.17
    4             Theater  0.17
    
    
    ----Αλμυρός----
                    venue  freq
    0    Greek Restaurant  0.29
    1  Seafood Restaurant  0.14
    2     Vacation Rental  0.14
    3     Bed & Breakfast  0.14
    4       Souvlaki Shop  0.14
    
    
    ----Αμυγδαλίτσα----
              venue  freq
    0  Neighborhood   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Ανύφιον----
              venue  freq
    0           Bar   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Αραχναίον----
              venue  freq
    0       Taverna   0.5
    1  Soccer Field   0.5
    2  Antique Shop   0.0
    3        Museum   0.0
    4         Plaza   0.0
    
    
    ----Αργολικόν----
                  venue  freq
    0  Greek Restaurant   1.0
    1      Antique Shop   0.0
    2            Museum   0.0
    3              Pool   0.0
    4             Plaza   0.0
    
    
    ----Αρχαία Επίδαυρος----
                    venue  freq
    0    Greek Restaurant  0.21
    1                Café  0.15
    2               Beach  0.09
    3          Restaurant  0.06
    4  Seafood Restaurant  0.06
    
    
    ----Ασίνη----
                  venue  freq
    0  Greek Restaurant  0.25
    1        Hotel Pool  0.25
    2            Bakery  0.25
    3      Soccer Field  0.25
    4            Museum  0.00
    
    
    ----Ασκληπιείο Επιδαύρου----
                     venue  freq
    0          Coffee Shop  0.25
    1        Historic Site  0.25
    2              Theater  0.25
    3  Monument / Landmark  0.25
    4         Antique Shop  0.00
    
    
    ----Ασπρόβρυση----
              venue  freq
    0         Hotel   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Αχλαδίτσα----
              venue  freq
    0         Beach  0.75
    1    Playground  0.25
    2  Antique Shop  0.00
    3        Museum  0.00
    4          Pool  0.00
    
    
    ----Αχλαδόκαμπος----
              venue  freq
    0   Coffee Shop  0.25
    1        Bakery  0.25
    2  Soccer Field  0.25
    3      Mountain  0.25
    4  Antique Shop  0.00
    
    
    ----Βερβερούδα----
               venue  freq
    0         Resort   0.4
    1  Movie Theater   0.2
    2     Restaurant   0.2
    3          Beach   0.2
    4         Museum   0.0
    
    
    ----Βιβάριον----
                    venue  freq
    0        Fish Taverna  0.29
    1    Greek Restaurant  0.29
    2          Restaurant  0.14
    3  Seafood Restaurant  0.14
    4     Harbor / Marina  0.14
    
    
    ----Βλαχοπουλέικα----
                  venue  freq
    0    Ice Cream Shop   0.5
    1  Greek Restaurant   0.5
    2      Antique Shop   0.0
    3            Museum   0.0
    4              Pool   0.0
    
    
    ----Γιαννουλαίικα----
              venue  freq
    0           Bar   0.5
    1       Taverna   0.5
    2  Antique Shop   0.0
    3        Museum   0.0
    4          Pool   0.0
    
    
    ----Δήμαινα----
              venue  freq
    0           Bar   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Δίδυμα----
                  venue  freq
    0  Greek Restaurant  0.33
    1              Cave  0.17
    2               Bar  0.17
    3          Mountain  0.17
    4    Soccer Stadium  0.17
    
    
    ----Δαλαμανάρα----
                   venue  freq
    0    Warehouse Store   0.2
    1       Liquor Store   0.2
    2             Bakery   0.2
    3        Event Space   0.2
    4  Food & Drink Shop   0.2
    
    
    ----Δημοσιά----
                         venue  freq
    0  Grilled Meat Restaurant   1.0
    1             Antique Shop   0.0
    2                   Museum   0.0
    3                     Pool   0.0
    4                    Plaza   0.0
    
    
    ----Διχάλια----
                  venue  freq
    0  Basketball Court   1.0
    1      Antique Shop   0.0
    2            Museum   0.0
    3              Pool   0.0
    4             Plaza   0.0
    
    
    ----Δορούφι----
              venue  freq
    0         Beach   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Δρέπανον----
                  venue  freq
    0  Greek Restaurant  0.36
    1             Beach  0.18
    2       Pizza Place  0.09
    3           Taverna  0.09
    4             Hotel  0.09
    
    
    ----Εξοχή----
                   venue  freq
    0  Convenience Store  0.33
    1      Boat or Ferry  0.33
    2             Bakery  0.33
    3               Port  0.00
    4               Pool  0.00
    
    
    ----Επάνω Επίδαυρος----
                 venue  freq
    0  Vacation Rental   0.5
    1            Motel   0.5
    2     Antique Shop   0.0
    3           Museum   0.0
    4             Pool   0.0
    
    
    ----Ερμιόνη----
                  venue  freq
    0              Café  0.25
    1  Greek Restaurant  0.15
    2            Bakery  0.10
    3       Pizza Place  0.10
    4           Taverna  0.10
    
    
    ----Ηλιόκαστρον----
               venue  freq
    0     Steakhouse   0.5
    1      BBQ Joint   0.5
    2  Movie Theater   0.0
    3           Pool   0.0
    4          Plaza   0.0
    
    
    ----Ηραίον----
               venue  freq
    0   Liquor Store   0.5
    1  Grocery Store   0.5
    2   Antique Shop   0.0
    3         Museum   0.0
    4           Pool   0.0
    
    
    ----Θερμησία----
                  venue  freq
    0      Cocktail Bar   0.2
    1  Sculpture Garden   0.2
    2      Fish Taverna   0.2
    3             Beach   0.2
    4  Greek Restaurant   0.2
    
    
    ----Θυνί----
              venue  freq
    0         Beach   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Κάμπος----
                  venue  freq
    0  Greek Restaurant  0.25
    1             Hotel  0.25
    2     Auto Workshop  0.12
    3    Ice Cream Shop  0.12
    4              Café  0.12
    
    
    ----Κάντια----
                  venue  freq
    0  Greek Restaurant  0.33
    1        Restaurant  0.17
    2         BBQ Joint  0.17
    3           Taverna  0.17
    4             Beach  0.17
    
    
    ----Καλαμάκιον----
                    venue  freq
    0    Greek Restaurant  0.25
    1  Athletics & Sports  0.25
    2             Taverna  0.25
    3           Racetrack  0.25
    4              Museum  0.00
    
    
    ----Καλλιθέα----
                  venue  freq
    0             Beach  0.33
    1  Greek Restaurant  0.17
    2             Hotel  0.17
    3        Campground  0.17
    4              Café  0.17
    
    
    ----Καποδίστριας----
                    venue  freq
    0  Italian Restaurant   0.2
    1               Hotel   0.2
    2          Donut Shop   0.2
    3      Sandwich Place   0.2
    4                Pool   0.2
    
    
    ----Καρυά----
                  venue  freq
    0           Taverna   0.5
    1  Greek Restaurant   0.5
    2      Antique Shop   0.0
    3     Movie Theater   0.0
    4             Plaza   0.0
    
    
    ----Κεφαλάριον----
                  venue  freq
    0  Greek Restaurant   0.6
    1           Taverna   0.4
    2      Antique Shop   0.0
    3     Movie Theater   0.0
    4             Plaza   0.0
    
    
    ----Κιβέριον----
                    venue  freq
    0           Beach Bar   0.2
    1          Waterfront   0.2
    2              Bakery   0.2
    3  Seafood Restaurant   0.2
    4               Beach   0.2
    
    
    ----Κοιλάς----
                    venue  freq
    0    Greek Restaurant  0.27
    1                Café  0.18
    2     Harbor / Marina  0.18
    3  Seafood Restaurant  0.18
    4                 Bar  0.09
    
    
    ----Κολιάκιον----
                  venue  freq
    0  Greek Restaurant  0.25
    1            Bakery  0.25
    2             Beach  0.25
    3              Café  0.25
    4   Nature Preserve  0.00
    
    
    ----Κορωνήσι----
                  venue  freq
    0             Hotel  0.18
    1  Greek Restaurant  0.15
    2              Café  0.08
    3               Bar  0.08
    4           Taverna  0.08
    
    
    ----Κορωνίς----
                 venue  freq
    0              Bar  0.33
    1            Beach  0.33
    2             Café  0.33
    3     Antique Shop  0.00
    4  Nature Preserve  0.00
    
    
    ----Κουγαίικα----
              venue  freq
    0  Soccer Field   1.0
    1  Antique Shop   0.0
    2     Racetrack   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Κουνούπι----
              venue  freq
    0         Beach   0.8
    1        Resort   0.2
    2  Antique Shop   0.0
    3        Museum   0.0
    4          Pool   0.0
    
    
    ----Κουτσοπόδιον----
              venue  freq
    0  Soccer Field  0.14
    1     BBQ Joint  0.14
    2   Supermarket  0.14
    3          Café  0.14
    4  Betting Shop  0.14
    
    
    ----Κρανίδιον----
              venue  freq
    0         Plaza  0.15
    1        Bakery  0.15
    2  Soccer Field  0.08
    3   Supermarket  0.08
    4          Café  0.08
    
    
    ----Κρύα Βρύση----
              venue  freq
    0      Mountain   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Κόστα----
                    venue  freq
    0       Boat or Ferry  0.43
    1               Beach  0.29
    2                Port  0.14
    3  Seafood Restaurant  0.14
    4                Pool  0.00
    
    
    ----Λάκκες----
              venue  freq
    0         Beach   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Λίμναι----
              venue  freq
    0          Café   1.0
    1  Antique Shop   0.0
    2     Racetrack   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Λευκάκια----
                   venue  freq
    0        Coffee Shop   0.2
    1                Bar   0.2
    2              Hotel   0.2
    3       Soccer Field   0.2
    4  French Restaurant   0.2
    
    
    ----Λουκαΐτιον----
              venue  freq
    0          Farm   0.5
    1         Trail   0.5
    2  Antique Shop   0.0
    3        Museum   0.0
    4         Plaza   0.0
    
    
    ----Λυγούριον----
                  venue  freq
    0           Taverna  0.50
    1  Greek Restaurant  0.17
    2               Bar  0.17
    3              Café  0.17
    4            Museum  0.00
    
    
    ----Λύρκεια----
         venue  freq
    0  Kafenio   0.5
    1   Tunnel   0.5
    2   Museum   0.0
    3     Pool   0.0
    4    Plaza   0.0
    
    
    ----Μάνεσης----
              venue  freq
    0          Café   0.5
    1    Steakhouse   0.5
    2  Antique Shop   0.0
    3        Museum   0.0
    4          Pool   0.0
    
    
    ----Μαγούλα----
                  venue  freq
    0  Greek Restaurant   1.0
    1      Antique Shop   0.0
    2            Museum   0.0
    3              Pool   0.0
    4             Plaza   0.0
    
    
    ----Μαλαντρένιον----
              venue  freq
    0        Bakery   0.5
    1  Soccer Field   0.5
    2  Antique Shop   0.0
    3        Museum   0.0
    4          Pool   0.0
    
    
    ----Μετόχιον----
                 venue  freq
    0     Fish Taverna  0.33
    1      Flower Shop  0.33
    2    Big Box Store  0.33
    3     Antique Shop  0.00
    4  Nature Preserve  0.00
    
    
    ----Μοναστηράκιον----
                  venue  freq
    0       Art Gallery   0.5
    1  Greek Restaurant   0.5
    2            Museum   0.0
    3              Pool   0.0
    4             Plaza   0.0
    
    
    ----Μπούρτζιον----
                venue  freq
    0           Hotel  0.20
    1            Café  0.16
    2  Ice Cream Shop  0.06
    3         Taverna  0.06
    4           Plaza  0.05
    
    
    ----Μυκήναι----
                  venue  freq
    0  Greek Restaurant  0.50
    1        Restaurant  0.25
    2           Taverna  0.25
    3      Antique Shop  0.00
    4            Museum  0.00
    
    
    ----Μύλοι----
                    venue  freq
    0    Greek Restaurant  0.44
    1       Souvlaki Shop  0.22
    2        Fish Taverna  0.11
    3  Seafood Restaurant  0.11
    4               Beach  0.11
    
    
    ----Νέα Επίδαυρος----
                   venue  freq
    0  Convenience Store  0.14
    1                Bar  0.14
    2              Plaza  0.14
    3         Campground  0.14
    4      Souvlaki Shop  0.14
    
    
    ----Νέα Κίος----
                         venue  freq
    0  Grilled Meat Restaurant  0.18
    1                     Café  0.18
    2         Basketball Court  0.09
    3                   Bistro  0.09
    4                    Beach  0.09
    
    
    ----Νέα Μαραθέα----
                  venue  freq
    0      Fish Taverna  0.33
    1  Greek Restaurant  0.33
    2        Restaurant  0.17
    3   Harbor / Marina  0.17
    4      Antique Shop  0.00
    
    
    ----Νέα Τίρυνς----
                   venue  freq
    0  Recreation Center  0.33
    1    Vacation Rental  0.33
    2               Café  0.33
    3             Museum  0.00
    4               Pool  0.00
    
    
    ----Νέον Ροεινόν----
                 venue  freq
    0           Resort   0.5
    1  Bed & Breakfast   0.5
    2     Antique Shop   0.0
    3           Museum   0.0
    4             Pool   0.0
    
    
    ----Ναύπλιον----
                venue  freq
    0           Hotel  0.17
    1            Café  0.15
    2         Taverna  0.05
    3             Bar  0.05
    4  Ice Cream Shop  0.05
    
    
    ----Παληοχώρα----
              venue  freq
    0       Taverna   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Παναγία----
                  venue  freq
    0        Campground  0.33
    1  Greek Restaurant  0.17
    2   Vacation Rental  0.17
    3             Hotel  0.17
    4             Beach  0.17
    
    
    ----Παναρίτης----
              venue  freq
    0         Plaza  0.25
    1         Hotel  0.25
    2          Farm  0.25
    3    Steakhouse  0.25
    4  Antique Shop  0.00
    
    
    ----Πανόραμα----
                 venue  freq
    0            Motel  0.25
    1  Vacation Rental  0.25
    2           Bakery  0.25
    3      Supermarket  0.25
    4    Movie Theater  0.00
    
    
    ----Παραλία Ασίνης----
            venue  freq
    0       Beach  0.29
    1       Hotel  0.14
    2         Bar  0.14
    3        Café  0.14
    4  Campground  0.14
    
    
    ----Παραλία Φούρνων----
              venue  freq
    0          Cave   1.0
    1  Antique Shop   0.0
    2     Racetrack   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Πετροθάλασσα----
              venue  freq
    0         Beach   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Πηγάδια----
                    venue  freq
    0  Seafood Restaurant  0.29
    1                Lake  0.14
    2          Playground  0.14
    3                Park  0.14
    4               Beach  0.14
    
    
    ----Πλέπιον----
               venue  freq
    0         Resort  0.50
    1  Boat or Ferry  0.25
    2          Beach  0.25
    3           Port  0.00
    4           Pool  0.00
    
    
    ----Πορτοχέλιον----
                  venue  freq
    0  Greek Restaurant  0.16
    1              Café  0.14
    2               Bar  0.11
    3            Bakery  0.08
    4    Ice Cream Shop  0.08
    
    
    ----Πουλλακίδα----
                 venue  freq
    0  Nature Preserve  0.25
    1             Farm  0.25
    2      Flower Shop  0.25
    3     Soccer Field  0.25
    4             Port  0.00
    
    
    ----Προφήτης Ηλίας----
              venue  freq
    0       Taverna   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Πρόσυμνα----
         venue  freq
    0  Stables  0.33
    1    Plaza  0.33
    2     Café  0.33
    3   Museum  0.00
    4     Pool  0.00
    
    
    ----Πυργέλλα----
                  venue  freq
    0  Greek Restaurant   1.0
    1      Antique Shop   0.0
    2            Museum   0.0
    3              Pool   0.0
    4             Plaza   0.0
    
    
    ----Πυργιώτικα----
                  venue  freq
    0           Taverna  0.50
    1             Hotel  0.25
    2  Greek Restaurant  0.25
    3      Antique Shop  0.00
    4            Museum  0.00
    
    
    ----Ράδον----
                venue  freq
    0  Scenic Lookout   1.0
    1    Antique Shop   0.0
    2       Racetrack   0.0
    3            Pool   0.0
    4           Plaza   0.0
    
    
    ----Σαλάντιον----
              venue  freq
    0         Beach   0.5
    1       Taverna   0.5
    2  Antique Shop   0.0
    3        Museum   0.0
    4          Pool   0.0
    
    
    ----Σκαφιδάκιον----
                  venue  freq
    0  Basketball Court   1.0
    1      Antique Shop   0.0
    2            Museum   0.0
    3              Pool   0.0
    4             Plaza   0.0
    
    
    ----Σκοτεινή----
              venue  freq
    0      Mountain   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Σπηλεία----
              venue  freq
    0          Farm   0.5
    1  Soccer Field   0.5
    2  Antique Shop   0.0
    3          Port   0.0
    4         Plaza   0.0
    
    
    ----Σπηλιωτάκης----
              venue  freq
    0     Racetrack   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Σωληνάριον----
                    venue  freq
    0  Seafood Restaurant  0.67
    1                Park  0.33
    2        Antique Shop  0.00
    3           Racetrack  0.00
    4                Pool  0.00
    
    
    ----Τίρυνς----
               venue  freq
    0         Museum  0.25
    1           Pool  0.25
    2          Hotel  0.25
    3  Historic Site  0.25
    4   Antique Shop  0.00
    
    
    ----Τημένιον----
                 venue  freq
    0  Vacation Rental  0.33
    1     Fish Taverna  0.33
    2        Nightclub  0.33
    3     Antique Shop  0.00
    4           Museum  0.00
    
    
    ----Τολόν----
                  venue  freq
    0             Hotel  0.23
    1  Greek Restaurant  0.19
    2           Taverna  0.06
    3              Café  0.06
    4               Bar  0.06
    
    
    ----Τραχειά----
                   venue  freq
    0  Convenience Store  0.25
    1             Bakery  0.25
    2              Beach  0.25
    3      Boat or Ferry  0.25
    4             Market  0.00
    
    
    ----Φίχτιον----
              venue  freq
    0  Antique Shop  0.25
    1        Tunnel  0.25
    2          Café  0.25
    3   Supermarket  0.25
    4        Museum  0.00
    
    
    ----Φούρνοι----
              venue  freq
    0          Park   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Φρουσιούνα----
              venue  freq
    0      Mountain   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Χάνι Μερκούρη----
              venue  freq
    0         Hotel   1.0
    1  Antique Shop   0.0
    2        Museum   0.0
    3          Pool   0.0
    4         Plaza   0.0
    
    
    ----Χηνίτσα----
                  venue  freq
    0  Greek Restaurant  0.12
    1           Taverna  0.12
    2       Snack Place  0.12
    3            Resort  0.12
    4        Restaurant  0.12
    
    
    ----Χινίτσα----
                  venue  freq
    0  Greek Restaurant  0.14
    1   Harbor / Marina  0.14
    2       Snack Place  0.14
    3            Resort  0.14
    4        Restaurant  0.14
    
    
    ----Χουταλαίικα----
              venue  freq
    0           Bar   0.5
    1       Taverna   0.5
    2  Antique Shop   0.0
    3        Museum   0.0
    4          Pool   0.0
    
    


##### Now let's put this into a dataframe in order to carry on. First we create a function.


```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```


```python
num_top_venues = 5

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Estate']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
estate_venues_sorted = pd.DataFrame(columns=columns)
estate_venues_sorted['Estate'] = argolis_grouped['Estate']

for ind in np.arange(argolis_grouped.shape[0]):
    estate_venues_sorted.iloc[ind, 1:] = return_most_common_venues(argolis_grouped.iloc[ind, :], num_top_venues)

print('The dimensions of the newly created dataframe: ', estate_venues_sorted.shape)
print('')
estate_venues_sorted.head()
```

    The dimensions of the newly created dataframe:  (117, 6)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>¶γιοι Ανάργυροι</td>
      <td>Beach</td>
      <td>Hotel</td>
      <td>Pool</td>
      <td>Wine Bar</td>
      <td>Food Truck</td>
    </tr>
    <tr>
      <th>1</th>
      <td>¶γιος Αδριανός</td>
      <td>Greek Restaurant</td>
      <td>Wine Bar</td>
      <td>French Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>¶γιος Αιμιλιανός</td>
      <td>Beach</td>
      <td>Hotel</td>
      <td>Greek Restaurant</td>
      <td>Wine Bar</td>
      <td>Food Truck</td>
    </tr>
    <tr>
      <th>3</th>
      <td>¶γιος Ανδρέας</td>
      <td>Nature Preserve</td>
      <td>Wine Bar</td>
      <td>Ice Cream Shop</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>¶γιος Ιωάννης</td>
      <td>Beach</td>
      <td>Wine Bar</td>
      <td>French Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
    </tr>
  </tbody>
</table>
</div>



##### The time has come to implement machine learning. We are going to use the k-means clustering algorithm to do that.


```python
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

argolis_grouped_clustering = argolis_grouped.drop('Estate', 1)

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(argolis_grouped_clustering)
    distortions.append(sum(np.min(cdist(argolis_grouped_clustering, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / argolis_grouped_clustering.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```


![png](output_45_0.png)


##### As we can see from the diagram above, the optimal number of clusters maybe is 6 (elbow point). So we are going to carry on with that.


```python
# set number of clusters
kclusters = 6

argolis_grouped_clustering = argolis_grouped.drop('Estate', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(argolis_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]
```




    array([2, 5, 2, 0, 2, 2, 0, 0, 0, 0], dtype=int32)



##### We are now going to create a new dataframe that includes the clusters as well as the top 5 venues for each estate.


```python
# add clustering labels
estate_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

argolis_merged = df_argolis

argolis_merged = argolis_merged.join(estate_venues_sorted.set_index('Estate'), on='Estate')

print('The shape of the merged dataframe: ', argolis_merged.shape)
argolis_merged.head()
```

    The shape of the merged dataframe:  (192, 11)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Borough</th>
      <th>County</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8422</th>
      <td>Πορτοχέλιον</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.324680</td>
      <td>23.140156</td>
      <td>0.0</td>
      <td>Greek Restaurant</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Bakery</td>
      <td>Ice Cream Shop</td>
    </tr>
    <tr>
      <th>8462</th>
      <td>Κρανίδιον</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.377983</td>
      <td>23.157290</td>
      <td>0.0</td>
      <td>Plaza</td>
      <td>Bakery</td>
      <td>Mobile Phone Shop</td>
      <td>Surf Spot</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>8467</th>
      <td>Ερμιόνη</td>
      <td>ΔΗΜΟΣ ΕΡΜΙΟΝΗΣ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.383213</td>
      <td>23.242247</td>
      <td>0.0</td>
      <td>Café</td>
      <td>Greek Restaurant</td>
      <td>Taverna</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>8484</th>
      <td>Θερμησία</td>
      <td>ΔΗΜΟΣ ΕΡΜΙΟΝΗΣ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.409878</td>
      <td>23.322416</td>
      <td>0.0</td>
      <td>Cocktail Bar</td>
      <td>Sculpture Garden</td>
      <td>Beach</td>
      <td>Greek Restaurant</td>
      <td>Fish Taverna</td>
    </tr>
    <tr>
      <th>8488</th>
      <td>Κοιλάς</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.412220</td>
      <td>23.123734</td>
      <td>0.0</td>
      <td>Greek Restaurant</td>
      <td>Café</td>
      <td>Seafood Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Beach</td>
    </tr>
  </tbody>
</table>
</div>



##### Not every estate appeared to house a venue; this resulted in some NaN values, which we are going to get rid of.


```python
missing_data = argolis_merged.isnull()

for col in missing_data.columns.values.tolist():
    print(col)
    print(missing_data[col].value_counts())
    print('')
```

    Estate
    False    192
    Name: Estate, dtype: int64
    
    Borough
    False    192
    Name: Borough, dtype: int64
    
    County
    False    192
    Name: County, dtype: int64
    
    Latitude
    False    192
    Name: Latitude, dtype: int64
    
    Longitude
    False    192
    Name: Longitude, dtype: int64
    
    Cluster Labels
    False    127
    True      65
    Name: Cluster Labels, dtype: int64
    
    1st Most Common Venue
    False    127
    True      65
    Name: 1st Most Common Venue, dtype: int64
    
    2nd Most Common Venue
    False    127
    True      65
    Name: 2nd Most Common Venue, dtype: int64
    
    3rd Most Common Venue
    False    127
    True      65
    Name: 3rd Most Common Venue, dtype: int64
    
    4th Most Common Venue
    False    127
    True      65
    Name: 4th Most Common Venue, dtype: int64
    
    5th Most Common Venue
    False    127
    True      65
    Name: 5th Most Common Venue, dtype: int64
    



```python
argolis_merged.dropna(axis = 0, inplace = True)
argolis_merged.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Borough</th>
      <th>County</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8422</th>
      <td>Πορτοχέλιον</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.324680</td>
      <td>23.140156</td>
      <td>0.0</td>
      <td>Greek Restaurant</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Bakery</td>
      <td>Ice Cream Shop</td>
    </tr>
    <tr>
      <th>8462</th>
      <td>Κρανίδιον</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.377983</td>
      <td>23.157290</td>
      <td>0.0</td>
      <td>Plaza</td>
      <td>Bakery</td>
      <td>Mobile Phone Shop</td>
      <td>Surf Spot</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>8467</th>
      <td>Ερμιόνη</td>
      <td>ΔΗΜΟΣ ΕΡΜΙΟΝΗΣ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.383213</td>
      <td>23.242247</td>
      <td>0.0</td>
      <td>Café</td>
      <td>Greek Restaurant</td>
      <td>Taverna</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>8484</th>
      <td>Θερμησία</td>
      <td>ΔΗΜΟΣ ΕΡΜΙΟΝΗΣ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.409878</td>
      <td>23.322416</td>
      <td>0.0</td>
      <td>Cocktail Bar</td>
      <td>Sculpture Garden</td>
      <td>Beach</td>
      <td>Greek Restaurant</td>
      <td>Fish Taverna</td>
    </tr>
    <tr>
      <th>8488</th>
      <td>Κοιλάς</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.412220</td>
      <td>23.123734</td>
      <td>0.0</td>
      <td>Greek Restaurant</td>
      <td>Café</td>
      <td>Seafood Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Beach</td>
    </tr>
  </tbody>
</table>
</div>



##### Edit the data type of the clusters in order for the visualization to not present any errors due to that particular reason.


```python
argolis_merged['Cluster Labels'] = argolis_merged['Cluster Labels'].astype(int)
argolis_merged.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Borough</th>
      <th>County</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8422</th>
      <td>Πορτοχέλιον</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.324680</td>
      <td>23.140156</td>
      <td>0</td>
      <td>Greek Restaurant</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Bakery</td>
      <td>Ice Cream Shop</td>
    </tr>
    <tr>
      <th>8462</th>
      <td>Κρανίδιον</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.377983</td>
      <td>23.157290</td>
      <td>0</td>
      <td>Plaza</td>
      <td>Bakery</td>
      <td>Mobile Phone Shop</td>
      <td>Surf Spot</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>8467</th>
      <td>Ερμιόνη</td>
      <td>ΔΗΜΟΣ ΕΡΜΙΟΝΗΣ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.383213</td>
      <td>23.242247</td>
      <td>0</td>
      <td>Café</td>
      <td>Greek Restaurant</td>
      <td>Taverna</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>8484</th>
      <td>Θερμησία</td>
      <td>ΔΗΜΟΣ ΕΡΜΙΟΝΗΣ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.409878</td>
      <td>23.322416</td>
      <td>0</td>
      <td>Cocktail Bar</td>
      <td>Sculpture Garden</td>
      <td>Beach</td>
      <td>Greek Restaurant</td>
      <td>Fish Taverna</td>
    </tr>
    <tr>
      <th>8488</th>
      <td>Κοιλάς</td>
      <td>ΔΗΜΟΣ ΚΡΑΝΙΔΙΟΥ</td>
      <td>ΝΟΜΟΣ ΑΡΓΟΛΙΔΟΣ</td>
      <td>37.412220</td>
      <td>23.123734</td>
      <td>0</td>
      <td>Greek Restaurant</td>
      <td>Café</td>
      <td>Seafood Restaurant</td>
      <td>Harbor / Marina</td>
      <td>Beach</td>
    </tr>
  </tbody>
</table>
</div>



## Results

##### Cluster visualization.


```python
# create map
map_clusters = folium.Map(location=[latitude, longitude], tiles="Stamen Toner", zoom_start=10)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(argolis_merged['Latitude'], argolis_merged['Longitude'], argolis_merged['Estate'], argolis_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0MyA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0MycsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbMzcuNTY4NjEzODUsMjIuODYwNTA1NDYwMzg1OV0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfMGI5MjNkM2IxZjkxNDY0NDkzYTcxNDJjNmM3ZWYwNWMgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3N0YW1lbi10aWxlcy17c30uYS5zc2wuZmFzdGx5Lm5ldC90b25lci97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MDc0YjgzOTdlZWE0YjhlYTk4ZWU1OWM0NWY1MDJlYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjMyNDY4MDMzLDIzLjE0MDE1NTc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2VkMzIzM2IyOTYxZTRhZjdhMGZjMDE3Yjk2ZThmZDI1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVlOTAzMGU0ZmIyYjQxM2NiMGRhOTg3ZjZkZmE5NTRkID0gJCgnPGRpdiBpZD0iaHRtbF81ZTkwMzBlNGZiMmI0MTNjYjBkYTk4N2Y2ZGZhOTU0ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDOv8+Bz4TOv8+Hzq3Ou865zr/OvSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VkMzIzM2IyOTYxZTRhZjdhMGZjMDE3Yjk2ZThmZDI1LnNldENvbnRlbnQoaHRtbF81ZTkwMzBlNGZiMmI0MTNjYjBkYTk4N2Y2ZGZhOTU0ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MDc0YjgzOTdlZWE0YjhlYTk4ZWU1OWM0NWY1MDJlYi5iaW5kUG9wdXAocG9wdXBfZWQzMjMzYjI5NjFlNGFmN2EwZmMwMTdiOTZlOGZkMjUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWRjNzU2YWNiYjEyNGIyYmE1NDQ3OGZmOTMwNjYzZjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4zNzc5ODMwOSwyMy4xNTcyODk1MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iYmQzN2QxOTIwMTQ0ZWU4YjUxYTkzMGI5MDdiMzk1ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85OGJkMzhlZGFlMDY0NWY3ODNkNDdkODk3ZWViZjY2YSA9ICQoJzxkaXYgaWQ9Imh0bWxfOThiZDM4ZWRhZTA2NDVmNzgzZDQ3ZDg5N2VlYmY2NmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6az4HOsc69zq/OtM65zr/OvSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JiZDM3ZDE5MjAxNDRlZThiNTFhOTMwYjkwN2IzOTVkLnNldENvbnRlbnQoaHRtbF85OGJkMzhlZGFlMDY0NWY3ODNkNDdkODk3ZWViZjY2YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xZGM3NTZhY2JiMTI0YjJiYTU0NDc4ZmY5MzA2NjNmNC5iaW5kUG9wdXAocG9wdXBfYmJkMzdkMTkyMDE0NGVlOGI1MWE5MzBiOTA3YjM5NWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWEzMDk5ODBhY2ExNDA1NzljMzM4YWUxMWI3OGZjYzggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4zODMyMTMwNCwyMy4yNDIyNDY2M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iNDY3MzA1NWE3NjQ0N2U5OGI5ODVkMzljZWJkZTI1YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kMGM0OTA4NmNlM2I0ODdmODdiNTFlMGNjMWQ3OTMyYSA9ICQoJzxkaXYgaWQ9Imh0bWxfZDBjNDkwODZjZTNiNDg3Zjg3YjUxZTBjYzFkNzkzMmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Vz4HOvM65z4zOvc63IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjQ2NzMwNTVhNzY0NDdlOThiOTg1ZDM5Y2ViZGUyNWIuc2V0Q29udGVudChodG1sX2QwYzQ5MDg2Y2UzYjQ4N2Y4N2I1MWUwY2MxZDc5MzJhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzlhMzA5OTgwYWNhMTQwNTc5YzMzOGFlMTFiNzhmY2M4LmJpbmRQb3B1cChwb3B1cF9iNDY3MzA1NWE3NjQ0N2U5OGI5ODVkMzljZWJkZTI1Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMGFiY2YwMmFlMTU0OTVjOTg4MTM5NjYwYzQyYzBlNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQwOTg3Nzc4LDIzLjMyMjQxNjMxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY1ZmY0OTAyODQxMTQwMzQ4ZDIzMDI2MjcwYmEzOWM3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EyN2E5YWIzMDU4MDQyYWU5OTkyZWI0N2RhMjg1ZTJhID0gJCgnPGRpdiBpZD0iaHRtbF9hMjdhOWFiMzA1ODA0MmFlOTk5MmViNDdkYTI4NWUyYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpjOtc+BzrzOt8+Dzq/OsSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY1ZmY0OTAyODQxMTQwMzQ4ZDIzMDI2MjcwYmEzOWM3LnNldENvbnRlbnQoaHRtbF9hMjdhOWFiMzA1ODA0MmFlOTk5MmViNDdkYTI4NWUyYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMGFiY2YwMmFlMTU0OTVjOTg4MTM5NjYwYzQyYzBlNS5iaW5kUG9wdXAocG9wdXBfNjVmZjQ5MDI4NDExNDAzNDhkMjMwMjYyNzBiYTM5YzcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGRlNGUwOTM2ZDI0NGE4N2I5OGYwMGVjNjk1NTNlOWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40MTIyMiwyMy4xMjM3MzM1Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iNDZmNTZkOTJhNmY0ZjliYmFiYmNlYmExMTI5OTg5MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MzM1MDFmOTA3OTk0YTA2OWE1Zjc3ZGIwY2Q1NjQwZSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDMzNTAxZjkwNzk5NGEwNjlhNWY3N2RiMGNkNTY0MGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6azr/Ouc67zqzPgiBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I0NmY1NmQ5MmE2ZjRmOWJiYWJiY2ViYTExMjk5ODkzLnNldENvbnRlbnQoaHRtbF80MzM1MDFmOTA3OTk0YTA2OWE1Zjc3ZGIwY2Q1NjQwZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80ZGU0ZTA5MzZkMjQ0YTg3Yjk4ZjAwZWM2OTU1M2U5ZC5iaW5kUG9wdXAocG9wdXBfYjQ2ZjU2ZDkyYTZmNGY5YmJhYmJjZWJhMTEyOTk4OTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmY3YzZiOTk4YmMyNGU2YTllMjllMWE0ODQ0MmZjMGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40MjgzMTQyMSwyMy4xNzYzNzQ0NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iOTM3MzBhYzgzODA0MDdmYTcyM2IyOTM4MGI1Yjc3YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jOTFlNmFjMDc3MzI0ZjNiYjIxMWQ4OGE1MDgxMDA2OCA9ICQoJzxkaXYgaWQ9Imh0bWxfYzkxZTZhYzA3NzMyNGYzYmIyMTFkODhhNTA4MTAwNjgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6mzr/Pjc+Bzr3Ov865IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjkzNzMwYWM4MzgwNDA3ZmE3MjNiMjkzODBiNWI3N2Muc2V0Q29udGVudChodG1sX2M5MWU2YWMwNzczMjRmM2JiMjExZDg4YTUwODEwMDY4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJmN2M2Yjk5OGJjMjRlNmE5ZTI5ZTFhNDg0NDJmYzBhLmJpbmRQb3B1cChwb3B1cF9iOTM3MzBhYzgzODA0MDdmYTcyM2IyOTM4MGI1Yjc3Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMjQ4ZTNmYTA4ODg0ODg0OTU4ZWExMGZlMDM2OTRmZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQ0MTc1MzM5LDIzLjI2NDgwMjkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QwNjFlZTgxM2JhNzQyYmM5YzQyY2NkNjA0NGYxNTQzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc1ZWUzMWZmMzBkNTQ5ZThiMWY4NjI2ZTQzODgwYTk3ID0gJCgnPGRpdiBpZD0iaHRtbF83NWVlMzFmZjMwZDU0OWU4YjFmODYyNmU0Mzg4MGE5NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpfOu865z4zOus6xz4PPhM+Bzr/OvSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QwNjFlZTgxM2JhNzQyYmM5YzQyY2NkNjA0NGYxNTQzLnNldENvbnRlbnQoaHRtbF83NWVlMzFmZjMwZDU0OWU4YjFmODYyNmU0Mzg4MGE5Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMjQ4ZTNmYTA4ODg0ODg0OTU4ZWExMGZlMDM2OTRmZS5iaW5kUG9wdXAocG9wdXBfZDA2MWVlODEzYmE3NDJiYzljNDJjY2Q2MDQ0ZjE1NDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzczMTZhM2Y3NWU1NGZjMjk4ZWIxYTE2ZGU4MTQwODUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40NjA5MjYwNiwyMy4xNzEwNzk2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81YWFjMzgzZGQwMDQ0ZjczYWUyYzAwNzc3YzI3ZDAwMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82OWFmY2FhYjUzMjA0NzMyYTQxNmQ0MWVlMTdjNDU2YiA9ICQoJzxkaXYgaWQ9Imh0bWxfNjlhZmNhYWI1MzIwNDczMmE0MTZkNDFlZTE3YzQ1NmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Uzq/OtM+FzrzOsSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVhYWMzODNkZDAwNDRmNzNhZTJjMDA3NzdjMjdkMDAzLnNldENvbnRlbnQoaHRtbF82OWFmY2FhYjUzMjA0NzMyYTQxNmQ0MWVlMTdjNDU2Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNzMxNmEzZjc1ZTU0ZmMyOThlYjFhMTZkZTgxNDA4NS5iaW5kUG9wdXAocG9wdXBfNWFhYzM4M2RkMDA0NGY3M2FlMmMwMDc3N2MyN2QwMDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzc3NTk2ZGIyM2U4NGRlNWJmM2M1MzE2YTI3ODEyNTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40ODI4NzU4MiwyMy4wMTAyMTE5NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lNTJlNmE4MTllZTY0ZmNjYWJjZmU1OGNmZjEwMjU3OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZTFjYzBkNjY3OTQ0NDMzOTRhYTc3NzYwZDZlYTE1OSA9ICQoJzxkaXYgaWQ9Imh0bWxfOWUxY2MwZDY2Nzk0NDQzMzk0YWE3Nzc2MGQ2ZWExNTkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Kz4HOuc6xIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTUyZTZhODE5ZWU2NGZjY2FiY2ZlNThjZmYxMDI1Nzguc2V0Q29udGVudChodG1sXzllMWNjMGQ2Njc5NDQ0MzM5NGFhNzc3NjBkNmVhMTU5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M3NzU5NmRiMjNlODRkZTViZjNjNTMxNmEyNzgxMjUwLmJpbmRQb3B1cChwb3B1cF9lNTJlNmE4MTllZTY0ZmNjYWJjZmU1OGNmZjEwMjU3OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNmFkYWY4MWU3YjE0MWY3OGQyOGE0ZjM3YjYzMmIzNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUxNjQ5NDc1LDIyLjg2NTI4MjA2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M2YzZlYzA1YTZmMzRkNGJhMDEyYjE3YmQ4ZWNkYzAyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg3ZmZlNjUwZDdjMDRiMTk4MmNmZmQ0ZGI5Zjc5OWYyID0gJCgnPGRpdiBpZD0iaHRtbF84N2ZmZTY1MGQ3YzA0YjE5ODJjZmZkNGRiOWY3OTlmMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOv8+Bz4nOvc6uz4POuSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M2YzZlYzA1YTZmMzRkNGJhMDEyYjE3YmQ4ZWNkYzAyLnNldENvbnRlbnQoaHRtbF84N2ZmZTY1MGQ3YzA0YjE5ODJjZmZkNGRiOWY3OTlmMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lNmFkYWY4MWU3YjE0MWY3OGQyOGE0ZjM3YjYzMmIzNi5iaW5kUG9wdXAocG9wdXBfYzZjNmVjMDVhNmYzNGQ0YmEwMTJiMTdiZDhlY2RjMDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTQ5NjAzNzJjZjNkNGJiM2E1NGMyYzU5YmVhYzEyZWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41MjAwMTE5LDIyLjcyOTI2MzMxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzViNzZlNDJiMDc3YjRjZTRiMDFiMWRmZjliYjM5Njg3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk0NDU5Y2Q4ZGNlNjQ3MDBhYTk4ZWEwNmMzY2U5ZWI4ID0gJCgnPGRpdiBpZD0iaHRtbF85NDQ1OWNkOGRjZTY0NzAwYWE5OGVhMDZjM2NlOWViOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOuc6yzq3Pgc65zr/OvSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzViNzZlNDJiMDc3YjRjZTRiMDFiMWRmZjliYjM5Njg3LnNldENvbnRlbnQoaHRtbF85NDQ1OWNkOGRjZTY0NzAwYWE5OGVhMDZjM2NlOWViOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNDk2MDM3MmNmM2Q0YmIzYTU0YzJjNTliZWFjMTJlZS5iaW5kUG9wdXAocG9wdXBfNWI3NmU0MmIwNzdiNGNlNGIwMWIxZGZmOWJiMzk2ODcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWZiYjA2OGY3NDY2NGI3OGJlNGVlMDg3NWJlOTY0NzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41MjI2NDc4NiwyMi41ODAzMDg5MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZTZiMzJlMmU1N2I0NjlhYmExNTM1NzViMzA3MDRlNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MmRkZDEyZjg5ZmU0MmE2YTkzNzYyNzM5MzFkMmI0ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfNzJkZGQxMmY4OWZlNDJhNmE5Mzc2MjczOTMxZDJiNGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Rz4fOu86xzrTPjM66zrHOvM+Azr/PgiBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZlNmIzMmUyZTU3YjQ2OWFiYTE1MzU3NWIzMDcwNGU2LnNldENvbnRlbnQoaHRtbF83MmRkZDEyZjg5ZmU0MmE2YTkzNzYyNzM5MzFkMmI0Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lZmJiMDY4Zjc0NjY0Yjc4YmU0ZWUwODc1YmU5NjQ3MC5iaW5kUG9wdXAocG9wdXBfNmU2YjMyZTJlNTdiNDY5YWJhMTUzNTc1YjMwNzA0ZTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjQ1MThhMGFiMDM3NGJlMWJkNDg3Yzc3ZDUzZTlmZjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41MzgwODIxMiwyMi44OTAzMTIxOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMzAyMTZlNWRjMDg0Y2Q4YTE3NWM0NmFlMzkyODc1NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZmExMWZlMWRmMmQ0M2Q2YWM5NTUzYjk1ODg4N2Q0NCA9ICQoJzxkaXYgaWQ9Imh0bWxfOWZhMTFmZTFkZjJkNDNkNmFjOTU1M2I5NTg4ODdkNDQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Uz4HOrc+AzrHOvc6/zr0gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMzAyMTZlNWRjMDg0Y2Q4YTE3NWM0NmFlMzkyODc1NS5zZXRDb250ZW50KGh0bWxfOWZhMTFmZTFkZjJkNDNkNmFjOTU1M2I5NTg4ODdkNDQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjQ1MThhMGFiMDM3NGJlMWJkNDg3Yzc3ZDUzZTlmZjIuYmluZFBvcHVwKHBvcHVwX2YzMDIxNmU1ZGMwODRjZDhhMTc1YzQ2YWUzOTI4NzU1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIyYzEzNTYwZjdkNTRiOTRiZDZjMGFkYTNkMWIxZTdmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTQzMDQxMjMsMjIuODYxNTM3OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjhiNDA4YjlmYzEwNDYyZmFjN2NkMGIyNGFhNmM3YzIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjA4MTI1ZjZiZTE5NGI2MmJkMGQzYjI5MzNlY2UzMWUgPSAkKCc8ZGl2IGlkPSJodG1sXzIwODEyNWY2YmUxOTRiNjJiZDBkM2IyOTMzZWNlMzFlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc+Dzq/Ovc63IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjhiNDA4YjlmYzEwNDYyZmFjN2NkMGIyNGFhNmM3YzIuc2V0Q29udGVudChodG1sXzIwODEyNWY2YmUxOTRiNjJiZDBkM2IyOTMzZWNlMzFlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIyYzEzNTYwZjdkNTRiOTRiZDZjMGFkYTNkMWIxZTdmLmJpbmRQb3B1cChwb3B1cF9iOGI0MDhiOWZjMTA0NjJmYWM3Y2QwYjI0YWE2YzdjMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mOGI1YjkxNmZiMGE0YWI3OGY3OWY5NDAyMDVlMjk3ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU1NzMwMDU3LDIyLjg1OTA2NzkyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM0ZGYzY2UiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjNGRmM2NlIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdiMzJmMzVjMjFlOTQ3OGM4NWQwZjkyZjJkMzJkOTkzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY3YTA3MTNjZjM1ZDRjYTQ4YmZhYmIyMGM5MDQxYzc0ID0gJCgnPGRpdiBpZD0iaHRtbF82N2EwNzEzY2YzNWQ0Y2E0OGJmYWJiMjBjOTA0MWM3NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpvOtc+FzrrOrM66zrnOsSBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdiMzJmMzVjMjFlOTQ3OGM4NWQwZjkyZjJkMzJkOTkzLnNldENvbnRlbnQoaHRtbF82N2EwNzEzY2YzNWQ0Y2E0OGJmYWJiMjBjOTA0MWM3NCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mOGI1YjkxNmZiMGE0YWI3OGY3OWY5NDAyMDVlMjk3ZC5iaW5kUG9wdXAocG9wdXBfN2IzMmYzNWMyMWU5NDc4Yzg1ZDBmOTJmMmQzMmQ5OTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWI2NDE3NWI2ZjFmNDQ5ZmFiMTFiNDBkYTlhN2FmMmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NjI5NjUzOSwyMy4xNDk0NzcwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82M2M5NzhjMDMxN2M0NTU3Yjk1OTU4YzI4ODU3MmM1ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZTMwODViNzEzNzk0YTIyOTVlYzE1MDMyZDRhZjhlMSA9ICQoJzxkaXYgaWQ9Imh0bWxfZmUzMDg1YjcxMzc5NGEyMjk1ZWMxNTAzMmQ0YWY4ZTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6kz4HOsc+HzrXOuc6sIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjNjOTc4YzAzMTdjNDU1N2I5NTk1OGMyODg1NzJjNWQuc2V0Q29udGVudChodG1sX2ZlMzA4NWI3MTM3OTRhMjI5NWVjMTUwMzJkNGFmOGUxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FiNjQxNzViNmYxZjQ0OWZhYjExYjQwZGE5YTdhZjJkLmJpbmRQb3B1cChwb3B1cF82M2M5NzhjMDMxN2M0NTU3Yjk1OTU4YzI4ODU3MmM1ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hOWVkYTdlYTQ1MGI0ZGQ1YjBiYjRmMGQ4ZTI2YmVhNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU2MzE2NzU3LDIyLjY4NTA4NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWMyMmI2YzQ1MmIxNGJjZGE3NzgyYzY5ZmFkMTI0YjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjE5YjI0YzY2YTA4NGU0YWEyODYwMjFmZWE4MTAzYWQgPSAkKCc8ZGl2IGlkPSJodG1sXzYxOWIyNGM2NmEwODRlNGFhMjg2MDIxZmVhODEwM2FkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oo866zrHPhs65zrTOrM66zrnOv869IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWMyMmI2YzQ1MmIxNGJjZGE3NzgyYzY5ZmFkMTI0YjQuc2V0Q29udGVudChodG1sXzYxOWIyNGM2NmEwODRlNGFhMjg2MDIxZmVhODEwM2FkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E5ZWRhN2VhNDUwYjRkZDViMGJiNGYwZDhlMjZiZWE3LmJpbmRQb3B1cChwb3B1cF8xYzIyYjZjNDUyYjE0YmNkYTc3ODJjNjlmYWQxMjRiNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYmM3YTc0N2M0Mzc0OGE2YTNiYjQ5NGQ3NzAyNWUzNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU2MzkxOTA3LDIyLjc5NzA1NDI5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NmOTQ4Yjk2MzljNzQ5ZGNhZjEwODUwZjNkYjc4NGY2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E5YzFlYTJjNWNlZjQ1Y2U5ODc2NzNjMDJkZDIxZGI4ID0gJCgnPGRpdiBpZD0iaHRtbF9hOWMxZWEyYzVjZWY0NWNlOTg3NjczYzAyZGQyMWRiOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zp3Osc+Nz4DOu865zr/OvSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NmOTQ4Yjk2MzljNzQ5ZGNhZjEwODUwZjNkYjc4NGY2LnNldENvbnRlbnQoaHRtbF9hOWMxZWEyYzVjZWY0NWNlOTg3NjczYzAyZGQyMWRiOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYmM3YTc0N2M0Mzc0OGE2YTNiYjQ5NGQ3NzAyNWUzNi5iaW5kUG9wdXAocG9wdXBfY2Y5NDhiOTYzOWM3NDlkY2FmMTA4NTBmM2RiNzg0ZjYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjVjZTFiNzQ2NDMzNDhiZTk4ZWJhNjNlMGVkMmE2YWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NjgzOTc1MiwyMi44MjgxNDIxN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iY2NiMjViZWM2YWY0M2FmOTRjMjMxMjI5OWQzNTEzMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNmI5MWE3Mzc4NTI0ZTE5OTI4OTJjZDUyZmQ3ZWRhOCA9ICQoJzxkaXYgaWQ9Imh0bWxfZTZiOTFhNzM3ODUyNGUxOTkyODkyY2Q1MmZkN2VkYTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPsK2z4HOuc6xIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmNjYjI1YmVjNmFmNDNhZjk0YzIzMTIyOTlkMzUxMzEuc2V0Q29udGVudChodG1sX2U2YjkxYTczNzg1MjRlMTk5Mjg5MmNkNTJmZDdlZGE4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI1Y2UxYjc0NjQzMzQ4YmU5OGViYTYzZTBlZDJhNmFlLmJpbmRQb3B1cChwb3B1cF9iY2NiMjViZWM2YWY0M2FmOTRjMjMxMjI5OWQzNTEzMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYzA0ODI3YmQ0MTM0OTc1OTM0YTVmMGI0NzFlNGNjYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU4MDUwOTE5LDIyLjg3Nzc1NjEyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQzZWJkMzU4ZDMxYTQ1NjNiYzg3MDk3Y2Y0Y2NiMWNmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EyMmM5NTA2MGFlNjRjOWI5MjJhYzY2MThiZWI1ODIyID0gJCgnPGRpdiBpZD0iaHRtbF9hMjJjOTUwNjBhZTY0YzliOTIyYWM2NjE4YmViNTgyMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDPhc+BzrPOuc+Oz4TOuc66zrEgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80M2ViZDM1OGQzMWE0NTYzYmM4NzA5N2NmNGNjYjFjZi5zZXRDb250ZW50KGh0bWxfYTIyYzk1MDYwYWU2NGM5YjkyMmFjNjYxOGJlYjU4MjIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2MwNDgyN2JkNDEzNDk3NTkzNGE1ZjBiNDcxZTRjY2MuYmluZFBvcHVwKHBvcHVwXzQzZWJkMzU4ZDMxYTQ1NjNiYzg3MDk3Y2Y0Y2NiMWNmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FmYzk2YzNlMTczNTQxOWE4ZGY1YzA3ZmEwOWM0NjI3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTg0NTE4NDMsMjIuNzQzNDc0OTZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmNmYWQ4ZDYyMjJjNDdmOThhMGMzYjBmNjYzYTMwYzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmNjYjdmZjZiZmI4NGQ1MjhhZTNlODhjZWFhNjYyNTQgPSAkKCc8ZGl2IGlkPSJodG1sXzJjY2I3ZmY2YmZiODRkNTI4YWUzZTg4Y2VhYTY2MjU0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Onc6tzrEgzprOr86/z4IgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iY2ZhZDhkNjIyMmM0N2Y5OGEwYzNiMGY2NjNhMzBjMS5zZXRDb250ZW50KGh0bWxfMmNjYjdmZjZiZmI4NGQ1MjhhZTNlODhjZWFhNjYyNTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWZjOTZjM2UxNzM1NDE5YThkZjVjMDdmYTA5YzQ2MjcuYmluZFBvcHVwKHBvcHVwX2JjZmFkOGQ2MjIyYzQ3Zjk4YTBjM2IwZjY2M2EzMGMxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJlMzU3YTEzY2Y3ZTQ2MGRiNzVmNDU4Mzk3ZDBmNGFlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTkzNTAyMDQsMjIuNzAwNDk4NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmOTY0ZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjk2NGYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTkzMWQzNzA1YjA3NGY5NmFmNDZlNDQ5MTk1MGRiYjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDVlN2ZkMTFkYWMyNGQyMWE5MzQ1MzllOTc3ZTNhODEgPSAkKCc8ZGl2IGlkPSJodG1sXzQ1ZTdmZDExZGFjMjRkMjFhOTM0NTM5ZTk3N2UzYTgxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms61z4bOsc67zqzPgc65zr/OvSBDbHVzdGVyIDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U5MzFkMzcwNWIwNzRmOTZhZjQ2ZTQ0OTE5NTBkYmI0LnNldENvbnRlbnQoaHRtbF80NWU3ZmQxMWRhYzI0ZDIxYTkzNDUzOWU5NzdlM2E4MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yZTM1N2ExM2NmN2U0NjBkYjc1ZjQ1ODM5N2QwZjRhZS5iaW5kUG9wdXAocG9wdXBfZTkzMWQzNzA1YjA3NGY5NmFmNDZlNDQ5MTk1MGRiYjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzMyNzVhZmUxNTEwNGJiMDk5NjdiMGJkZWQ5Y2IwOWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41OTcyNDQyNiwyMi44NDM2NjQxN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmY5NjRmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmOTY0ZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yZmU1YWExYTAwODA0MGUxYjY0Yzg2ZjgwMDc1MDkwMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yMDZmMDM1ZjdjZGM0ZDk2YTVmMjMxZDkyYzEwNmUxMyA9ICQoJzxkaXYgaWQ9Imh0bWxfMjA2ZjAzNWY3Y2RjNGQ5NmE1ZjIzMWQ5MmMxMDZlMTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPsK2zrPOuc6/z4IgzpHOtM+BzrnOsc69z4zPgiBDbHVzdGVyIDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJmZTVhYTFhMDA4MDQwZTFiNjRjODZmODAwNzUwOTAyLnNldENvbnRlbnQoaHRtbF8yMDZmMDM1ZjdjZGM0ZDk2YTVmMjMxZDkyYzEwNmUxMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83MzI3NWFmZTE1MTA0YmIwOTk2N2IwYmRlZDljYjA5YS5iaW5kUG9wdXAocG9wdXBfMmZlNWFhMWEwMDgwNDBlMWI2NGM4NmY4MDA3NTA5MDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzc4MTNmZjgzYTZmNDYwOTk0ZWFmMmViMDUxYzg0MDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MDM0NjYwMywyMi45NDQwMjY5NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85MGYxYWVhYjM3M2Y0NWUyYjFjZGViNDJkNjBiNGFjNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81NGE1NWY3MzBhYTQ0ODBjODFhZDY1YTQwMDI4MWEwNiA9ICQoJzxkaXYgaWQ9Imh0bWxfNTRhNTVmNzMwYWE0NDgwYzgxYWQ2NWE0MDAyODFhMDYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6czrXPhM+Mz4fOuc6/zr0gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MGYxYWVhYjM3M2Y0NWUyYjFjZGViNDJkNjBiNGFjNi5zZXRDb250ZW50KGh0bWxfNTRhNTVmNzMwYWE0NDgwYzgxYWQ2NWE0MDAyODFhMDYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzc4MTNmZjgzYTZmNDYwOTk0ZWFmMmViMDUxYzg0MDEuYmluZFBvcHVwKHBvcHVwXzkwZjFhZWFiMzczZjQ1ZTJiMWNkZWI0MmQ2MGI0YWM2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU3NDdkNzc5MDgzYzQ0Y2VhM2JiN2EyNGNjN2Y4MWQ5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjA1MzA4NTMsMjIuODE4MjQ4NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGI0MjQ0YmY5ZGViNDQ2Mjg2NzAxNmM0ZTMwMjdjNDkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzVjOWI5NjBjNDAzNDU0MGExNWY0Njg3ZmUzMTEzYzUgPSAkKCc8ZGl2IGlkPSJodG1sX2M1YzliOTYwYzQwMzQ1NDBhMTVmNDY4N2ZlMzExM2M1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Onc6tzrEgzqTOr8+Bz4XOvc+CIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGI0MjQ0YmY5ZGViNDQ2Mjg2NzAxNmM0ZTMwMjdjNDkuc2V0Q29udGVudChodG1sX2M1YzliOTYwYzQwMzQ1NDBhMTVmNDY4N2ZlMzExM2M1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU3NDdkNzc5MDgzYzQ0Y2VhM2JiN2EyNGNjN2Y4MWQ5LmJpbmRQb3B1cChwb3B1cF9kYjQyNDRiZjlkZWI0NDYyODY3MDE2YzRlMzAyN2M0OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMzIyOTEwZjBkMjE0MTY1ODczZjZiNTJkY2Y3OGU3OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYxMTE3MTcyLDIzLjAzNjk0NzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk1MzRmY2YyN2I3ZTRjZGI4ZGUxODAwOWRlYjVkMGE1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y2OWJmZDZlNDU2NjRhMGJiNWY3ZTBiZGNhMzNiYTg0ID0gJCgnPGRpdiBpZD0iaHRtbF9mNjliZmQ2ZTQ1NjY0YTBiYjVmN2UwYmRjYTMzYmE4NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpvPhc6zzr/Pjc+BzrnOv869IENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTUzNGZjZjI3YjdlNGNkYjhkZTE4MDA5ZGViNWQwYTUuc2V0Q29udGVudChodG1sX2Y2OWJmZDZlNDU2NjRhMGJiNWY3ZTBiZGNhMzNiYTg0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QzMjI5MTBmMGQyMTQxNjU4NzNmNmI1MmRjZjc4ZTc5LmJpbmRQb3B1cChwb3B1cF85NTM0ZmNmMjdiN2U0Y2RiOGRlMTgwMDlkZWI1ZDBhNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NDgxZDE3NDQ3N2U0ZThhYjU1ZjYxN2QxMzIwYjY0MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYxMzEwMTk2LDIyLjg1NjY1MTMxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y5ZmQwYjNlZmRlNjQyYjdhMThiMzFlZWZiYmExYjUyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA0MWNiZjZkYTg1ZDRhOTRhODU2NGJjYjdmZWM5OTA2ID0gJCgnPGRpdiBpZD0iaHRtbF8wNDFjYmY2ZGE4NWQ0YTk0YTg1NjRiY2I3ZmVjOTkwNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zp3Orc6/zr0gzqHOv861zrnOvc+Mzr0gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mOWZkMGIzZWZkZTY0MmI3YTE4YjMxZWVmYmJhMWI1Mi5zZXRDb250ZW50KGh0bWxfMDQxY2JmNmRhODVkNGE5NGE4NTY0YmNiN2ZlYzk5MDYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTQ4MWQxNzQ0NzdlNGU4YWI1NWY2MTdkMTMyMGI2NDIuYmluZFBvcHVwKHBvcHVwX2Y5ZmQwYjNlZmRlNjQyYjdhMThiMzFlZWZiYmExYjUyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MwNDNiMzMwOTBmODRmNzM5OGM1Zjg3NTYzOWJkODM0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjE1ODkwNTAwMDAwMDA2LDIyLjc2OTQ5ODgzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc1NjUwMWU4ZTE1MDQ2OGZhNzA1MmYzNjZiZTZiYzJmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE5YjlkYzliMGMyYTQwYjFhM2NkNWM1OTA2NzU5NjE5ID0gJCgnPGRpdiBpZD0iaHRtbF8xOWI5ZGM5YjBjMmE0MGIxYTNjZDVjNTkwNjc1OTYxOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpTOsc67zrHOvM6xzr3OrM+BzrEgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NTY1MDFlOGUxNTA0NjhmYTcwNTJmMzY2YmU2YmMyZi5zZXRDb250ZW50KGh0bWxfMTliOWRjOWIwYzJhNDBiMWEzY2Q1YzU5MDY3NTk2MTkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzA0M2IzMzA5MGY4NGY3Mzk4YzVmODc1NjM5YmQ4MzQuYmluZFBvcHVwKHBvcHVwXzc1NjUwMWU4ZTE1MDQ2OGZhNzA1MmYzNjZiZTZiYzJmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkyYTA0OGNjY2RjNjQ3YTc5MjdlZmJlOGUxM2RkNTM5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjMwMjkwOTksMjIuNzY1MzY1NjAwMDAwMDAzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjk2NGYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmY5NjRmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FiNGYyYTc1Yjk2MzQzNWNhNThhYmE5ZDQ1ZDFkZTI3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M1NzNlYzc3ZWY3NDRhZTA4NWZhYzA3OTcxNjYwZTZlID0gJCgnPGRpdiBpZD0iaHRtbF9jNTczZWM3N2VmNzQ0YWUwODVmYWMwNzk3MTY2MGU2ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDPhc+BzrPOrc67zrvOsSBDbHVzdGVyIDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FiNGYyYTc1Yjk2MzQzNWNhNThhYmE5ZDQ1ZDFkZTI3LnNldENvbnRlbnQoaHRtbF9jNTczZWM3N2VmNzQ0YWUwODVmYWMwNzk3MTY2MGU2ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85MmEwNDhjY2NkYzY0N2E3OTI3ZWZiZThlMTNkZDUzOS5iaW5kUG9wdXAocG9wdXBfYWI0ZjJhNzViOTYzNDM1Y2E1OGFiYTlkNDVkMWRlMjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjk2ZjE3MDI5YmE5NDU5MGFhOWZkOGI2M2NhYmI2NmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MzUzMzAyMDAwMDAwMDYsMjMuMTUzNTAzNDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTZjY2RhNmQ3NWUzNDA5MWExYjVlZTIzNTRkYmIyYTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGZhM2FiYzYzYzUyNDNhZDllNjAzNzMwMTUxN2FjZTkgPSAkKCc8ZGl2IGlkPSJodG1sXzRmYTNhYmM2M2M1MjQzYWQ5ZTYwMzczMDE1MTdhY2U5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc+Bz4fOsc6vzrEgzpXPgM6vzrTOsc+Fz4HOv8+CIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTZjY2RhNmQ3NWUzNDA5MWExYjVlZTIzNTRkYmIyYTcuc2V0Q29udGVudChodG1sXzRmYTNhYmM2M2M1MjQzYWQ5ZTYwMzczMDE1MTdhY2U5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y5NmYxNzAyOWJhOTQ1OTBhYTlmZDhiNjNjYWJiNjZhLmJpbmRQb3B1cChwb3B1cF8xNmNjZGE2ZDc1ZTM0MDkxYTFiNWVlMjM1NGRiYjJhNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMGRiOGViZTUwOGU0MDYwYTM5OTU2MTBkYjM3Yjk5NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYzMjAzNDMsMjIuNzI3NzY0MTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGI5NzA0YTZjMjk4NGNhYmFhNTBlYzAzMWY4ZDVjOWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmViZDk1NmYyZWE1NGFjNThkZjljZDM2NDkzZDk4YWIgPSAkKCc8ZGl2IGlkPSJodG1sXzZlYmQ5NTZmMmVhNTRhYzU4ZGY5Y2QzNjQ5M2Q5OGFiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Cts+BzrPOv8+CIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGI5NzA0YTZjMjk4NGNhYmFhNTBlYzAzMWY4ZDVjOWIuc2V0Q29udGVudChodG1sXzZlYmQ5NTZmMmVhNTRhYzU4ZGY5Y2QzNjQ5M2Q5OGFiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzEwZGI4ZWJlNTA4ZTQwNjBhMzk5NTYxMGRiMzdiOTk3LmJpbmRQb3B1cChwb3B1cF9kYjk3MDRhNmMyOTg0Y2FiYWE1MGVjMDMxZjhkNWM5Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82NGYyNjQ3NjdmYTA0ZjMwODZlMTBmZjAyMTZmN2E3NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYzNDI5NjQyLDIyLjgwNDc3NTI0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM3YjhkZWUwNGNiNjQzZDA4YTk2MzdhNmI5OTdiYzQ1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc5YTIzZGI2ODAwZTRkM2Q4YTdiNWExMGIyMmE4NGZjID0gJCgnPGRpdiBpZD0iaHRtbF83OWEyM2RiNjgwMGU0ZDNkOGE3YjVhMTBiMjJhODRmYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpHOs86vzrEgzqTPgc65zqzOtM6xIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzdiOGRlZTA0Y2I2NDNkMDhhOTYzN2E2Yjk5N2JjNDUuc2V0Q29udGVudChodG1sXzc5YTIzZGI2ODAwZTRkM2Q4YTdiNWExMGIyMmE4NGZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY0ZjI2NDc2N2ZhMDRmMzA4NmUxMGZmMDIxNmY3YTc3LmJpbmRQb3B1cChwb3B1cF8zN2I4ZGVlMDRjYjY0M2QwOGE5NjM3YTZiOTk3YmM0NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNWQ5ZjhmNmJjNjA0ZjFlYTg3OTg2OTA5YmEwY2MzOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYzNjgyOTM4LDIyLjU0NDMyMjk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUzYTFkNDMwNWIzZjRjYTc5ZWZjNDA5Y2IwNTg3YzdjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVmZjk5M2JmMGQ1ZjQxNmM4ZTllMzRmMjNhZmNmMmYzID0gJCgnPGRpdiBpZD0iaHRtbF81ZmY5OTNiZjBkNWY0MTZjOGU5ZTM0ZjIzYWZjZjJmMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOsc+Bz4XOrCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUzYTFkNDMwNWIzZjRjYTc5ZWZjNDA5Y2IwNTg3YzdjLnNldENvbnRlbnQoaHRtbF81ZmY5OTNiZjBkNWY0MTZjOGU5ZTM0ZjIzYWZjZjJmMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mNWQ5ZjhmNmJjNjA0ZjFlYTg3OTg2OTA5YmEwY2MzOS5iaW5kUG9wdXAocG9wdXBfNTNhMWQ0MzA1YjNmNGNhNzllZmM0MDljYjA1ODdjN2MpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzk4MmVmOGI1Yjk3NDY1ZTg3Y2Q5MDA5ZDdmZTIxNGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NDAzOTk5MywyMi43ODc1MjUxOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lYjMwYThhYWE2Yzg0Yjk5YmI5MjM4MDk0ZmViYTQwOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wNTQ2OTJlNzJjNjQ0NzNjYjY5ODYzMGU2M2E2OWY0OSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDU0NjkyZTcyYzY0NDczY2I2OTg2MzBlNjNhNjlmNDkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Xz4HOsc6vzr/OvSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ViMzBhOGFhYTZjODRiOTliYjkyMzgwOTRmZWJhNDA5LnNldENvbnRlbnQoaHRtbF8wNTQ2OTJlNzJjNjQ0NzNjYjY5ODYzMGU2M2E2OWY0OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83OTgyZWY4YjViOTc0NjVlODdjZDkwMDlkN2ZlMjE0YS5iaW5kUG9wdXAocG9wdXBfZWIzMGE4YWFhNmM4NGI5OWJiOTIzODA5NGZlYmE0MDkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDdkNjViYjA0MTZjNDAyNzg2YjJlN2FjNTAzMWM0ZGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NTEwMDQ3OSwyMi43NjAyMTM4NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lNDAzNjQ3ZGRjYmM0MmZlOWVlNjBhMDc2ZTYxYzU0ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMmY4Mjc3Y2E1ZjU0YTY2ODNjNWQ2NTZiYjQ0ZDY4ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfZjJmODI3N2NhNWY1NGE2NjgzYzVkNjU2YmI0NGQ2OGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Jz4HOsSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U0MDM2NDdkZGNiYzQyZmU5ZWU2MGEwNzZlNjFjNTRlLnNldENvbnRlbnQoaHRtbF9mMmY4Mjc3Y2E1ZjU0YTY2ODNjNWQ2NTZiYjQ0ZDY4ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wN2Q2NWJiMDQxNmM0MDI3ODZiMmU3YWM1MDMxYzRkZi5iaW5kUG9wdXAocG9wdXBfZTQwMzY0N2RkY2JjNDJmZTllZTYwYTA3NmU2MWM1NGUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDRjNzQ0ZjczZjU0NGFkYWE1ZGVhNmE5ZTBjZTM2MzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NTU2MDUzMiwyMi43ODgzNTg2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80YmM0ODA0NDk0OGM0MjRiYTM4NzYwNzhjYmQzZGU1MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yYzRmNWEzOTFmN2I0Yjc0YmZlZDc1ODk4ZWJjOGM4NSA9ICQoJzxkaXYgaWQ9Imh0bWxfMmM0ZjVhMzkxZjdiNGI3NGJmZWQ3NTg5OGViYzhjODUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Rzr3Pjc+GzrnOv869IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGJjNDgwNDQ5NDhjNDI0YmEzODc2MDc4Y2JkM2RlNTMuc2V0Q29udGVudChodG1sXzJjNGY1YTM5MWY3YjRiNzRiZmVkNzU4OThlYmM4Yzg1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA0Yzc0NGY3M2Y1NDRhZGFhNWRlYTZhOWUwY2UzNjMwLmJpbmRQb3B1cChwb3B1cF80YmM0ODA0NDk0OGM0MjRiYTM4NzYwNzhjYmQzZGU1Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hNWMxMDUxMWZhYTQ0MDZiYWI2OTg1MGNmOGU3YzAwYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjY2MTc4MTMxLDIyLjc0ODU3NzEyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U3YjI2NzBkMzgyODQwMWRiMDQ2Zjc0M2U5ZmY4MzBlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRhNjFiYmY4ODdlODRlNmViN2VjY2UwMTE5MmU5N2UyID0gJCgnPGRpdiBpZD0iaHRtbF80YTYxYmJmODg3ZTg0ZTZlYjdlY2NlMDExOTJlOTdlMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zorOvc6xz4fOv8+CIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTdiMjY3MGQzODI4NDAxZGIwNDZmNzQzZTlmZjgzMGUuc2V0Q29udGVudChodG1sXzRhNjFiYmY4ODdlODRlNmViN2VjY2UwMTE5MmU5N2UyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E1YzEwNTExZmFhNDQwNmJhYjY5ODUwY2Y4ZTdjMDBhLmJpbmRQb3B1cChwb3B1cF9lN2IyNjcwZDM4Mjg0MDFkYjA0NmY3NDNlOWZmODMwZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zOTk0MWMyOGJkNzM0ZjFhOTFmYzYxNDI3ZDViZjMxNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjY3MzY4MzE3LDIzLjEyNzA3MTM4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RkY2VkYmI1MGUzMDQ4Y2JiNTNiY2IwMTJiYjdjYTYxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVlYzYwZGZiMWM5NDRjNDI5ZDY3MTViMzQzOTM4NWVkID0gJCgnPGRpdiBpZD0iaHRtbF81ZWM2MGRmYjFjOTQ0YzQyOWQ2NzE1YjM0MzkzODVlZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zp3Orc6xIM6Vz4DOr860zrHPhc+Bzr/PgiBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RkY2VkYmI1MGUzMDQ4Y2JiNTNiY2IwMTJiYjdjYTYxLnNldENvbnRlbnQoaHRtbF81ZWM2MGRmYjFjOTQ0YzQyOWQ2NzE1YjM0MzkzODVlZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zOTk0MWMyOGJkNzM0ZjFhOTFmYzYxNDI3ZDViZjMxNi5iaW5kUG9wdXAocG9wdXBfZGRjZWRiYjUwZTMwNDhjYmI1M2JjYjAxMmJiN2NhNjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODE4Njg5NTNjOWJkNDAxODljMTUxZGY0OTE3MjM4YWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NzYwOTc4NywyMi45NTU5MDc4Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81NzdlODVmNjY0ZGY0YzliYmUxZjM5ZGIwZDUzYTBjNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jYjU0N2Q3ZjNmMWY0ODk1YjkwNzc4NjQyMjBjZDEzNSA9ICQoJzxkaXYgaWQ9Imh0bWxfY2I1NDdkN2YzZjFmNDg5NWI5MDc3ODY0MjIwY2QxMzUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Rz4HOsc+Hzr3Osc6vzr/OvSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU3N2U4NWY2NjRkZjRjOWJiZTFmMzlkYjBkNTNhMGM2LnNldENvbnRlbnQoaHRtbF9jYjU0N2Q3ZjNmMWY0ODk1YjkwNzc4NjQyMjBjZDEzNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MTg2ODk1M2M5YmQ0MDE4OWMxNTFkZjQ5MTcyMzhhZi5iaW5kUG9wdXAocG9wdXBfNTc3ZTg1ZjY2NGRmNGM5YmJlMWYzOWRiMGQ1M2EwYzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWE1OTgzMTcxODZlNDFiY2I0NTdkMWU2MjJjZTY1OGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42ODA1MzQzNiwyMi43MTM2OTkzNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85YjkzMDJjYzVjNTQ0NjA4YjZhMTJhMTYxOWE3ZTkwYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84ZTZhMjAzMjdhY2Q0NmQ3YjI5YWM2MjM2NjQ2NDViYSA9ICQoJzxkaXYgaWQ9Imh0bWxfOGU2YTIwMzI3YWNkNDZkN2IyOWFjNjIzNjY0NjQ1YmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6azr/Phc+Ez4POv8+Az4zOtM65zr/OvSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzliOTMwMmNjNWM1NDQ2MDhiNmExMmExNjE5YTdlOTBhLnNldENvbnRlbnQoaHRtbF84ZTZhMjAzMjdhY2Q0NmQ3YjI5YWM2MjM2NjQ2NDViYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85YTU5ODMxNzE4NmU0MWJjYjQ1N2QxZTYyMmNlNjU4YS5iaW5kUG9wdXAocG9wdXBfOWI5MzAyY2M1YzU0NDYwOGI2YTEyYTE2MTlhN2U5MGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDNlNjJlZjc0MDJmNDk4ZmE0OTk5NmRhYzJkZjBmZWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42ODk5OTg2MywyMy4wNzA0ODc5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZjk2ZmI5MTk3ZmM0YjdiODUwYjYyZTBiMjMwZDgyYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xYWIzY2FhODMwYmY0NmE2YjhhNGMxNGNkOTNmMTdhNiA9ICQoJzxkaXYgaWQ9Imh0bWxfMWFiM2NhYTgzMGJmNDZhNmI4YTRjMTRjZDkzZjE3YTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Uzq7OvM6xzrnOvc6xIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmY5NmZiOTE5N2ZjNGI3Yjg1MGI2MmUwYjIzMGQ4MmEuc2V0Q29udGVudChodG1sXzFhYjNjYWE4MzBiZjQ2YTZiOGE0YzE0Y2Q5M2YxN2E2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzAzZTYyZWY3NDAyZjQ5OGZhNDk5OTZkYWMyZGYwZmVmLmJpbmRQb3B1cChwb3B1cF9iZjk2ZmI5MTk3ZmM0YjdiODUwYjYyZTBiMjMwZDgyYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mN2EwNWJmODA3YjI0Y2MwOGQzODExMTQ3NzEwMGViMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjcwMjE3MTMzLDIyLjc0ODIwMzI4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjk2NGYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmY5NjRmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI3ZTM2OWI5ZWFlNzQ5Yzg5MTA2OTQ2YTc0MDczZTE1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMxNDgwMTVmNDFhODQ1ZTI5ZTNjMmZjNDM3MTI1NTNiID0gJCgnPGRpdiBpZD0iaHRtbF8zMTQ4MDE1ZjQxYTg0NWUyOWUzYzJmYzQzNzEyNTUzYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpzOv869zrHPg8+EzrfPgc6szrrOuc6/zr0gQ2x1c3RlciA1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yN2UzNjliOWVhZTc0OWM4OTEwNjk0NmE3NDA3M2UxNS5zZXRDb250ZW50KGh0bWxfMzE0ODAxNWY0MWE4NDVlMjllM2MyZmM0MzcxMjU1M2IpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjdhMDViZjgwN2IyNGNjMDhkMzgxMTE0NzcxMDBlYjMuYmluZFBvcHVwKHBvcHVwXzI3ZTM2OWI5ZWFlNzQ5Yzg5MTA2OTQ2YTc0MDczZTE1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVhNGJkMmM1Yzg2MjQ0NjRiNGU0ZTFlOTc4OTNlYzU2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNzAxNjAyOTQsMjIuNTQ5NTk4NjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWI3NWZlODEzNGMwNDdjMGE0YWRlOWFkMDYzOWZhNDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzFkNDA1NDIyNDRjNGI5MjljNWNjOGI5YzIyNDc3OWEgPSAkKCc8ZGl2IGlkPSJodG1sXzMxZDQwNTQyMjQ0YzRiOTI5YzVjYzhiOWMyMjQ3NzlhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Om8+Nz4HOus61zrnOsSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FiNzVmZTgxMzRjMDQ3YzBhNGFkZTlhZDA2MzlmYTQ3LnNldENvbnRlbnQoaHRtbF8zMWQ0MDU0MjI0NGM0YjkyOWM1Y2M4YjljMjI0Nzc5YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81YTRiZDJjNWM4NjI0NDY0YjRlNGUxZTk3ODkzZWM1Ni5iaW5kUG9wdXAocG9wdXBfYWI3NWZlODEzNGMwNDdjMGE0YWRlOWFkMDYzOWZhNDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWM0ZWFiNGEyMDA1NGQwYjhkZjBiMjFiMzMxOTM2YmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy43MTA1NDA3NywyMi44NDAwOTE3MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNTgxZTdiODIxNTc0MmQ2OGQyMWY1ZTE1Yjc2ZDIwMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZjhlZWRhN2ZlZDg0M2UyOWYyMWM3NzhjMDQ0YWI1YSA9ICQoJzxkaXYgaWQ9Imh0bWxfOWY4ZWVkYTdmZWQ4NDNlMjlmMjFjNzc4YzA0NGFiNWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gz4HPjM+Dz4XOvM69zrEgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNTgxZTdiODIxNTc0MmQ2OGQyMWY1ZTE1Yjc2ZDIwMi5zZXRDb250ZW50KGh0bWxfOWY4ZWVkYTdmZWQ4NDNlMjlmMjFjNzc4YzA0NGFiNWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWM0ZWFiNGEyMDA1NGQwYjhkZjBiMjFiMzMxOTM2YmMuYmluZFBvcHVwKHBvcHVwXzI1ODFlN2I4MjE1NzQyZDY4ZDIxZjVlMTViNzZkMjAyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NhZjY1ODFhNzhjODQ1M2FiOTQzMzNjYTQ2ZjY4MTI2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNzEwMjA1MDgsMjIuODc4Nzg3OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmEwZTAwNTgzZTQwNDI2ODhmNWI0Y2QyZDI4NTFmNWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDg4ZTk1NGM0YTRiNDU3ZWI0NDNiZGI2ODIxODY4MGIgPSAkKCc8ZGl2IGlkPSJodG1sXzQ4OGU5NTRjNGE0YjQ1N2ViNDQzYmRiNjgyMTg2ODBiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Om86vzrzOvc6xzrkgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mYTBlMDA1ODNlNDA0MjY4OGY1YjRjZDJkMjg1MWY1YS5zZXRDb250ZW50KGh0bWxfNDg4ZTk1NGM0YTRiNDU3ZWI0NDNiZGI2ODIxODY4MGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2FmNjU4MWE3OGM4NDUzYWI5NDMzM2NhNDZmNjgxMjYuYmluZFBvcHVwKHBvcHVwX2ZhMGUwMDU4M2U0MDQyNjg4ZjViNGNkMmQyODUxZjVhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y3NjVhNjE2ODNhZTQ1YTdhMDFmNGVhNmRiODk3YjM5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNzEwNTk0MTgsMjIuNDI0MTg4NjFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2IyZjM5NiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNiMmYzOTYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTc1ODhhZTA3ODk5NGI5NDk2MmQ1YTE5MjI0NDIzZGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjgxZDE4MzBmYzBjNDc3NjgzNzc4MWVjYjUzYTU1YmEgPSAkKCc8ZGl2IGlkPSJodG1sX2I4MWQxODMwZmMwYzQ3NzY4Mzc3ODFlY2I1M2E1NWJhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Ops+Bzr/Phc+DzrnOv8+Nzr3OsSBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU3NTg4YWUwNzg5OTRiOTQ5NjJkNWExOTIyNDQyM2RhLnNldENvbnRlbnQoaHRtbF9iODFkMTgzMGZjMGM0Nzc2ODM3NzgxZWNiNTNhNTViYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mNzY1YTYxNjgzYWU0NWE3YTAxZjRlYTZkYjg5N2IzOS5iaW5kUG9wdXAocG9wdXBfNTc1ODhhZTA3ODk5NGI5NDk2MmQ1YTE5MjI0NDIzZGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDRmNTdjNzg1MGJkNDY5ZWIxYjdlZjdkODMwNmJhMTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy43MTcwNjc3MiwyMi43NDM5MjMxOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmY5NjRmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmOTY0ZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hNWM5MGExNzgzODg0NTZhYTZiYWNmNjdjYmI1OWU1YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82YTEyNjg5ZDQ1MzQ0OTE2OTAxMmIwNmVlZTAyZjRlMiA9ICQoJzxkaXYgaWQ9Imh0bWxfNmExMjY4OWQ0NTM0NDkxNjkwMTJiMDZlZWUwMmY0ZTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6cz4XOus6uzr3Osc65IENsdXN0ZXIgNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTVjOTBhMTc4Mzg4NDU2YWE2YmFjZjY3Y2JiNTllNWEuc2V0Q29udGVudChodG1sXzZhMTI2ODlkNDUzNDQ5MTY5MDEyYjA2ZWVlMDJmNGUyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA0ZjU3Yzc4NTBiZDQ2OWViMWI3ZWY3ZDgzMDZiYTE0LmJpbmRQb3B1cChwb3B1cF9hNWM5MGExNzgzODg0NTZhYTZiYWNmNjdjYmI1OWU1YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iZDcwMmJhZDEyMGM0Nzk4OTAxNmI4NzVlMTIxNmFkMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjcyMDU1MDU0LDIyLjcyMzI2Mjc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRmOWRlYzlmZDdiOTRhM2U4MjJlNTE1OWFmMTk0MDk3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYxYzdmNzE1OTM0ZDRmYTJiMTM2MmQ4YjExOGJmMWE4ID0gJCgnPGRpdiBpZD0iaHRtbF82MWM3ZjcxNTkzNGQ0ZmEyYjEzNjJkOGIxMThiZjFhOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqbOr8+Hz4TOuc6/zr0gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80ZjlkZWM5ZmQ3Yjk0YTNlODIyZTUxNTlhZjE5NDA5Ny5zZXRDb250ZW50KGh0bWxfNjFjN2Y3MTU5MzRkNGZhMmIxMzYyZDhiMTE4YmYxYTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmQ3MDJiYWQxMjBjNDc5ODkwMTZiODc1ZTEyMTZhZDMuYmluZFBvcHVwKHBvcHVwXzRmOWRlYzlmZDdiOTRhM2U4MjJlNTE1OWFmMTk0MDk3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJlNTdlNDEzZjliOTQyN2RhNzFlNjQ4MzU0OGI0YmEwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNzIzODY1NTEsMjIuNjM4NTY2OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTIxZjAyYjhmZDg4NDIyNmIxNWRlNjc1MTQ3MTg0ZWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzE3OWVmMTIwNTVmNDYwNzk4ZjIwMGI0NzVmZTQ4NjMgPSAkKCc8ZGl2IGlkPSJodG1sX2MxNzllZjEyMDU1ZjQ2MDc5OGYyMDBiNDc1ZmU0ODYzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM6xzrvOsc69z4TPgc6tzr3Ouc6/zr0gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xMjFmMDJiOGZkODg0MjI2YjE1ZGU2NzUxNDcxODRlYy5zZXRDb250ZW50KGh0bWxfYzE3OWVmMTIwNTVmNDYwNzk4ZjIwMGI0NzVmZTQ4NjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmU1N2U0MTNmOWI5NDI3ZGE3MWU2NDgzNTQ4YjRiYTAuYmluZFBvcHVwKHBvcHVwXzEyMWYwMmI4ZmQ4ODQyMjZiMTVkZTY3NTE0NzE4NGVjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FhMGNkYzMyYjJkNzQwYjZhOGMzODA2Njc3YjkzNWIzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNzg1Mjg1OTUsMjIuNDM0ODczNThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2IyZjM5NiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNiMmYzOTYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTJlNjI4NWFkNGYxNDYxOGE4YjFkYWExOTJkMDA2YzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWExMjE4Y2NkOGYzNDEyN2E4YjdhYjJjMmM4NTIzMTUgPSAkKCc8ZGl2IGlkPSJodG1sXzFhMTIxOGNjZDhmMzQxMjdhOGI3YWIyYzJjODUyMzE1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oo866zr/PhM61zrnOvc6uIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTJlNjI4NWFkNGYxNDYxOGE4YjFkYWExOTJkMDA2YzMuc2V0Q29udGVudChodG1sXzFhMTIxOGNjZDhmMzQxMjdhOGI3YWIyYzJjODUyMzE1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FhMGNkYzMyYjJkNzQwYjZhOGMzODA2Njc3YjkzNWIzLmJpbmRQb3B1cChwb3B1cF8xMmU2Mjg1YWQ0ZjE0NjE4YThiMWRhYTE5MmQwMDZjMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82OGM2ZjEyMjQwMTk0OWIzODRmYzNjMzhjMDZkYTIyNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjY3MzY1MjY1LDIyLjgyNDI4MzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGM1MGI5NTY3YmIxNGQ4NTkyODNhMWNkMDMwMjQyNmEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzIwZDRhODIzMjViNDUxMzlhNzVlMzg2NDRlZmFmMzcgPSAkKCc8ZGl2IGlkPSJodG1sXzMyMGQ0YTgyMzI1YjQ1MTM5YTc1ZTM4NjQ0ZWZhZjM3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM61z4TPjM+HzrnOv869IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGM1MGI5NTY3YmIxNGQ4NTkyODNhMWNkMDMwMjQyNmEuc2V0Q29udGVudChodG1sXzMyMGQ0YTgyMzI1YjQ1MTM5YTc1ZTM4NjQ0ZWZhZjM3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY4YzZmMTIyNDAxOTQ5YjM4NGZjM2MzOGMwNmRhMjI1LmJpbmRQb3B1cChwb3B1cF8wYzUwYjk1NjdiYjE0ZDg1OTI4M2ExY2QwMzAyNDI2YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wZDU0MjdkNWUxNjk0YzRhYWVjNzk3MzliMmE0ZmU3YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjY1NzY3NjcsMjIuODIzNDU1ODFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGM2OWRlNWI3ODFjNDVkYTg3OTQ5ZjAyZDBiZTc5NDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzU3MTljNTZkMGI1NDk4NThlMWQyN2RhZTIxMmIyNWIgPSAkKCc8ZGl2IGlkPSJodG1sXzc1NzE5YzU2ZDBiNTQ5ODU4ZTFkMjdkYWUyMTJiMjViIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM6szr3Otc+DzrfPgiBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RjNjlkZTViNzgxYzQ1ZGE4Nzk0OWYwMmQwYmU3OTQ3LnNldENvbnRlbnQoaHRtbF83NTcxOWM1NmQwYjU0OTg1OGUxZDI3ZGFlMjEyYjI1Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZDU0MjdkNWUxNjk0YzRhYWVjNzk3MzliMmE0ZmU3Yy5iaW5kUG9wdXAocG9wdXBfZGM2OWRlNWI3ODFjNDVkYTg3OTQ5ZjAyZDBiZTc5NDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGQ5MDhhMTJkMjNlNDRhOWIxYzI2MzAyNjJjN2YxN2MgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NDY5MjY4OCwyMi44MTMwNDU0OTk5OTk5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzRkZjNjZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM0ZGYzY2UiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGRjNWU2N2NhZjc0NDNjNDk2NDc5ZmRlMmYyNDQ5ZjUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTQ3MDIwYmIxOTZmNDNlMWJjNDIwNWM3OTUxYjM5NzYgPSAkKCc8ZGl2IGlkPSJodG1sXzU0NzAyMGJiMTk2ZjQzZTFiYzQyMDVjNzk1MWIzOTc2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OoM6/z4XOu867zrHOus6vzrTOsSBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhkYzVlNjdjYWY3NDQzYzQ5NjQ3OWZkZTJmMjQ0OWY1LnNldENvbnRlbnQoaHRtbF81NDcwMjBiYjE5NmY0M2UxYmM0MjA1Yzc5NTFiMzk3Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kZDkwOGExMmQyM2U0NGE5YjFjMjYzMDI2MmM3ZjE3Yy5iaW5kUG9wdXAocG9wdXBfOGRjNWU2N2NhZjc0NDNjNDk2NDc5ZmRlMmYyNDQ5ZjUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWJlMjc5Zjc2ZDFjNDYwMWE3MWNiMTBhYmJjZmQwMTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NzgyMjI2NiwyMi44MzQ3MjQ0M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZWI1NzZmOTYwNTY0OTVmODc3NWNmMjhjNDg2NGM4MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YTE3YTcyYzFhZDY0MDU0OTc4NDkwYWNiMWI2MjZjMiA9ICQoJzxkaXYgaWQ9Imh0bWxfN2ExN2E3MmMxYWQ2NDA1NDk3ODQ5MGFjYjFiNjI2YzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6RzrzPhc6zzrTOsc67zq/PhM+DzrEgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kZWI1NzZmOTYwNTY0OTVmODc3NWNmMjhjNDg2NGM4My5zZXRDb250ZW50KGh0bWxfN2ExN2E3MmMxYWQ2NDA1NDk3ODQ5MGFjYjFiNjI2YzIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWJlMjc5Zjc2ZDFjNDYwMWE3MWNiMTBhYmJjZmQwMTQuYmluZFBvcHVwKHBvcHVwX2RlYjU3NmY5NjA1NjQ5NWY4Nzc1Y2YyOGM0ODY0YzgzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2EzMjU0ODZkM2QzYTQxYzFiNGI0NmI2Mjk4MzYyYjg4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTM1MzM1NTQsMjIuODc4MzUxMjFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjBiZjFlZDY4ZmQ4NGM4NzlmZWNmZGYyN2Q3OTM3Y2EgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmZhZjI3ZDFjZGRhNDE2MmJhZjM4MTc1ZjZmMmQ4ODggPSAkKCc8ZGl2IGlkPSJodG1sX2ZmYWYyN2QxY2RkYTQxNjJiYWYzODE3NWY2ZjJkODg4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6xzrvOu865zrjOrc6xIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjBiZjFlZDY4ZmQ4NGM4NzlmZWNmZGYyN2Q3OTM3Y2Euc2V0Q29udGVudChodG1sX2ZmYWYyN2QxY2RkYTQxNjJiYWYzODE3NWY2ZjJkODg4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2EzMjU0ODZkM2QzYTQxYzFiNGI0NmI2Mjk4MzYyYjg4LmJpbmRQb3B1cChwb3B1cF82MGJmMWVkNjhmZDg0Yzg3OWZlY2ZkZjI3ZDc5MzdjYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNGMwYzRlMWJiZmY0ZjUzOGEzN2RkMDEyNmEwYTUyZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUzNjU1NjI0LDIyLjkxMDg5MDU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RiZjk0ODNhOTM5MzQyNmQ5ZjlhNzg0NGQxNjczN2FmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E0ZDBiYzcwNmFkODQyOGE5MTQ2ZjQ0ODQ4ZmQ0MDE3ID0gJCgnPGRpdiBpZD0iaHRtbF9hNGQwYmM3MDZhZDg0MjhhOTE0NmY0NDg0OGZkNDAxNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zp3Orc6xIM6czrHPgc6xzrjOrc6xIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGJmOTQ4M2E5MzkzNDI2ZDlmOWE3ODQ0ZDE2NzM3YWYuc2V0Q29udGVudChodG1sX2E0ZDBiYzcwNmFkODQyOGE5MTQ2ZjQ0ODQ4ZmQ0MDE3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U0YzBjNGUxYmJmZjRmNTM4YTM3ZGQwMTI2YTBhNTJlLmJpbmRQb3B1cChwb3B1cF9kYmY5NDgzYTkzOTM0MjZkOWY5YTc4NDRkMTY3MzdhZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81NzYwZmU1OWU1YTY0ZWU5YmVhZGQyM2QzYjA2YWI4MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUzNDc2MzM0LDIyLjkxNDk0MTc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M5N2VjNmQ3OTNmYTQ4MjdiNzFlNzNiNWI0OTAxMWU1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk4NTlmNDMwMGM3ZTQ0YmZiYmNiMjZjOGIwMjA2ODY0ID0gJCgnPGRpdiBpZD0iaHRtbF85ODU5ZjQzMDBjN2U0NGJmYmJjYjI2YzhiMDIwNjg2NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpLOuc6yzqzPgc65zr/OvSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M5N2VjNmQ3OTNmYTQ4MjdiNzFlNzNiNWI0OTAxMWU1LnNldENvbnRlbnQoaHRtbF85ODU5ZjQzMDBjN2U0NGJmYmJjYjI2YzhiMDIwNjg2NCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NzYwZmU1OWU1YTY0ZWU5YmVhZGQyM2QzYjA2YWI4My5iaW5kUG9wdXAocG9wdXBfYzk3ZWM2ZDc5M2ZhNDgyN2I3MWU3M2I1YjQ5MDExZTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDAwZmU3YzhiYzczNDQ5YmE1ZTZiODc2ZGE4Zjc0YmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NDcxNjExMDAwMDAwMDQsMjIuODU1NDM4MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWM0MTIxMDZiMzUzNGE3MjlmMjcxN2RlNmZmMzYyMjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmFmYTI5ZThkZGRiNDNiY2IxNjI5ZDFhZDJjZTcwOTUgPSAkKCc8ZGl2IGlkPSJodG1sX2ZhZmEyOWU4ZGRkYjQzYmNiMTYyOWQxYWQyY2U3MDk1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc6zzq/OsSDOoM6xz4HOsc+DzrrOtc+Fzq4gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hYzQxMjEwNmIzNTM0YTcyOWYyNzE3ZGU2ZmYzNjIyNi5zZXRDb250ZW50KGh0bWxfZmFmYTI5ZThkZGRiNDNiY2IxNjI5ZDFhZDJjZTcwOTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDAwZmU3YzhiYzczNDQ5YmE1ZTZiODc2ZGE4Zjc0YmMuYmluZFBvcHVwKHBvcHVwX2FjNDEyMTA2YjM1MzRhNzI5ZjI3MTdkZTZmZjM2MjI2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM3OTk2ZjU4MWY4OTRhZGZiYzFlODZlYzNkMDkxYjgzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTI2ODg1OTksMjIuODc3ODgwMTAwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NkNDNmNDAzYWJkODRkNGI4NzYxZmViYmY5NjdmYjE3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ3Njk1OGY2NDZmYTQ2MTBhNWMzNGUyMDA5YTZiOWVjID0gJCgnPGRpdiBpZD0iaHRtbF80NzY5NThmNjQ2ZmE0NjEwYTVjMzRlMjAwOWE2YjllYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDOsc+BzrHOu86vzrEgzpHPg86vzr3Ot8+CIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2Q0M2Y0MDNhYmQ4NGQ0Yjg3NjFmZWJiZjk2N2ZiMTcuc2V0Q29udGVudChodG1sXzQ3Njk1OGY2NDZmYTQ2MTBhNWMzNGUyMDA5YTZiOWVjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM3OTk2ZjU4MWY4OTRhZGZiYzFlODZlYzNkMDkxYjgzLmJpbmRQb3B1cChwb3B1cF9jZDQzZjQwM2FiZDg0ZDRiODc2MWZlYmJmOTY3ZmIxNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYzU0MDQzZTdjNDc0YTA3YmU4Zjc1NmZiMDU1YWI1YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU3NzQxNTQ3LDIyLjg4OTg4MzA0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM0ZGYzY2UiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjNGRmM2NlIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc1Njk4ZTdhMGE1MjQ0Mjc4M2E4YTM3NzAxYTUyODUxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FiMjQwZDAzZmYzZjRjZWQ4YmFjMDE2ZDJmYTgwODU1ID0gJCgnPGRpdiBpZD0iaHRtbF9hYjI0MGQwM2ZmM2Y0Y2VkOGJhYzAxNmQyZmE4MDg1NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpHPg8+Az4HPjM6yz4HPhc+DzrcgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NTY5OGU3YTBhNTI0NDI3ODNhOGEzNzcwMWE1Mjg1MS5zZXRDb250ZW50KGh0bWxfYWIyNDBkMDNmZjNmNGNlZDhiYWMwMTZkMmZhODA4NTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGM1NDA0M2U3YzQ3NGEwN2JlOGY3NTZmYjA1NWFiNWMuYmluZFBvcHVwKHBvcHVwXzc1Njk4ZTdhMGE1MjQ0Mjc4M2E4YTM3NzAxYTUyODUxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2JlMjdjN2E4YTY0NTQ1YzhhMDBlM2Q0YmViMDcwMWRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTgyNjcyMTIsMjIuODY0NTYyOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGE0MGE0MGM2MmNiNDAyNTgyMGZjNmI3ZThkMjNkNWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmMyNmU1ZjNiNmYyNGQxZWJhYWRiYTFmOWRkZDU3YTEgPSAkKCc8ZGl2IGlkPSJodG1sX2ZjMjZlNWYzYjZmMjRkMWViYWFkYmExZjlkZGQ1N2ExIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OoM6xzrvOt86/z4fPjs+BzrEgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wYTQwYTQwYzYyY2I0MDI1ODIwZmM2YjdlOGQyM2Q1Yi5zZXRDb250ZW50KGh0bWxfZmMyNmU1ZjNiNmYyNGQxZWJhYWRiYTFmOWRkZDU3YTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmUyN2M3YThhNjQ1NDVjOGEwMGUzZDRiZWIwNzAxZGIuYmluZFBvcHVwKHBvcHVwXzBhNDBhNDBjNjJjYjQwMjU4MjBmYzZiN2U4ZDIzZDViKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhlODRmOGE1NGY0YTQzYmRhOGEzMjk3ZDI3MWMxYmMxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTk4NDQyMDgsMjIuODcyNzc3OTRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzk5MDA4ODQxM2YxNDAyY2ExMjBlYmU2ZWEzNDA3ZWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjU3Mjk2N2I1NTJhNGJiNGE1YjY1YmUzNTc1YjcyYzIgPSAkKCc8ZGl2IGlkPSJodG1sXzY1NzI5NjdiNTUyYTRiYjRhNWI2NWJlMzU3NWI3MmMyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OoM+Bzr/Phs6uz4TOt8+CIM6XzrvOr86xz4IgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83OTkwMDg4NDEzZjE0MDJjYTEyMGViZTZlYTM0MDdlZi5zZXRDb250ZW50KGh0bWxfNjU3Mjk2N2I1NTJhNGJiNGE1YjY1YmUzNTc1YjcyYzIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGU4NGY4YTU0ZjRhNDNiZGE4YTMyOTdkMjcxYzFiYzEuYmluZFBvcHVwKHBvcHVwXzc5OTAwODg0MTNmMTQwMmNhMTIwZWJlNmVhMzQwN2VmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E0MmIxZmM2YWY1MTQ3M2FiMDZkNDFkYzNkNzJlNTg5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTkxMDk4NzksMjIuODQxMzg0ODldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODgzOTEwM2VkMWY5NGEwODllMmVmYTdmYTM5ZWE2MDEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTI3OGVkNGU3MjQ4NGZmYmJiN2E2NGYxYjZhZjJjZjUgPSAkKCc8ZGl2IGlkPSJodG1sXzUyNzhlZDRlNzI0ODRmZmJiYjdhNjRmMWI2YWYyY2Y1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OoM6xzr3Osc6zzq/OsSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg4MzkxMDNlZDFmOTRhMDg5ZTJlZmE3ZmEzOWVhNjAxLnNldENvbnRlbnQoaHRtbF81Mjc4ZWQ0ZTcyNDg0ZmZiYmI3YTY0ZjFiNmFmMmNmNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNDJiMWZjNmFmNTE0NzNhYjA2ZDQxZGMzZDcyZTU4OS5iaW5kUG9wdXAocG9wdXBfODgzOTEwM2VkMWY5NGEwODllMmVmYTdmYTM5ZWE2MDEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGZkNTQyY2VhYzc1NDk3NGFlMTg4ZmE3ODUxMDM5OTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MjgzODM2NCwyMi44MjE4NzY1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjNGRmM2NlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzRkZjNjZSIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lNjgwYTk4MmU0MmU0MzRiYjI2NGE4Y2E1OTViN2I5NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kNjczMGZmNmRjN2E0NTVkODQyZDM1YWE4YThjZDRjYiA9ICQoJzxkaXYgaWQ9Imh0bWxfZDY3MzBmZjZkYzdhNDU1ZDg0MmQzNWFhOGE4Y2Q0Y2IiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gzrHOvc6xz4HOr8+EzrfPgiBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U2ODBhOTgyZTQyZTQzNGJiMjY0YThjYTU5NWI3Yjk3LnNldENvbnRlbnQoaHRtbF9kNjczMGZmNmRjN2E0NTVkODQyZDM1YWE4YThjZDRjYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kZmQ1NDJjZWFjNzU0OTc0YWUxODhmYTc4NTEwMzk5OS5iaW5kUG9wdXAocG9wdXBfZTY4MGE5ODJlNDJlNDM0YmIyNjRhOGNhNTk1YjdiOTcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzY0MjM3NjZlNjU2NDcwOGFiMGVlNjhmZDEzYzljZjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MTc1Njg5NywyMi43OTc1NTk3NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmY5NjRmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmOTY0ZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kYjgyOThjYmZmNWI0ZjZmOWFmOWI3MDA3MWEwOThhZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81NTQxNjk2MTVkN2E0YTY0YjBjMmE2NGNiOTIxMmFlZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNTU0MTY5NjE1ZDdhNGE2NGIwYzJhNjRjYjkyMTJhZWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Rz4HOs86/zrvOuc66z4zOvSBDbHVzdGVyIDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RiODI5OGNiZmY1YjRmNmY5YWY5YjcwMDcxYTA5OGFlLnNldENvbnRlbnQoaHRtbF81NTQxNjk2MTVkN2E0YTY0YjBjMmE2NGNiOTIxMmFlZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zNjQyMzc2NmU2NTY0NzA4YWIwZWU2OGZkMTNjOWNmMS5iaW5kUG9wdXAocG9wdXBfZGI4Mjk4Y2JmZjViNGY2ZjlhZjliNzAwNzFhMDk4YWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2E1YmU5MzNlYjYxNDc4Mzg2YTlhNTk0ZTUxMDU4MDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41OTQ1ODU0MiwyMi43OTkyODU4OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjNGRmM2NlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzRkZjNjZSIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wYTZiMTgxYTI4NjU0ZDI4OGUxY2M2NmU3ODgxNWE4NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zZjlkM2JlNmY0MmI0YWNhYmYwMzZlZDZkYWRlNTRmZiA9ICQoJzxkaXYgaWQ9Imh0bWxfM2Y5ZDNiZTZmNDJiNGFjYWJmMDM2ZWQ2ZGFkZTU0ZmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6kzq/Pgc+Fzr3PgiBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBhNmIxODFhMjg2NTRkMjg4ZTFjYzY2ZTc4ODE1YTg3LnNldENvbnRlbnQoaHRtbF8zZjlkM2JlNmY0MmI0YWNhYmYwMzZlZDZkYWRlNTRmZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zYTViZTkzM2ViNjE0NzgzODZhOWE1OTRlNTEwNTgwMy5iaW5kUG9wdXAocG9wdXBfMGE2YjE4MWEyODY1NGQyODhlMWNjNjZlNzg4MTVhODcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTNhYjVlOTgzOTY0NGI3YWE4ZjlmOWY3NzEzNTFkNzYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41ODYyMDQ1MywyMi44MDgxOTUxMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjNGRmM2NlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzRkZjNjZSIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lZThhNzllZTIyNjg0ZTY5YmViNzhjNjYwNTBkMWIzOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iYTIzZmQzYWI0MjY0OTQzYWI5NTNiNzE4OTk5YzZhMCA9ICQoJzxkaXYgaWQ9Imh0bWxfYmEyM2ZkM2FiNDI2NDk0M2FiOTUzYjcxODk5OWM2YTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6azrHPgM6/zrTOr8+Dz4TPgc65zrHPgiBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VlOGE3OWVlMjI2ODRlNjliZWI3OGM2NjA1MGQxYjM5LnNldENvbnRlbnQoaHRtbF9iYTIzZmQzYWI0MjY0OTQzYWI5NTNiNzE4OTk5YzZhMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lM2FiNWU5ODM5NjQ0YjdhYThmOWY5Zjc3MTM1MWQ3Ni5iaW5kUG9wdXAocG9wdXBfZWU4YTc5ZWUyMjY4NGU2OWJlYjc4YzY2MDUwZDFiMzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWI2YmU4OGVlMGFkNDhhZGI1MTE4OWEwZmI3MDE4YWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41MTc5NTU3OCwyMi44NTcxNDMzOTk5OTk5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTU3ODU0YTU2MmMzNDc2NjhiMTUyMDhmZThmN2M4MDAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTZhY2ZmM2M4M2UxNDNmYzk2YTEzYTRmOWQ4YTc1NjIgPSAkKCc8ZGl2IGlkPSJodG1sXzU2YWNmZjNjODNlMTQzZmM5NmExM2E0ZjlkOGE3NTYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OpM6/zrvPjM69IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTU3ODU0YTU2MmMzNDc2NjhiMTUyMDhmZThmN2M4MDAuc2V0Q29udGVudChodG1sXzU2YWNmZjNjODNlMTQzZmM5NmExM2E0ZjlkOGE3NTYyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ViNmJlODhlZTBhZDQ4YWRiNTExODlhMGZiNzAxOGFhLmJpbmRQb3B1cChwb3B1cF85NTc4NTRhNTYyYzM0NzY2OGIxNTIwOGZlOGY3YzgwMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNmNmZjQ1OTM3NzE0NGY0OTYxZjlhNTE0NWQzMDA4OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYwNjc2MTkzLDIyLjk3OTI3Mjg0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ViODc4ZTAyNzVkMDQyODY4YzQ3NTAyZTIyOGI5YTFhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ3ZTczYzNiN2RhMzRiZmY5ZGMwYWJiMjA3NzA3YTY2ID0gJCgnPGRpdiBpZD0iaHRtbF80N2U3M2MzYjdkYTM0YmZmOWRjMGFiYjIwNzcwN2E2NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqfOv8+Fz4TOsc67zrHOr865zrrOsSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ViODc4ZTAyNzVkMDQyODY4YzQ3NTAyZTIyOGI5YTFhLnNldENvbnRlbnQoaHRtbF80N2U3M2MzYjdkYTM0YmZmOWRjMGFiYjIwNzcwN2E2Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iNmNmZjQ1OTM3NzE0NGY0OTYxZjlhNTE0NWQzMDA4OC5iaW5kUG9wdXAocG9wdXBfZWI4NzhlMDI3NWQwNDI4NjhjNDc1MDJlMjI4YjlhMWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmUwNWFkMDJhYTAxNDVhM2I4NDcxNmY3YjMwNjA0ZGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41OTY0NTg0NCwyMi45NzI3MjMwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMWNiMWM5MTQ3ZDk0YTc0OTJjMGMyOTMyMDFjMTEwMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zOGI4NDY3MWYxNDY0YTllOWRhODA1MDg4MDhiZjYxZSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzhiODQ2NzFmMTQ2NGE5ZTlkYTgwNTA4ODA4YmY2MWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6TzrnOsc69zr3Ov8+FzrvOsc6vzrnOus6xIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDFjYjFjOTE0N2Q5NGE3NDkyYzBjMjkzMjAxYzExMDIuc2V0Q29udGVudChodG1sXzM4Yjg0NjcxZjE0NjRhOWU5ZGE4MDUwODgwOGJmNjFlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZlMDVhZDAyYWEwMTQ1YTNiODQ3MTZmN2IzMDYwNGRmLmJpbmRQb3B1cChwb3B1cF9kMWNiMWM5MTQ3ZDk0YTc0OTJjMGMyOTMyMDFjMTEwMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zMzZlNTcwNTJhMWQ0MTY5OTk4ZWVkNjY1NDRjZDlkMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYxNDc3NjYxLDIzLjAxNjM5NTU3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM0ZGYzY2UiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjNGRmM2NlIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg0YWVjYzhlMzhlNzRiNDZhMTkwNjAwNWEyODQxMmMwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FmYjAzOTJkMDNkZjQ3NTk5ZWZmNDgwNGUyMzc4Nzc0ID0gJCgnPGRpdiBpZD0iaHRtbF9hZmIwMzkyZDAzZGY0NzU5OWVmZjQ4MDRlMjM3ODc3NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqPPgM63zrvOtc6vzrEgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84NGFlY2M4ZTM4ZTc0YjQ2YTE5MDYwMDVhMjg0MTJjMC5zZXRDb250ZW50KGh0bWxfYWZiMDM5MmQwM2RmNDc1OTllZmY0ODA0ZTIzNzg3NzQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzM2ZTU3MDUyYTFkNDE2OTk5OGVlZDY2NTQ0Y2Q5ZDEuYmluZFBvcHVwKHBvcHVwXzg0YWVjYzhlMzhlNzRiNDZhMTkwNjAwNWEyODQxMmMwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U3ZDEwY2U4ZDNmMTQ1YzFiYjMwMzc3MzQ3ZmFiNmNmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTk3MjIxMzcsMjMuMDA0NTQzM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjNGRmM2NlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzRkZjNjZSIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zZTBmODZhMzM3OGI0OWZiYWRjZTM4NTIxYmE5Y2Y3NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83ZmE0ZjY1MTUxNzg0MGJlOGJlODZlNTE4OWI4YTMzZiA9ICQoJzxkaXYgaWQ9Imh0bWxfN2ZhNGY2NTE1MTc4NDBiZThiZTg2ZTUxODliOGEzM2YiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6nzqzOvc65IM6czrXPgc66zr/Pjc+BzrcgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zZTBmODZhMzM3OGI0OWZiYWRjZTM4NTIxYmE5Y2Y3Ny5zZXRDb250ZW50KGh0bWxfN2ZhNGY2NTE1MTc4NDBiZThiZTg2ZTUxODliOGEzM2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTdkMTBjZThkM2YxNDVjMWJiMzAzNzczNDdmYWI2Y2YuYmluZFBvcHVwKHBvcHVwXzNlMGY4NmEzMzc4YjQ5ZmJhZGNlMzg1MjFiYTljZjc3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U1YmFhZmY1YzFkNjQyNTdiZmIwYTk5YWE3ZWE3NmNjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTk0NjU3ODk5OTk5OTk0LDIzLjA3NjI5MDEzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBjMmNiODA1N2U4NDQ3MGNiNmUwZGQ2ZjNmY2JkZjBhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk0ZTY0ZWIyMjliMjQ1YjM4MjVlYjQ5OGE4ZjBhMTI3ID0gJCgnPGRpdiBpZD0iaHRtbF85NGU2NGViMjI5YjI0NWIzODI1ZWI0OThhOGYwYTEyNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpHPg866zrvOt8+AzrnOtc6vzr8gzpXPgM65zrTOsc+Nz4HOv8+FIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGMyY2I4MDU3ZTg0NDcwY2I2ZTBkZDZmM2ZjYmRmMGEuc2V0Q29udGVudChodG1sXzk0ZTY0ZWIyMjliMjQ1YjM4MjVlYjQ5OGE4ZjBhMTI3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U1YmFhZmY1YzFkNjQyNTdiZmIwYTk5YWE3ZWE3NmNjLmJpbmRQb3B1cChwb3B1cF8wYzJjYjgwNTdlODQ0NzBjYjZlMGRkNmYzZmNiZGYwYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYzYzOTk0Mjk2YzQ0NTViYTc2YzU4NmZiM2FiODUyNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjYyODY5NjQ0LDIzLjA4ODg5Mzg5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NkY2U5YzQ0ZTQxZDQ3MjE4N2JjNDExYzJhYzQ2YmIxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRiNzBlNGVkNjQyMjQ0MGE4NDAxOGNmYTM3YjAxZmI3ID0gJCgnPGRpdiBpZD0iaHRtbF80YjcwZTRlZDY0MjI0NDBhODQwMThjZmEzN2IwMWZiNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbOs865zr/PgiDOkc69zrTPgc6tzrHPgiBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NkY2U5YzQ0ZTQxZDQ3MjE4N2JjNDExYzJhYzQ2YmIxLnNldENvbnRlbnQoaHRtbF80YjcwZTRlZDY0MjI0NDBhODQwMThjZmEzN2IwMWZiNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zYzYzOTk0Mjk2YzQ0NTViYTc2YzU4NmZiM2FiODUyNy5iaW5kUG9wdXAocG9wdXBfY2RjZTljNDRlNDFkNDcyMTg3YmM0MTFjMmFjNDZiYjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2FkZDlhMGNkNjY3NDA5NDk0ZGE0OGMxYjAxNDBlZDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42Mjc0MDcwNywyMy4xMzQ0NTA5MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xY2M2MDk2MDgyM2Q0YzZkODlhNGRlMTEwN2E1MDkyNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NzFjNTJiMGIwMjE0YzU0OTlmNTEyMzBkMWRlMTEyNCA9ICQoJzxkaXYgaWQ9Imh0bWxfODcxYzUyYjBiMDIxNGM1NDk5ZjUxMjMwZDFkZTExMjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gzrHOvc+Mz4HOsc68zrEgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xY2M2MDk2MDgyM2Q0YzZkODlhNGRlMTEwN2E1MDkyNi5zZXRDb250ZW50KGh0bWxfODcxYzUyYjBiMDIxNGM1NDk5ZjUxMjMwZDFkZTExMjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2FkZDlhMGNkNjY3NDA5NDk0ZGE0OGMxYjAxNDBlZDEuYmluZFBvcHVwKHBvcHVwXzFjYzYwOTYwODIzZDRjNmQ4OWE0ZGUxMTA3YTUwOTI2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VjODEzZDkzOGY0MTQ0NWFiYTU2ZDI4NzJiNTM1M2M0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjMxOTY5NDUsMjMuMTI0MzIyODldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGE5YjU3ZTZjNzBkNDlmZTg5MGRmYzcwOGEzNmY4ZGIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTQ4ZWY0YzU1NjZiNDg2MzllZTU1ZTU4MjA3NDRhZDkgPSAkKCc8ZGl2IGlkPSJodG1sXzE0OGVmNGM1NTY2YjQ4NjM5ZWU1NWU1ODIwNzQ0YWQ5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Olc+AzqzOvc+JIM6Vz4DOr860zrHPhc+Bzr/PgiBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRhOWI1N2U2YzcwZDQ5ZmU4OTBkZmM3MDhhMzZmOGRiLnNldENvbnRlbnQoaHRtbF8xNDhlZjRjNTU2NmI0ODYzOWVlNTVlNTgyMDc0NGFkOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYzgxM2Q5MzhmNDE0NDVhYmE1NmQyODcyYjUzNTNjNC5iaW5kUG9wdXAocG9wdXBfNGE5YjU3ZTZjNzBkNDlmZTg5MGRmYzcwOGEzNmY4ZGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTVjNTRhZjYzMjRmNGVkMmE0ZjdiNDZlOTY0NTU5Y2IgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42MTEwMDM4OCwyMy4xNjA5NzI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzFiYjdiZWRlZjJkNDQwMGRiZmY4ZDIwMTMyYTY1ZmExID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzljOTQwYzVmNjQwZjQyOGE5OTkyMzIwYWQ2MzQxNjg1ID0gJCgnPGRpdiBpZD0iaHRtbF85Yzk0MGM1ZjY0MGY0MjhhOTk5MjMyMGFkNjM0MTY4NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDOsc69zrHOs86vzrEgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xYmI3YmVkZWYyZDQ0MDBkYmZmOGQyMDEzMmE2NWZhMS5zZXRDb250ZW50KGh0bWxfOWM5NDBjNWY2NDBmNDI4YTk5OTIzMjBhZDYzNDE2ODUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTVjNTRhZjYzMjRmNGVkMmE0ZjdiNDZlOTY0NTU5Y2IuYmluZFBvcHVwKHBvcHVwXzFiYjdiZWRlZjJkNDQwMGRiZmY4ZDIwMTMyYTY1ZmExKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQzNDExZTM4NjQzYzRlNWQ5N2E4YThlNTkwZjczMTdiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTY3ODQ4MjEsMjMuMTQ0MzE1NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTJmY2E2NDY1ZmIwNDRmYjg4NDYzN2VkZWE0M2UxNmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDk0MWZiNWFiNDM1NGE3ZTg3NjRiYWQ0MThkMjA4MjMgPSAkKCc8ZGl2IGlkPSJodG1sX2Q5NDFmYjVhYjQzNTRhN2U4NzY0YmFkNDE4ZDIwODIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Olc6+zr/Ph86uIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTJmY2E2NDY1ZmIwNDRmYjg4NDYzN2VkZWE0M2UxNmIuc2V0Q29udGVudChodG1sX2Q5NDFmYjVhYjQzNTRhN2U4NzY0YmFkNDE4ZDIwODIzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQzNDExZTM4NjQzYzRlNWQ5N2E4YThlNTkwZjczMTdiLmJpbmRQb3B1cChwb3B1cF81MmZjYTY0NjVmYjA0NGZiODg0NjM3ZWRlYTQzZTE2Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMmQ3NGUwOWI0MzU0ZjY4YWQ3ODRlODg5NWJkMGY3ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU5MDc4MjE3LDIzLjE1NzUwMzEzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y4OTExMjk0YTMzMDQwMDY5MTgwOTY1OTNlMDMyYjI0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MxYWE3YTE0ZjlmMTQ4MGI4M2YxYmMxZjg0NDc5YzQ5ID0gJCgnPGRpdiBpZD0iaHRtbF9jMWFhN2ExNGY5ZjE0ODBiODNmMWJjMWY4NDQ3OWM0OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOv867zrnOrM66zrnOv869IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjg5MTEyOTRhMzMwNDAwNjkxODA5NjU5M2UwMzJiMjQuc2V0Q29udGVudChodG1sX2MxYWE3YTE0ZjlmMTQ4MGI4M2YxYmMxZjg0NDc5YzQ5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2MyZDc0ZTA5YjQzNTRmNjhhZDc4NGU4ODk1YmQwZjdlLmJpbmRQb3B1cChwb3B1cF9mODkxMTI5NGEzMzA0MDA2OTE4MDk2NTkzZTAzMmIyNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yNjM3YTQ5ZWU2OTM0NzU1YjBkZjgzYjY5NTI0N2Y5YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU2OTk4NDQ0LDIzLjA3MTIyODAzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNmNGE4NTc5NjQzMjRjYzQ4MmE3NjBkMTFiZmMxMmZlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBhZmZjMTE3MWZjZDQzMjFhZjdjZTcxNGVkM2NiZTczID0gJCgnPGRpdiBpZD0iaHRtbF8wYWZmYzExNzFmY2Q0MzIxYWY3Y2U3MTRlZDNjYmU3MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpHOtM6szrzOuc6/zr0gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zZjRhODU3OTY0MzI0Y2M0ODJhNzYwZDExYmZjMTJmZS5zZXRDb250ZW50KGh0bWxfMGFmZmMxMTcxZmNkNDMyMWFmN2NlNzE0ZWQzY2JlNzMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjYzN2E0OWVlNjkzNDc1NWIwZGY4M2I2OTUyNDdmOWIuYmluZFBvcHVwKHBvcHVwXzNmNGE4NTc5NjQzMjRjYzQ4MmE3NjBkMTFiZmMxMmZlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MyZjNmNWJiOTRlYjQ4ZDY4OGM3ZmM0MGJhMDIzNGI0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTg5MzA1ODgsMjMuMTA1NDQwMTRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzExM2VlMzc5YTI4NDcyN2JiZjdlYmU0YWNjZTNhNzIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWJjYWIyNDI5NzcxNDY5ZDkyNDc3ZWE1Nzc1NzllZjYgPSAkKCc8ZGl2IGlkPSJodG1sXzliY2FiMjQyOTc3MTQ2OWQ5MjQ3N2VhNTc3NTc5ZWY2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OlM63zrzOv8+DzrnOrCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMxMTNlZTM3OWEyODQ3MjdiYmY3ZWJlNGFjY2UzYTcyLnNldENvbnRlbnQoaHRtbF85YmNhYjI0Mjk3NzE0NjlkOTI0NzdlYTU3NzU3OWVmNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMmYzZjViYjk0ZWI0OGQ2ODhjN2ZjNDBiYTAyMzRiNC5iaW5kUG9wdXAocG9wdXBfMzExM2VlMzc5YTI4NDcyN2JiZjdlYmU0YWNjZTNhNzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzRjZmNkY2IzMjJlNGE0MGFjY2EwMDdlNjcxODhkM2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NDUzMTQ3OSwyMi45OTA3MDM1OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMTk5NmYzIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzE5OTZmMyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNzllYTM0NjgxYjI0NDMwYjJhNmY5NjNmYjAxNmZiMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMWZkZWE0MWJjZTQ0ZjA4OGI4NTdkYThhMjEyMmE0MSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDFmZGVhNDFiY2U0NGYwODhiODU3ZGE4YTIxMjJhNDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPsK2zrPOuc6/z4Igzp3Ouc66z4zOu86xzr/PgiBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM3OWVhMzQ2ODFiMjQ0MzBiMmE2Zjk2M2ZiMDE2ZmIzLnNldENvbnRlbnQoaHRtbF8wMWZkZWE0MWJjZTQ0ZjA4OGI4NTdkYThhMjEyMmE0MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83NGNmY2RjYjMyMmU0YTQwYWNjYTAwN2U2NzE4OGQzZS5iaW5kUG9wdXAocG9wdXBfMzc5ZWEzNDY4MWIyNDQzMGIyYTZmOTYzZmIwMTZmYjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTQ0YTJjMDEwOWM5NGJhNjlhNTY1ZWJhOTJjOTI5ZTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41MjcyNTIyMDAwMDAwMSwyMi45NTk1MzU2MDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWQwZDY0NmQ0YWIxNGM2MzhlMTM2NTg0N2IxYmUzYWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDJkN2NjMTMzYTM1NDUwMDljZWE4MWM1N2QzNDY2ZmEgPSAkKCc8ZGl2IGlkPSJodG1sXzAyZDdjYzEzM2EzNTQ1MDA5Y2VhODFjNTdkMzQ2NmZhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6szr3PhM65zrEgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZDBkNjQ2ZDRhYjE0YzYzOGUxMzY1ODQ3YjFiZTNhYS5zZXRDb250ZW50KGh0bWxfMDJkN2NjMTMzYTM1NDUwMDljZWE4MWM1N2QzNDY2ZmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTQ0YTJjMDEwOWM5NGJhNjlhNTY1ZWJhOTJjOTI5ZTIuYmluZFBvcHVwKHBvcHVwXzVkMGQ2NDZkNGFiMTRjNjM4ZTEzNjU4NDdiMWJlM2FhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Q3YzdlOGU3Y2YwOTQ3NWU4ZmU5ZmFhZWMzNTUzMDdhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDExMTM2NjMsMjMuMzMyMTY0NzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWExYjM2MmZiM2E2NGMzZTgxOWY2M2NjYmVkMjA2ZGYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGE0MjkxY2FjZmViNDdiNmE3NWJjZDYyZTBlNzY3YmEgPSAkKCc8ZGl2IGlkPSJodG1sXzBhNDI5MWNhY2ZlYjQ3YjZhNzViY2Q2MmUwZTc2N2JhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc66z4TOriDOjs60z4HOsc+CIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWExYjM2MmZiM2E2NGMzZTgxOWY2M2NjYmVkMjA2ZGYuc2V0Q29udGVudChodG1sXzBhNDI5MWNhY2ZlYjQ3YjZhNzViY2Q2MmUwZTc2N2JhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q3YzdlOGU3Y2YwOTQ3NWU4ZmU5ZmFhZWMzNTUzMDdhLmJpbmRQb3B1cChwb3B1cF9lYTFiMzYyZmIzYTY0YzNlODE5ZjYzY2NiZWQyMDZkZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMzljYjkzYmIzZTA0MDY0OTc5ZWQ1NmMzNzc2ZjZmYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQxMjQ3NTU5LDIzLjM0NjUwNDIxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQyODRkOTM0Zjc0ZTQ2ZmY5NGUxODMxN2JjYzgxMjM2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q3Mzc1OWFhMDM1NzQ2OTliZWRhMjc1ZjkyMmVhOWU3ID0gJCgnPGRpdiBpZD0iaHRtbF9kNzM3NTlhYTAzNTc0Njk5YmVkYTI3NWY5MjJlYTllNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpHOs86vzrEgzpHOuc66zrHPhM61z4HOr869zrcgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80Mjg0ZDkzNGY3NGU0NmZmOTRlMTgzMTdiY2M4MTIzNi5zZXRDb250ZW50KGh0bWxfZDczNzU5YWEwMzU3NDY5OWJlZGEyNzVmOTIyZWE5ZTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTM5Y2I5M2JiM2UwNDA2NDk3OWVkNTZjMzc3NmY2ZmIuYmluZFBvcHVwKHBvcHVwXzQyODRkOTM0Zjc0ZTQ2ZmY5NGUxODMxN2JjYzgxMjM2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYxOWVhYmZkNTU1NTQyNGY4NjcyZmEyNjYwMjQ2ZTAxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDExODMwOSwyMy4zNTQxODcwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNmQ3MWIyMTFiZTU0OWNhYjA2NDU4MGQyMGE1Yjk0YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84OTc0MDVmNDc3NTc0MDZjOWNmYmFlYWI3ZWVmMjdhYyA9ICQoJzxkaXYgaWQ9Imh0bWxfODk3NDA1ZjQ3NzU3NDA2YzljZmJhZWFiN2VlZjI3YWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gzrvOrc+AzrnOv869IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjZkNzFiMjExYmU1NDljYWIwNjQ1ODBkMjBhNWI5NGMuc2V0Q29udGVudChodG1sXzg5NzQwNWY0Nzc1NzQwNmM5Y2ZiYWVhYjdlZWYyN2FjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYxOWVhYmZkNTU1NTQyNGY4NjcyZmEyNjYwMjQ2ZTAxLmJpbmRQb3B1cChwb3B1cF9mNmQ3MWIyMTFiZTU0OWNhYjA2NDU4MGQyMGE1Yjk0Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iMDg0NmQ2M2ZhYzc0YzYzYTU4MjU2Y2U1MDBjOTkwMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQxMDI2Njg4LDIzLjM3MjQ5NzU2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU5YjUwMzM3OTBmNDRkMTFhOGQyNGY3YjI5ZGYwMGM5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NjYTU1MmU5MDdmNDQxOTk5YmQzMzQ2ODhhZDhlYzNkID0gJCgnPGRpdiBpZD0iaHRtbF9jY2E1NTJlOTA3ZjQ0MTk5OWJkMzM0Njg4YWQ4ZWMzZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqDOt86zzqzOtM65zrEgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81OWI1MDMzNzkwZjQ0ZDExYThkMjRmN2IyOWRmMDBjOS5zZXRDb250ZW50KGh0bWxfY2NhNTUyZTkwN2Y0NDE5OTliZDMzNDY4OGFkOGVjM2QpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjA4NDZkNjNmYWM3NGM2M2E1ODI1NmNlNTAwYzk5MDEuYmluZFBvcHVwKHBvcHVwXzU5YjUwMzM3OTBmNDRkMTFhOGQyNGY3YjI5ZGYwMGM5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NmOTY5NGZiNzE1ODRhYWNiOTRhNjUxNDNiMjMyYjhlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDA5ODUxMDcsMjMuMzgwNTM3MDNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDYyYmY4MjVkMjk1NDE4N2FhOTIyMzE2NzQyMDhhMzcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzRkOGM2MDQyY2I4NGY3Njg2ZDY1NjU2MDM3NmVkYTQgPSAkKCc8ZGl2IGlkPSJodG1sX2M0ZDhjNjA0MmNiODRmNzY4NmQ2NTY1NjAzNzZlZGE0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oo8+JzrvOt869zqzPgc65zr/OvSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ2MmJmODI1ZDI5NTQxODdhYTkyMjMxNjc0MjA4YTM3LnNldENvbnRlbnQoaHRtbF9jNGQ4YzYwNDJjYjg0Zjc2ODZkNjU2NTYwMzc2ZWRhNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jZjk2OTRmYjcxNTg0YWFjYjk0YTY1MTQzYjIzMmI4ZS5iaW5kUG9wdXAocG9wdXBfNDYyYmY4MjVkMjk1NDE4N2FhOTIyMzE2NzQyMDhhMzcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDE5N2Y2YzA0YWQwNDdkYWI2YjI3MjIyZTg1YzMzM2IgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40MTM2MDA5MiwyMy4zOTcyOTY5MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMWQyODFkMGMyYmU0NjY4YjY2OTQyYWI2ODZiMDM2MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84M2Q2N2Y4MWEwZWI0MzAwOWM0ZDhlMDJiNTQ1YjhjZiA9ICQoJzxkaXYgaWQ9Imh0bWxfODNkNjdmODFhMGViNDMwMDljNGQ4ZTAyYjU0NWI4Y2YiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6czrXPhM+Mz4fOuc6/zr0gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMWQyODFkMGMyYmU0NjY4YjY2OTQyYWI2ODZiMDM2Mi5zZXRDb250ZW50KGh0bWxfODNkNjdmODFhMGViNDMwMDljNGQ4ZTAyYjU0NWI4Y2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDE5N2Y2YzA0YWQwNDdkYWI2YjI3MjIyZTg1YzMzM2IuYmluZFBvcHVwKHBvcHVwX2YxZDI4MWQwYzJiZTQ2NjhiNjY5NDJhYjY4NmIwMzYyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVmYzkzMGFhOWQyYzRmNTQ4NDA0ZDJjNjg2YWJjYjJkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNDAyMDUzODMsMjMuMjgwNzI3MzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzE5OTZmMyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMxOTk2ZjMiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGFmMjBmMWEyOWIzNGYxOWJmYzY0ZDkwNDRkNDE3NjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzBkYjY0YzE1YmQyNGZlMmIyNTc3ZTUxOTU5ZTdlNjEgPSAkKCc8ZGl2IGlkPSJodG1sXzMwZGI2NGMxNWJkMjRmZTJiMjU3N2U1MTk1OWU3ZTYxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc+HzrvOsc60zq/PhM+DzrEgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wYWYyMGYxYTI5YjM0ZjE5YmZjNjRkOTA0NGQ0MTc2NC5zZXRDb250ZW50KGh0bWxfMzBkYjY0YzE1YmQyNGZlMmIyNTc3ZTUxOTU5ZTdlNjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWZjOTMwYWE5ZDJjNGY1NDg0MDRkMmM2ODZhYmNiMmQuYmluZFBvcHVwKHBvcHVwXzBhZjIwZjFhMjliMzRmMTliZmM2NGQ5MDQ0ZDQxNzY0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzgyZTQ3ODMzZjNlZTRlMmM5ODdmOGU3YTRjNDhhNjM1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuMzUxNjA4MjgsMjMuMjQwNjI5MTk5OTk5OTk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMxOTk2ZjMiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMTk5NmYzIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RhZDA3NWFmMWU4YTQzODE5MDk4OGIxNmUzZDM2ZDk2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhmYjU1ZmNmZDBlNTQ1YWFhZWVmYTU1NWVjMzQ2MTIxID0gJCgnPGRpdiBpZD0iaHRtbF84ZmI1NWZjZmQwZTU0NWFhYWVlZmE1NTVlYzM0NjEyMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbOs865zr/OuSDOkc69zqzPgc6zz4XPgc6/zrkgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kYWQwNzVhZjFlOGE0MzgxOTA5ODhiMTZlM2QzNmQ5Ni5zZXRDb250ZW50KGh0bWxfOGZiNTVmY2ZkMGU1NDVhYWFlZWZhNTU1ZWMzNDYxMjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODJlNDc4MzNmM2VlNGUyYzk4N2Y4ZTdhNGM0OGE2MzUuYmluZFBvcHVwKHBvcHVwX2RhZDA3NWFmMWU4YTQzODE5MDk4OGIxNmUzZDM2ZDk2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZlMTc2OWQ2ZDE1YzQzZTQ4OWM3NzFlNjNmYjA5ZWQzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuMzM5NzA2NDIsMjMuMTkyNTU2MzhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzE5OTZmMyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMxOTk2ZjMiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2QzZDQ0MjBlZjFjNDhiMGJiZWNkYmJjZTU1ZDMxZDEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzM5MjgyNmVlZjcyNDcyN2JiZmYzNTk1MGE5YzI5ZDcgPSAkKCc8ZGl2IGlkPSJodG1sX2MzOTI4MjZlZWY3MjQ3MjdiYmZmMzU5NTBhOWMyOWQ3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OoM61z4TPgc6/zrjOrM67zrHPg8+DzrEgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zZDNkNDQyMGVmMWM0OGIwYmJlY2RiYmNlNTVkMzFkMS5zZXRDb250ZW50KGh0bWxfYzM5MjgyNmVlZjcyNDcyN2JiZmYzNTk1MGE5YzI5ZDcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmUxNzY5ZDZkMTVjNDNlNDg5Yzc3MWU2M2ZiMDllZDMuYmluZFBvcHVwKHBvcHVwXzNkM2Q0NDIwZWYxYzQ4YjBiYmVjZGJiY2U1NWQzMWQxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZlYWViZTE5MzZhNTQyNGVhYjRjMGQ4OTA3ODlkNzgyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuMzAzMjI2NDcsMjMuMTk1NDQ5ODNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzE5OTZmMyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMxOTk2ZjMiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDYzNGFiYzE5YTYyNGZhZGFlYmZiYTM1Y2U5YjUzZDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTViY2YxY2VkYTlhNGMxNWFkNDhmNDU3MzMzOGY0M2YgPSAkKCc8ZGl2IGlkPSJodG1sX2E1YmNmMWNlZGE5YTRjMTVhZDQ4ZjQ1NzMzMzhmNDNmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6/z4XOvc6/z43PgM65IENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDYzNGFiYzE5YTYyNGZhZGFlYmZiYTM1Y2U5YjUzZDMuc2V0Q29udGVudChodG1sX2E1YmNmMWNlZGE5YTRjMTVhZDQ4ZjQ1NzMzMzhmNDNmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZlYWViZTE5MzZhNTQyNGVhYjRjMGQ4OTA3ODlkNzgyLmJpbmRQb3B1cChwb3B1cF80NjM0YWJjMTlhNjI0ZmFkYWViZmJhMzVjZTliNTNkMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NjkzNDA2MmQ1ZTA0OWExYWUxMzI2MDc4NzU1ZTBiMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjM1MTQwNjEsMjMuMDkyNzczNDRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzE5OTZmMyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMxOTk2ZjMiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDVhZDkwNDMyNGRhNGVkZWIxMDgxZTRhMTBiZmRjZmEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWM3MzgyNjUxNWYyNDgxNmIwYTBmMjY0ZTU3NDM5NWEgPSAkKCc8ZGl2IGlkPSJodG1sXzFjNzM4MjY1MTVmMjQ4MTZiMGEwZjI2NGU1NzQzOTVhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Cts6zzrnOv8+CIM6dzrnOus+MzrvOsc6/z4IgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNWFkOTA0MzI0ZGE0ZWRlYjEwODFlNGExMGJmZGNmYS5zZXRDb250ZW50KGh0bWxfMWM3MzgyNjUxNWYyNDgxNmIwYTBmMjY0ZTU3NDM5NWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzY5MzQwNjJkNWUwNDlhMWFlMTMyNjA3ODc1NWUwYjEuYmluZFBvcHVwKHBvcHVwXzA1YWQ5MDQzMjRkYTRlZGViMTA4MWU0YTEwYmZkY2ZhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzExZTBkMjY0ODU0ODRkMTViZjk5OGNlYWQ4ZGQwYTdiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuMzg0MzQ2MDEsMjMuMDgxNjg5ODNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzE5OTZmMyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMxOTk2ZjMiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWU0MDkyZGI3Mjc1NGU5YmI1Nzk5YTQ1MmIwYWU5Y2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTgwMmIwMmQxMDkwNGEyZGI1NzBjNmMyNTdmMDkzYTIgPSAkKCc8ZGl2IGlkPSJodG1sXzE4MDJiMDJkMTA5MDRhMmRiNTcwYzZjMjU3ZjA5M2EyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OmM+Fzr3OryBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVlNDA5MmRiNzI3NTRlOWJiNTc5OWE0NTJiMGFlOWNiLnNldENvbnRlbnQoaHRtbF8xODAyYjAyZDEwOTA0YTJkYjU3MGM2YzI1N2YwOTNhMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMWUwZDI2NDg1NDg0ZDE1YmY5OThjZWFkOGRkMGE3Yi5iaW5kUG9wdXAocG9wdXBfNWU0MDkyZGI3Mjc1NGU5YmI1Nzk5YTQ1MmIwYWU5Y2IpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDRmYjgyZTBkYjM5NDUzNzkzYmY2YTgwNjhjYmVkMTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4zODY4MjE3NSwyMy4wOTg3MzAwOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMTk5NmYzIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzE5OTZmMyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80YTg5ODY2NmNjZGM0ZTVlYmZmNjZjYWZhMGU0OWMxNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZDJmMGE1ODAxM2Y0ZDljOWE1MDJhYzA0YTc4NDU3ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfZmQyZjBhNTgwMTNmNGQ5YzlhNTAyYWMwNGE3ODQ1N2UiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6bzqzOus66zrXPgiBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRhODk4NjY2Y2NkYzRlNWViZmY2NmNhZmEwZTQ5YzE0LnNldENvbnRlbnQoaHRtbF9mZDJmMGE1ODAxM2Y0ZDljOWE1MDJhYzA0YTc4NDU3ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NGZiODJlMGRiMzk0NTM3OTNiZjZhODA2OGNiZWQxOS5iaW5kUG9wdXAocG9wdXBfNGE4OTg2NjZjY2RjNGU1ZWJmZjY2Y2FmYTBlNDljMTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDFjODg5YmQ3YTgxNDczY2I1OGQ1NjcxOTQ3MTNhYWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4zOTMzNjAxNCwyMy4xMDc5MDYzNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMTk5NmYzIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzE5OTZmMyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85NmZiYzk5MDQzNmM0NDA5OThiYjY3YmIwNGJlYTU3NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YWYyMTZjZWU0NGE0YWRjYjFlMDkzNzI4NTk4NzMyYiA9ICQoJzxkaXYgaWQ9Imh0bWxfN2FmMjE2Y2VlNDRhNGFkY2IxZTA5MzcyODU5ODczMmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Uzr/Pgc6/z43Phs65IENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTZmYmM5OTA0MzZjNDQwOTk4YmI2N2JiMDRiZWE1NzYuc2V0Q29udGVudChodG1sXzdhZjIxNmNlZTQ0YTRhZGNiMWUwOTM3Mjg1OTg3MzJiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QxYzg4OWJkN2E4MTQ3M2NiNThkNTY3MTk0NzEzYWFjLmJpbmRQb3B1cChwb3B1cF85NmZiYzk5MDQzNmM0NDA5OThiYjY3YmIwNGJlYTU3Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83YjM2MTgyY2U1ZGE0YWJmYWMyMzc4MTkwZDdmYzljYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQwOTY5NDY3LDIzLjE0OTc1OTI5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY4NWU1MzNlN2U1MzQ3ZGY5Mzk3MDJlYjYxMmU4ZTI4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdkOGM0MWQxODAxZTRiOWM4OTAxOTkyNTc0NjFhNWZlID0gJCgnPGRpdiBpZD0iaHRtbF83ZDhjNDFkMTgwMWU0YjljODkwMTk5MjU3NDYxYTVmZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOrM68z4DOv8+CIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjg1ZTUzM2U3ZTUzNDdkZjkzOTcwMmViNjEyZThlMjguc2V0Q29udGVudChodG1sXzdkOGM0MWQxODAxZTRiOWM4OTAxOTkyNTc0NjFhNWZlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdiMzYxODJjZTVkYTRhYmZhYzIzNzgxOTBkN2ZjOWNiLmJpbmRQb3B1cChwb3B1cF82ODVlNTMzZTdlNTM0N2RmOTM5NzAyZWI2MTJlOGUyOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mMGQ1ODU2MjZkY2Q0ZTViYWU0N2RjZTA3Njc5ZDNmZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQxODkxODYxLDIzLjE1MTE2MzFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmOTY0ZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjk2NGYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmUxZGVhMjhiMzc1NGUxYTg2ODVjMTNkMzE0MzRmMDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGEzNzIyOTMxMDA4NDVhMGJkNjBhM2NlMjc5YzM3OTIgPSAkKCc8ZGl2IGlkPSJodG1sXzBhMzcyMjkzMTAwODQ1YTBiZDYwYTNjZTI3OWMzNzkyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oks67zrHPh86/z4DOv8+FzrvOrc65zrrOsSBDbHVzdGVyIDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZlMWRlYTI4YjM3NTRlMWE4Njg1YzEzZDMxNDM0ZjAzLnNldENvbnRlbnQoaHRtbF8wYTM3MjI5MzEwMDg0NWEwYmQ2MGEzY2UyNzljMzc5Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMGQ1ODU2MjZkY2Q0ZTViYWU0N2RjZTA3Njc5ZDNmZC5iaW5kUG9wdXAocG9wdXBfNmUxZGVhMjhiMzc1NGUxYTg2ODVjMTNkMzE0MzRmMDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODM5Y2I3MTZjODc0NDk0ZmIwODRkMDRiNzQ2MmJhYWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40MDY4NTY1NCwyMy4xMzc1MjE3NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZmQ5ZDJlNjQ2YTI0NTU3ODg3N2Y0YzJkOWM5YmUwNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lMTk0YjAwMjRkZDc0MjgxYjgzZjM3ODA3ZTJiMGYzMSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTE5NGIwMDI0ZGQ3NDI4MWI4M2YzNzgwN2UyYjBmMzEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6azqzOvM+Azr/PgiBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzlmZDlkMmU2NDZhMjQ1NTc4ODc3ZjRjMmQ5YzliZTA3LnNldENvbnRlbnQoaHRtbF9lMTk0YjAwMjRkZDc0MjgxYjgzZjM3ODA3ZTJiMGYzMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MzljYjcxNmM4NzQ0OTRmYjA4NGQwNGI3NDYyYmFhZi5iaW5kUG9wdXAocG9wdXBfOWZkOWQyZTY0NmEyNDU1Nzg4NzdmNGMyZDljOWJlMDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWYzZGFhNWU4NTk4NDE4NmJkODlmODk4YWI5YTZjMGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4zOTQ3NDg2OSwyMy4xMDc2MTI2MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMTk5NmYzIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzE5OTZmMyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYWZiYWJhMjgzZjA0NzExOGYxZjhjYzRiMzExMzdiNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZDAzNWJkNWFlZjU0Y2IzYWMwNmQzYjAzNGYzM2NjMiA9ICQoJzxkaXYgaWQ9Imh0bWxfYmQwMzViZDVhZWY1NGNiM2FjMDZkM2IwMzRmMzNjYzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6Uzr/Pgc6/z43Phs65IENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2FmYmFiYTI4M2YwNDcxMThmMWY4Y2M0YjMxMTM3Yjcuc2V0Q29udGVudChodG1sX2JkMDM1YmQ1YWVmNTRjYjNhYzA2ZDNiMDM0ZjMzY2MyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFmM2RhYTVlODU5ODQxODZiZDg5Zjg5OGFiOWE2YzBmLmJpbmRQb3B1cChwb3B1cF9jYWZiYWJhMjgzZjA0NzExOGYxZjhjYzRiMzExMzdiNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NDZiNDQxZjY0Yzk0NmNiYTdmM2E5ZDNjY2Q3MDQ0NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQyMjEzNDQsMjMuMTE4MDIxMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjVmNWY0NjIyNWYwNDU5NGIyNWJlOTBmNDU5NzM3ZTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzYxOWY1MzUyNGRjNGZiMWFiMjI3NTA3OTM3ZGQzY2QgPSAkKCc8ZGl2IGlkPSJodG1sX2M2MTlmNTM1MjRkYzRmYjFhYjIyNzUwNzkzN2RkM2NkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms6/z4HPic69zq/PgiBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY1ZjVmNDYyMjVmMDQ1OTRiMjViZTkwZjQ1OTczN2UxLnNldENvbnRlbnQoaHRtbF9jNjE5ZjUzNTI0ZGM0ZmIxYWIyMjc1MDc5MzdkZDNjZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83NDZiNDQxZjY0Yzk0NmNiYTdmM2E5ZDNjY2Q3MDQ0NS5iaW5kUG9wdXAocG9wdXBfNjVmNWY0NjIyNWYwNDU5NGIyNWJlOTBmNDU5NzM3ZTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmIxZjcwZTFmZDEwNGFlZDgyNTJiNWY3NTZkZjVlOGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40MjU4MzQ2NiwyMy4xMzM2Mjg4NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMWZhOTQyZmY4ZDk0MGY2ODUzM2IxNjJhZWNiNDljMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NzIyMzdiMjY3NWY0NDUyYWMxZDAxZDhjOWJiOTI5MiA9ICQoJzxkaXYgaWQ9Imh0bWxfOTcyMjM3YjI2NzVmNDQ1MmFjMWQwMWQ4YzliYjkyOTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gzrHPgc6xzrvOr86xIM6mzr/Pjc+Bzr3Pic69IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjFmYTk0MmZmOGQ5NDBmNjg1MzNiMTYyYWVjYjQ5YzEuc2V0Q29udGVudChodG1sXzk3MjIzN2IyNjc1ZjQ0NTJhYzFkMDFkOGM5YmI5MjkyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZiMWY3MGUxZmQxMDRhZWQ4MjUyYjVmNzU2ZGY1ZThiLmJpbmRQb3B1cChwb3B1cF9mMWZhOTQyZmY4ZDk0MGY2ODUzM2IxNjJhZWNiNDljMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kOTZkZjU4MTE4NmM0OTk0YWJmNmQ5NDlhNGE5MmVmOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjMzMjEyMjc5OTk5OTk5LDIzLjEzMjU0NTQ3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUwMGNhMDQxMGZiMTRkOGJhZDc3NTUzMmU0ZDQ0YjgwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg0MjMxNGU4Njg0YjRiNjM4YjZhOTk0MWMzNGI4NTE0ID0gJCgnPGRpdiBpZD0iaHRtbF84NDIzMTRlODY4NGI0YjYzOGI2YTk5NDFjMzRiODUxNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpLOtc+BzrLOtc+Bzr/Pjc60zrEgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MDBjYTA0MTBmYjE0ZDhiYWQ3NzU1MzJlNGQ0NGI4MC5zZXRDb250ZW50KGh0bWxfODQyMzE0ZTg2ODRiNGI2MzhiNmE5OTQxYzM0Yjg1MTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDk2ZGY1ODExODZjNDk5NGFiZjZkOTQ5YTRhOTJlZjkuYmluZFBvcHVwKHBvcHVwXzUwMGNhMDQxMGZiMTRkOGJhZDc3NTUzMmU0ZDQ0YjgwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk0YmZiZGY5ZDk1YTQ5NGQ4ODc5OTI3MmJjNzdmNzI4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuMzA0MzIxMjksMjMuMTQyNDY1NTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzU5YTkzMDQwZjBjNDA0ZThmNDQ4ZGNkN2NjZWJjYzcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGRjNmE4NDJhNTM2NGU1ODkwODRiZWMzMjM1YmM4MzEgPSAkKCc8ZGl2IGlkPSJodG1sX2RkYzZhODQyYTUzNjRlNTg5MDg0YmVjMzIzNWJjODMxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Op865zr3Or8+Ez4POsSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M1OWE5MzA0MGYwYzQwNGU4ZjQ0OGRjZDdjY2ViY2M3LnNldENvbnRlbnQoaHRtbF9kZGM2YTg0MmE1MzY0ZTU4OTA4NGJlYzMyMzViYzgzMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NGJmYmRmOWQ5NWE0OTRkODg3OTkyNzJiYzc3ZjcyOC5iaW5kUG9wdXAocG9wdXBfYzU5YTkzMDQwZjBjNDA0ZThmNDQ4ZGNkN2NjZWJjYzcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWNjMTcyMTQ2NGFkNDIyNGE5MmQ3MmZmMDFjNzI5NGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4yOTM5Nzk2NCwyMy4xNjExMzg1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82NTU5NmE2YzQ0YjU0NmM0OWQ0ZjIwZTU2YTQ1MGYwMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NTVmMDU1M2U2MjQ0YzVmYjJjNzhhNDM3ZTAxODRiMCA9ICQoJzxkaXYgaWQ9Imh0bWxfODU1ZjA1NTNlNjI0NGM1ZmIyYzc4YTQzN2UwMTg0YjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6az4zPg8+EzrEgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82NTU5NmE2YzQ0YjU0NmM0OWQ0ZjIwZTU2YTQ1MGYwMC5zZXRDb250ZW50KGh0bWxfODU1ZjA1NTNlNjI0NGM1ZmIyYzc4YTQzN2UwMTg0YjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWNjMTcyMTQ2NGFkNDIyNGE5MmQ3MmZmMDFjNzI5NGYuYmluZFBvcHVwKHBvcHVwXzY1NTk2YTZjNDRiNTQ2YzQ5ZDRmMjBlNTZhNDUwZjAwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFlNjFiZTE5Mjg0ODQ1YTRhMzRlNmMzODAwZjI0MjVlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuMjkzMjYyNDgsMjMuMTg5NDQ1NDk5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMxOTk2ZjMiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMTk5NmYzIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FlNjE4NjE1Y2VkMzQ5ODdiNmIyMzczMDY5MDEwOGZmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg0YTYxMjI1ODYwNTRjNDBiM2RlNzU3ZDFmMzk2ZGU3ID0gJCgnPGRpdiBpZD0iaHRtbF84NGE2MTIyNTg2MDU0YzQwYjNkZTc1N2QxZjM5NmRlNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbOs865zr/PgiDOkc65zrzOuc67zrnOsc69z4zPgiBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FlNjE4NjE1Y2VkMzQ5ODdiNmIyMzczMDY5MDEwOGZmLnNldENvbnRlbnQoaHRtbF84NGE2MTIyNTg2MDU0YzQwYjNkZTc1N2QxZjM5NmRlNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xZTYxYmUxOTI4NDg0NWE0YTM0ZTZjMzgwMGYyNDI1ZS5iaW5kUG9wdXAocG9wdXBfYWU2MTg2MTVjZWQzNDk4N2I2YjIzNzMwNjkwMTA4ZmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2M4YWFhYjcyNTdkNDI5ZjhjMjk1M2EzMTM2YjdiMmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy40NTQyNTQxNSwyMy4yMzQzMzMwNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjNGRmM2NlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzRkZjNjZSIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMDk1NDZjODNiZDA0NDA1OTRiNjk3YTU0ZjBhYzdmMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMDQxNDg5MTI5Njc0Mzc1YmEyNmZhNTkyZTFjYWRiYSA9ICQoJzxkaXYgaWQ9Imh0bWxfZjA0MTQ4OTEyOTY3NDM3NWJhMjZmYTU5MmUxY2FkYmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6bzr/Phc66zrHOkM+EzrnOv869IENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjA5NTQ2YzgzYmQwNDQwNTk0YjY5N2E1NGYwYWM3ZjAuc2V0Q29udGVudChodG1sX2YwNDE0ODkxMjk2NzQzNzViYTI2ZmE1OTJlMWNhZGJhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNjOGFhYWI3MjU3ZDQyOWY4YzI5NTNhMzEzNmI3YjJkLmJpbmRQb3B1cChwb3B1cF9iMDk1NDZjODNiZDA0NDA1OTRiNjk3YTU0ZjBhYzdmMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84OGUyMTBjY2NhZmU0NzMyYTliNmNjNzM2ZThlYTRlYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUxNTA1MjgsMjMuMjA0NjEwODJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmFiNTIyYWM4YTgxNDBmNTkwNWU3NTFlOGMzMWRiYTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWE1ODJiODMwYTMyNGFlNDlkY2I0MTJmNTRlOWQ0YjcgPSAkKCc8ZGl2IGlkPSJodG1sXzVhNTgyYjgzMGEzMjRhZTQ5ZGNiNDEyZjU0ZTlkNGI3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Ooc6szrTOv869IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmFiNTIyYWM4YTgxNDBmNTkwNWU3NTFlOGMzMWRiYTEuc2V0Q29udGVudChodG1sXzVhNTgyYjgzMGEzMjRhZTQ5ZGNiNDEyZjU0ZTlkNGI3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg4ZTIxMGNjY2FmZTQ3MzJhOWI2Y2M3MzZlOGVhNGVjLmJpbmRQb3B1cChwb3B1cF82YWI1MjJhYzhhODE0MGY1OTA1ZTc1MWU4YzMxZGJhMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yMTk3YjM3YjNkY2Y0ODBhODY4ZDk4YjNhMDg1NDc2ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQ0NzcxOTU3LDIzLjEyNDEwNzM2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q2MzFjZTQ2MjE4ODRkYTJhODFmZDNhYTZlODAxODc2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhlNjA1YjMzZmQwYzRhM2RiOTM3YzU5ZDY4MjA1ZDM4ID0gJCgnPGRpdiBpZD0iaHRtbF84ZTYwNWIzM2ZkMGM0YTNkYjkzN2M1OWQ2ODIwNWQzOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqPOsc67zqzOvc+EzrnOv869IENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDYzMWNlNDYyMTg4NGRhMmE4MWZkM2FhNmU4MDE4NzYuc2V0Q29udGVudChodG1sXzhlNjA1YjMzZmQwYzRhM2RiOTM3YzU5ZDY4MjA1ZDM4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIxOTdiMzdiM2RjZjQ4MGE4NjhkOThiM2EwODU0NzZkLmJpbmRQb3B1cChwb3B1cF9kNjMxY2U0NjIxODg0ZGEyYTgxZmQzYWE2ZTgwMTg3Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZDA3ODdlNjVjNjI0NWVhOTc0MjMyZDIxYzU1YjlkZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjQ1MzIwMTI5LDIzLjEwMDg4MTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMxOTk2ZjMiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMTk5NmYzIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE2YTAzZGY1ZDU1ZTRkYWFhMGZmYjRhNTQxOThjMzRlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M2YTM5MWRmMjlkZjRlNzE5ZjkwMmFjYWU2YWVlZDM4ID0gJCgnPGRpdiBpZD0iaHRtbF9jNmEzOTFkZjI5ZGY0ZTcxOWY5MDJhY2FlNmFlZWQzOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbOs865zr/PgiDOmc+JzqzOvc69zrfPgiBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE2YTAzZGY1ZDU1ZTRkYWFhMGZmYjRhNTQxOThjMzRlLnNldENvbnRlbnQoaHRtbF9jNmEzOTFkZjI5ZGY0ZTcxOWY5MDJhY2FlNmFlZWQzOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hZDA3ODdlNjVjNjI0NWVhOTc0MjMyZDIxYzU1YjlkZi5iaW5kUG9wdXAocG9wdXBfMTZhMDNkZjVkNTVlNGRhYWEwZmZiNGE1NDE5OGMzNGUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2U0ZmEzYjU1ZjVlNDNkOGI2OWIwZjM4MDI4NzYwZjMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41NjcwNzAwMSwyMi43ODk1MDExOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYTU2NTc4YzhjNmM0MjNhYThmM2MwYWQwYmI5MjNlZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yMDc3ZmM0ODA0NDM0YzBhOGZlYzBjZTA3ZTZlNTQxMCA9ICQoJzxkaXYgaWQ9Imh0bWxfMjA3N2ZjNDgwNDQzNGMwYThmZWMwY2UwN2U2ZTU0MTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6cz4DOv8+Nz4HPhM62zrnOv869IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2E1NjU3OGM4YzZjNDIzYWE4ZjNjMGFkMGJiOTIzZWUuc2V0Q29udGVudChodG1sXzIwNzdmYzQ4MDQ0MzRjMGE4ZmVjMGNlMDdlNmU1NDEwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNlNGZhM2I1NWY1ZTQzZDhiNjliMGYzODAyODc2MGYzLmJpbmRQb3B1cChwb3B1cF9jYTU2NTc4YzhjNmM0MjNhYThmM2MwYWQwYmI5MjNlZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yOTE2NWNhOGM3NjQ0ZmVhYmVlOWQzODkyYzA1MjMwYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU3NTA1MDM1LDIyLjcyOTQ1NDA0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ViMmNhZmRlMzA1YzRjNzhiNDVhZGMxMmU5NGUxMjE0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y4MTI5YzRjOWM2NjRiODc4ZjY4N2FmYzVlZjIxNzg5ID0gJCgnPGRpdiBpZD0iaHRtbF9mODEyOWM0YzljNjY0Yjg3OGY2ODdhZmM1ZWYyMTc4OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zqTOt868zq3Ovc65zr/OvSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ViMmNhZmRlMzA1YzRjNzhiNDVhZGMxMmU5NGUxMjE0LnNldENvbnRlbnQoaHRtbF9mODEyOWM0YzljNjY0Yjg3OGY2ODdhZmM1ZWYyMTc4OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yOTE2NWNhOGM3NjQ0ZmVhYmVlOWQzODkyYzA1MjMwYi5iaW5kUG9wdXAocG9wdXBfZWIyY2FmZGUzMDVjNGM3OGI0NWFkYzEyZTk0ZTEyMTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjgyNDM0NDdmNjllNDhmZTgyYTgyYTI0NjRkYzVhY2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy42NjU4NDM5NiwyMi43MDQ5NTQxNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hN2M5YzdiZWM2OTU0NzczOTU0NGU0ZDI4ODM5YTI5ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMzVmMzdkZTNmYTk0YmExYmM2NDQ5ODVlZWE0YjNlMiA9ICQoJzxkaXYgaWQ9Imh0bWxfZjM1ZjM3ZGUzZmE5NGJhMWJjNjQ0OTg1ZWVhNGIzZTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6RzrPOr86xIM6RzrnOus6xz4TOtc+Bzq/Ovc63IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTdjOWM3YmVjNjk1NDc3Mzk1NDRlNGQyODgzOWEyOWQuc2V0Q29udGVudChodG1sX2YzNWYzN2RlM2ZhOTRiYTFiYzY0NDk4NWVlYTRiM2UyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI4MjQzNDQ3ZjY5ZTQ4ZmU4MmE4MmEyNDY0ZGM1YWNlLmJpbmRQb3B1cChwb3B1cF9hN2M5YzdiZWM2OTU0NzczOTU0NGU0ZDI4ODM5YTI5ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85Y2M3M2JlYThmZWU0MDNjOTEzOTg3MjUwZDQ1MDc2OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjY3OTEzMDU1LDIyLjY4NTEzNjgwMDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYjNjM2VkZDQzZDg0OWViODQwMTkzY2JmMmNjMjgyMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yMTMwMzc1NWI3N2U0NDZjYTY3MWJhOGFlZjUzYWUwZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMjEzMDM3NTViNzdlNDQ2Y2E2NzFiYThhZWY1M2FlMGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6gzrHOvc+Mz4HOsc68zrEgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xYjNjM2VkZDQzZDg0OWViODQwMTkzY2JmMmNjMjgyMC5zZXRDb250ZW50KGh0bWxfMjEzMDM3NTViNzdlNDQ2Y2E2NzFiYThhZWY1M2FlMGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWNjNzNiZWE4ZmVlNDAzYzkxMzk4NzI1MGQ0NTA3NjkuYmluZFBvcHVwKHBvcHVwXzFiM2MzZWRkNDNkODQ5ZWI4NDAxOTNjYmYyY2MyODIwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2JmMDNlZjM0NTEzOTQwOWQ4NDE5MGY2NmViMmY5MzFlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNjIwNTE3NzMsMjIuNTk2MTU3MDddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzRkZjNjZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM0ZGYzY2UiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjdlY2UyZThkODhhNGM2Yjg0ZDBjMThhMTc5MzRiN2EgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTU4Zjc2YWYxNTkxNDhiOTg0N2I5ZTljMmQ1NmEwOWYgPSAkKCc8ZGl2IGlkPSJodG1sX2U1OGY3NmFmMTU5MTQ4Yjk4NDdiOWU5YzJkNTZhMDlmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc6zz4HOuc67zq/PhM+DzrEgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82N2VjZTJlOGQ4OGE0YzZiODRkMGMxOGExNzkzNGI3YS5zZXRDb250ZW50KGh0bWxfZTU4Zjc2YWYxNTkxNDhiOTg0N2I5ZTljMmQ1NmEwOWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmYwM2VmMzQ1MTM5NDA5ZDg0MTkwZjY2ZWIyZjkzMWUuYmluZFBvcHVwKHBvcHVwXzY3ZWNlMmU4ZDg4YTRjNmI4NGQwYzE4YTE3OTM0YjdhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Q3OGZkZDYxODRkNzQ2YzRhNjAxOTFkMjRjZTVmMTk4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTczODMzNDcsMjIuNTY0NDA5MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2IyZjM5NiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNiMmYzOTYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDc3ZTg3ZWFhMTJjNDFkMGEzZTEyMzI3NjcyYjQ5YzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzAxZTI5NWY5Yjc4NDc1Njg2YzBkZTVhMDZkM2JkODEgPSAkKCc8ZGl2IGlkPSJodG1sXzMwMWUyOTVmOWI3ODQ3NTY4NmMwZGU1YTA2ZDNiZDgxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oms+Bz43OsSDOks+Bz43Pg863IENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDc3ZTg3ZWFhMTJjNDFkMGEzZTEyMzI3NjcyYjQ5YzMuc2V0Q29udGVudChodG1sXzMwMWUyOTVmOWI3ODQ3NTY4NmMwZGU1YTA2ZDNiZDgxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q3OGZkZDYxODRkNzQ2YzRhNjAxOTFkMjRjZTVmMTk4LmJpbmRQb3B1cChwb3B1cF9kNzdlODdlYWExMmM0MWQwYTNlMTIzMjc2NzJiNDljMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNzU5ZTVmNGEwNmY0NmQwYTRjZjRkZmM4NjM1YTAyYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUxNzM5NTAyLDIyLjY5Mzg4NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2JkY2Q0OTcwMDI0NGM5NWE1Y2E0YzFkZjQ1ZDIxNjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDJhMmZmMTJiMjRkNGU0OGJkOGEyNjM2NjVkMWMwOWEgPSAkKCc8ZGl2IGlkPSJodG1sXzQyYTJmZjEyYjI0ZDRlNDhiZDhhMjYzNjY1ZDFjMDlhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Oo8+AzrfOu865z4nPhM6szrrOt8+CIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2JkY2Q0OTcwMDI0NGM5NWE1Y2E0YzFkZjQ1ZDIxNjEuc2V0Q29udGVudChodG1sXzQyYTJmZjEyYjI0ZDRlNDhiZDhhMjYzNjY1ZDFjMDlhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U3NTllNWY0YTA2ZjQ2ZDBhNGNmNGRmYzg2MzVhMDJjLmJpbmRQb3B1cChwb3B1cF8zYmRjZDQ5NzAwMjQ0Yzk1YTVjYTRjMWRmNDVkMjE2MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MGIwMTUzYmFiNDM0NDI4OGI0NGIwNDc2NWRlM2MyYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUzMTA1NTQ1LDIyLjcxMDUzMTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM0ZGYzY2UiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjNGRmM2NlIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE0YWVkM2VlMDJmZjRkODk4ZjY1ZGVmNDI2OTBmYWQyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ4NTYxNGE1N2YwMDQ3OGFiYTZjZDBkM2MyNDJkMDgzID0gJCgnPGRpdiBpZD0iaHRtbF80ODU2MTRhNTdmMDA0NzhhYmE2Y2QwZDNjMjQyZDA4MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOv8+FzrPOsc6vzrnOus6xIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTRhZWQzZWUwMmZmNGQ4OThmNjVkZWY0MjY5MGZhZDIuc2V0Q29udGVudChodG1sXzQ4NTYxNGE1N2YwMDQ3OGFiYTZjZDBkM2MyNDJkMDgzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYwYjAxNTNiYWI0MzQ0Mjg4YjQ0YjA0NzY1ZGUzYzJhLmJpbmRQb3B1cChwb3B1cF8xNGFlZDNlZTAyZmY0ZDg5OGY2NWRlZjQyNjkwZmFkMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MzVmNTkwMjc0MDA0NDJhOTc1ZTY2MGM4NDk3ZDIwNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjU1MDE3MDksMjIuNzE1NzM2MzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmOTY0ZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjk2NGYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjNiZTFiZjQyOGZkNDNiOGFlZWIyNzI4Yzg3YTM4MzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjRiZTUxMmZhZjkyNDE1OWFjNWJkYTYwMDRmMDc1NTEgPSAkKCc8ZGl2IGlkPSJodG1sX2Y0YmU1MTJmYWY5MjQxNTlhYzViZGE2MDA0ZjA3NTUxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7OnM+NzrvOv865IENsdXN0ZXIgNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjNiZTFiZjQyOGZkNDNiOGFlZWIyNzI4Yzg3YTM4MzMuc2V0Q29udGVudChodG1sX2Y0YmU1MTJmYWY5MjQxNTlhYzViZGE2MDA0ZjA3NTUxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYzNWY1OTAyNzQwMDQ0MmE5NzVlNjYwYzg0OTdkMjA0LmJpbmRQb3B1cChwb3B1cF8yM2JlMWJmNDI4ZmQ0M2I4YWVlYjI3MjhjODdhMzgzMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kYmU2ZGFkODI2ZjU0YzVmODBlYmY1MWM5ZjJkYTcwOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3LjUzMDYyNDM5LDIyLjY5MjIxMzA2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JkZjg5NmFhYmMzNDQzNTc5NzljMTVjMjU0NmI0NzdkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RmYzAzMWU1ZDBhYzQzOWY5MTc3YmYyMDY2OTg4YjcyID0gJCgnPGRpdiBpZD0iaHRtbF9kZmMwMzFlNWQwYWM0MzlmOTE3N2JmMjA2Njk4OGI3MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zprOsc67zrHOvM6szrrOuc6/zr0gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZGY4OTZhYWJjMzQ0MzU3OTc5YzE1YzI1NDZiNDc3ZC5zZXRDb250ZW50KGh0bWxfZGZjMDMxZTVkMGFjNDM5ZjkxNzdiZjIwNjY5ODhiNzIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGJlNmRhZDgyNmY1NGM1ZjgwZWJmNTFjOWYyZGE3MDkuYmluZFBvcHVwKHBvcHVwX2JkZjg5NmFhYmMzNDQzNTc5NzljMTVjMjU0NmI0NzdkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzEwMzQzOGQ4YzIzZjQyYjhiZDFkZWI4N2M5MjdhY2RiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTczNDUyLDIyLjY5MzY3OTgxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M4OTc3MWEzYTYzYzRiMTFiZDFmZmQ1MDVmODM0YmExID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VjZTg1MWI5MTM2ZDQ5OWJiYWQ3NzdhYmM0N2VlMTcxID0gJCgnPGRpdiBpZD0iaHRtbF9lY2U4NTFiOTEzNmQ0OTliYmFkNzc3YWJjNDdlZTE3MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpTOuc+HzqzOu865zrEgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jODk3NzFhM2E2M2M0YjExYmQxZmZkNTA1ZjgzNGJhMS5zZXRDb250ZW50KGh0bWxfZWNlODUxYjkxMzZkNDk5YmJhZDc3N2FiYzQ3ZWUxNzEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTAzNDM4ZDhjMjNmNDJiOGJkMWRlYjg3YzkyN2FjZGIuYmluZFBvcHVwKHBvcHVwX2M4OTc3MWEzYTYzYzRiMTFiZDFmZmQ1MDVmODM0YmExKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE3NjI5MmJiZDIwMjQ3Nzg5ZGU5YTdlNzNkMzNkOTEwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzcuNTY1NTI4ODcsMjIuNzE3NTQ0NTZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNGVmMzg3MDExZjIyNDY1MWI1MTVlYTU0Yjc2NWVmNDMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjFlOGUyNzYxZTU3NGM2OWI4N2E1ZmFmMWU4ZmZjM2YgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2Y1YzlkNTM3YzQ3NDhjYTk5MTJjZjdhY2Y1MzZmNTAgPSAkKCc8ZGl2IGlkPSJodG1sXzdmNWM5ZDUzN2M0NzQ4Y2E5OTEyY2Y3YWNmNTM2ZjUwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij7Okc67zrzPhc+Bz4zPgiBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYxZThlMjc2MWU1NzRjNjliODdhNWZhZjFlOGZmYzNmLnNldENvbnRlbnQoaHRtbF83ZjVjOWQ1MzdjNDc0OGNhOTkxMmNmN2FjZjUzNmY1MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNzYyOTJiYmQyMDI0Nzc4OWRlOWE3ZTczZDMzZDkxMC5iaW5kUG9wdXAocG9wdXBfNjFlOGUyNzYxZTU3NGM2OWI4N2E1ZmFmMWU4ZmZjM2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODZjZGE0ZTNhYWQ4NDdkNWJjNWRkZDIzZmNjYzMyZDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy41ODk5NTA1NiwyMi43MDU0NTE5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmY5NjRmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmOTY0ZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYzY2YjllZTIyOTU0N2ZlOWUzMjg2ZDRjMWNhNjM0OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zYmNhY2FiYmQyNDA0MjNhOGY2MDZmYTVkMTc3ZjI5OCA9ICQoJzxkaXYgaWQ9Imh0bWxfM2JjYWNhYmJkMjQwNDIzYThmNjA2ZmE1ZDE3N2YyOTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6czrHOs86/z43Ou86xIENsdXN0ZXIgNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2M2NmI5ZWUyMjk1NDdmZTllMzI4NmQ0YzFjYTYzNDkuc2V0Q29udGVudChodG1sXzNiY2FjYWJiZDI0MDQyM2E4ZjYwNmZhNWQxNzdmMjk4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg2Y2RhNGUzYWFkODQ3ZDViYzVkZGQyM2ZjY2MzMmQxLmJpbmRQb3B1cChwb3B1cF9jYzY2YjllZTIyOTU0N2ZlOWUzMjg2ZDRjMWNhNjM0OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84NTE4MmY5ZjRiZDc0MGY4YTFkNmRiYjg1ZGU5NmYyNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3Ljc2MzMzNjE4LDIyLjQ3NzY2NDk1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMxOTk2ZjMiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMTk5NmYzIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZmOTk5MjA2MjE4MzQ5ZDFhZGFjNjU5MjhiN2ZhNDhjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdkZGIzYmRjMzIyNDQ0ODdiMjIzNGE3ODZjMzAzMjU1ID0gJCgnPGRpdiBpZD0iaHRtbF83ZGRiM2JkYzMyMjQ0NDg3YjIyMzRhNzg2YzMwMzI1NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+wrbOs865zr/PgiDOnc65zrrPjM67zrHOv8+CIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmY5OTkyMDYyMTgzNDlkMWFkYWM2NTkyOGI3ZmE0OGMuc2V0Q29udGVudChodG1sXzdkZGIzYmRjMzIyNDQ0ODdiMjIzNGE3ODZjMzAzMjU1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg1MTgyZjlmNGJkNzQwZjhhMWQ2ZGJiODVkZTk2ZjI1LmJpbmRQb3B1cChwb3B1cF82Zjk5OTIwNjIxODM0OWQxYWRhYzY1OTI4YjdmYTQ4Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hODQ2NzE4YmI3NzQ0YTQ0YTNhNjBlNjBmZDM5M2MxNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM3Ljc0MjU3NjYsMjIuNDc0NDg3MzAwMDAwMDAzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzRlZjM4NzAxMWYyMjQ2NTFiNTE1ZWE1NGI3NjVlZjQzKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U3YWFiYTIxYzljODRjNzY5MTNiZWU1ZjNhMDc3OGJkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRjMzg5MTU3ODJhMjRmZTQ4NGJkMGFjYjZlMTkyZGUxID0gJCgnPGRpdiBpZD0iaHRtbF80YzM4OTE1NzgyYTI0ZmU0ODRiZDBhY2I2ZTE5MmRlMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+zpXOvs6/z4fOriBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U3YWFiYTIxYzljODRjNzY5MTNiZWU1ZjNhMDc3OGJkLnNldENvbnRlbnQoaHRtbF80YzM4OTE1NzgyYTI0ZmU0ODRiZDBhY2I2ZTE5MmRlMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hODQ2NzE4YmI3NzQ0YTQ0YTNhNjBlNjBmZDM5M2MxNi5iaW5kUG9wdXAocG9wdXBfZTdhYWJhMjFjOWM4NGM3NjkxM2JlZTVmM2EwNzc4YmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzg3NDUyMzVmOTJiNDU2ZGIyM2I3MDUxODVlMjMwMDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4yOTkxNzE0NSwyMy4xMzc2MjQ3NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF80ZWYzODcwMTFmMjI0NjUxYjUxNWVhNTRiNzY1ZWY0Myk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81M2E0MDBiY2JhNTI0MTg5ODliZThmN2UyMDZjNWUzOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zZGU4NWJmMjZlYTU0Y2FmOGJkZmZiYzM1YzdmNGE3NCA9ICQoJzxkaXYgaWQ9Imh0bWxfM2RlODViZjI2ZWE1NGNhZjhiZGZmYmMzNWM3ZjRhNzQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPs6nzrfOvc6vz4TPg86xIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTNhNDAwYmNiYTUyNDE4OTg5YmU4ZjdlMjA2YzVlMzguc2V0Q29udGVudChodG1sXzNkZTg1YmYyNmVhNTRjYWY4YmRmZmJjMzVjN2Y0YTc0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM4NzQ1MjM1ZjkyYjQ1NmRiMjNiNzA1MTg1ZTIzMDAzLmJpbmRQb3B1cChwb3B1cF81M2E0MDBiY2JhNTI0MTg5ODliZThmN2UyMDZjNWUzOCk7CgogICAgICAgICAgICAKICAgICAgICAKPC9zY3JpcHQ+" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



##### We can now examine each cluster, thus determining the discriminating venue categories that distinguish the clusters from one another.

##### Cluster 1


```python
cluster1 = argolis_merged.loc[argolis_merged['Cluster Labels'] == 0, argolis_merged.columns[[0] + list(range(5, argolis_merged.shape[1]))]]
cluster1.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8422</th>
      <td>Πορτοχέλιον</td>
      <td>0</td>
      <td>Greek Restaurant</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Bakery</td>
      <td>Ice Cream Shop</td>
    </tr>
    <tr>
      <th>8462</th>
      <td>Κρανίδιον</td>
      <td>0</td>
      <td>Plaza</td>
      <td>Bakery</td>
      <td>Mobile Phone Shop</td>
      <td>Surf Spot</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>8467</th>
      <td>Ερμιόνη</td>
      <td>0</td>
      <td>Café</td>
      <td>Greek Restaurant</td>
      <td>Taverna</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
    </tr>
  </tbody>
</table>
</div>



##### Cluster 2


```python
cluster2 = argolis_merged.loc[argolis_merged['Cluster Labels'] == 1, argolis_merged.columns[[0] + list(range(5, argolis_merged.shape[1]))]]
cluster2.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8606</th>
      <td>Πυργιώτικα</td>
      <td>1</td>
      <td>Taverna</td>
      <td>Hotel</td>
      <td>Greek Restaurant</td>
      <td>Wine Bar</td>
      <td>Food Truck</td>
    </tr>
    <tr>
      <th>8632</th>
      <td>Λυγούριον</td>
      <td>1</td>
      <td>Taverna</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Greek Restaurant</td>
      <td>Wine Bar</td>
    </tr>
    <tr>
      <th>8661</th>
      <td>Καρυά</td>
      <td>1</td>
      <td>Taverna</td>
      <td>Greek Restaurant</td>
      <td>Wine Bar</td>
      <td>French Restaurant</td>
      <td>Deli / Bodega</td>
    </tr>
  </tbody>
</table>
</div>



##### Cluster 3


```python
cluster3 = argolis_merged.loc[argolis_merged['Cluster Labels'] == 2, argolis_merged.columns[[0] + list(range(5, argolis_merged.shape[1]))]]
cluster3.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9607</th>
      <td>¶γιος Νικόλαος</td>
      <td>2</td>
      <td>Beach</td>
      <td>Wine Bar</td>
      <td>French Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>9617</th>
      <td>Αχλαδίτσα</td>
      <td>2</td>
      <td>Beach</td>
      <td>Playground</td>
      <td>Wine Bar</td>
      <td>French Restaurant</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>9618</th>
      <td>¶γιοι Ανάργυροι</td>
      <td>2</td>
      <td>Beach</td>
      <td>Hotel</td>
      <td>Pool</td>
      <td>Wine Bar</td>
      <td>Food Truck</td>
    </tr>
  </tbody>
</table>
</div>



##### Cluster 4


```python
cluster4 = argolis_merged.loc[argolis_merged['Cluster Labels'] == 3, argolis_merged.columns[[0] + list(range(5, argolis_merged.shape[1]))]]
cluster4.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8592</th>
      <td>Λευκάκια</td>
      <td>3</td>
      <td>Soccer Field</td>
      <td>Hotel</td>
      <td>Bar</td>
      <td>Coffee Shop</td>
      <td>French Restaurant</td>
    </tr>
    <tr>
      <th>9560</th>
      <td>Πουλλακίδα</td>
      <td>3</td>
      <td>Farm</td>
      <td>Nature Preserve</td>
      <td>Soccer Field</td>
      <td>Flower Shop</td>
      <td>Wine Bar</td>
    </tr>
    <tr>
      <th>9570</th>
      <td>Ασπρόβρυση</td>
      <td>3</td>
      <td>Hotel</td>
      <td>Wine Bar</td>
      <td>French Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
    </tr>
  </tbody>
</table>
</div>



##### Cluster 5


```python
cluster5 = argolis_merged.loc[argolis_merged['Cluster Labels'] == 4, argolis_merged.columns[[0] + list(range(5, argolis_merged.shape[1]))]]
cluster5.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8741</th>
      <td>Φρουσιούνα</td>
      <td>4</td>
      <td>Mountain</td>
      <td>Ice Cream Shop</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>8811</th>
      <td>Σκοτεινή</td>
      <td>4</td>
      <td>Mountain</td>
      <td>Ice Cream Shop</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>9660</th>
      <td>Κρύα Βρύση</td>
      <td>4</td>
      <td>Mountain</td>
      <td>Ice Cream Shop</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
      <td>Donut Shop</td>
    </tr>
  </tbody>
</table>
</div>



##### Cluster 6


```python
cluster6 = argolis_merged.loc[argolis_merged['Cluster Labels'] == 5, argolis_merged.columns[[0] + list(range(5, argolis_merged.shape[1]))]]
cluster6.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estate</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8619</th>
      <td>Κεφαλάριον</td>
      <td>5</td>
      <td>Greek Restaurant</td>
      <td>Taverna</td>
      <td>Wine Bar</td>
      <td>French Restaurant</td>
      <td>Deli / Bodega</td>
    </tr>
    <tr>
      <th>8622</th>
      <td>¶γιος Αδριανός</td>
      <td>5</td>
      <td>Greek Restaurant</td>
      <td>Wine Bar</td>
      <td>French Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>8649</th>
      <td>Πυργέλλα</td>
      <td>5</td>
      <td>Greek Restaurant</td>
      <td>Wine Bar</td>
      <td>French Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
    </tr>
  </tbody>
</table>
</div>



## Discussion/ Conclusion

##### What is really important here, is that we can examine the five (5) most common venues in each cluster, and conclude that they differ among clusters, thus obtaining intuition and business value. Let us take for example the first two (2) clusters.


```python
print('-----------------1st Cluster-----------------')
print('')
for colname in list(cluster1.columns[2:]):
    print(cluster1[colname].value_counts())
    print('')
    

print('-----------------2nd Cluster-----------------')
print('')    
for colname in list(cluster2.columns[2:]):
    print(cluster2[colname].value_counts())
    print('')
```

    -----------------1st Cluster-----------------
    
    Greek Restaurant           10
    Hotel                       7
    Café                        6
    Bakery                      6
    Resort                      5
    Plaza                       3
    Motel                       3
    Bar                         3
    Big Box Store               3
    Seafood Restaurant          2
    Kafenio                     2
    Beach                       2
    Campground                  2
    Boat or Ferry               2
    Racetrack                   2
    Basketball Court            2
    Vacation Rental             2
    Grilled Meat Restaurant     2
    Betting Shop                1
    Hotel Pool                  1
    Historic Site               1
    Park                        1
    Cocktail Bar                1
    Antique Shop                1
    Souvlaki Shop               1
    Cave                        1
    Neighborhood                1
    Waterfront                  1
    Cafeteria                   1
    Italian Restaurant          1
    BBQ Joint                   1
    Warehouse Store             1
    Liquor Store                1
    Nature Preserve             1
    Scenic Lookout              1
    Name: 1st Most Common Venue, dtype: int64
    
    Wine Bar              11
    Café                   8
    Fish Taverna           5
    Greek Restaurant       5
    Vacation Rental        5
    Bakery                 5
    Beach                  4
    Italian Restaurant     3
    Steakhouse             2
    Bar                    2
    Resort                 2
    Hotel                  2
    Soccer Field           2
    Boat or Ferry          2
    Tunnel                 2
    BBQ Joint              2
    Ouzeri                 1
    Theater                1
    Park                   1
    Basketball Court       1
    Coffee Shop            1
    Playground             1
    Restaurant             1
    Sculpture Garden       1
    Farm                   1
    Bed & Breakfast        1
    Event Space            1
    Grocery Store          1
    Nightclub              1
    Recreation Center      1
    Athletics & Sports     1
    Soccer Stadium         1
    Brewery                1
    Food Truck             1
    Souvlaki Shop          1
    Name: 2nd Most Common Venue, dtype: int64
    
    Beach                      8
    Wine Bar                   8
    French Restaurant          6
    Ice Cream Shop             5
    Coffee Shop                4
    Convenience Store          4
    Bar                        3
    Taverna                    3
    Flower Shop                3
    Bakery                     2
    Spa                        2
    Harbor / Marina            2
    Liquor Store               2
    Campground                 2
    Soccer Field               2
    Greek Restaurant           2
    Seafood Restaurant         2
    Café                       2
    Hotel                      2
    Restaurant                 2
    Nightclub                  1
    Plaza                      1
    Stables                    1
    Kafenio                    1
    Boat or Ferry              1
    Souvlaki Shop              1
    Cave                       1
    Mobile Phone Shop          1
    Park                       1
    Port                       1
    Pizza Place                1
    Monument / Landmark        1
    Theater                    1
    Grilled Meat Restaurant    1
    Fish Taverna               1
    Name: 3rd Most Common Venue, dtype: int64
    
    Deli / Bodega              13
    French Restaurant           7
    Wine Bar                    6
    Beach                       6
    Supermarket                 5
    Coffee Shop                 5
    Greek Restaurant            5
    Café                        3
    Taverna                     3
    Ice Cream Shop              2
    Harbor / Marina             2
    Seafood Restaurant          2
    Bar                         2
    Restaurant                  2
    Hotel Bar                   1
    Bakery                      1
    Hotel                       1
    Resort                      1
    Shoe Store                  1
    Grilled Meat Restaurant     1
    Event Space                 1
    Beach Bar                   1
    Surf Spot                   1
    Food Truck                  1
    Lake                        1
    Vacation Rental             1
    Mountain                    1
    Soccer Field                1
    Convenience Store           1
    Pizza Place                 1
    Movie Theater               1
    Campground                  1
    Name: 4th Most Common Venue, dtype: int64
    
    Dessert Shop          14
    Wine Bar              10
    Deli / Bodega         10
    Taverna                5
    French Restaurant      5
    Food Truck             5
    Café                   4
    Greek Restaurant       4
    Auto Workshop          2
    Bakery                 2
    Plaza                  2
    Fish Taverna           2
    Seafood Restaurant     2
    Convenience Store      2
    Beach                  1
    Snack Place            1
    Basketball Stadium     1
    Bed & Breakfast        1
    Food & Drink Shop      1
    Bistro                 1
    Coffee Shop            1
    Ice Cream Shop         1
    Historic Site          1
    Mountain               1
    Bar                    1
    Supermarket            1
    Name: 5th Most Common Venue, dtype: int64
    
    -----------------2nd Cluster-----------------
    
    Taverna              7
    Bar                  2
    German Restaurant    1
    Name: 1st Most Common Venue, dtype: int64
    
    Wine Bar            2
    Taverna             2
    Soccer Field        1
    Bakery              1
    Beach               1
    Greek Restaurant    1
    Café                1
    Hotel               1
    Name: 2nd Most Common Venue, dtype: int64
    
    Wine Bar             5
    French Restaurant    2
    Greek Restaurant     1
    Bar                  1
    Taverna              1
    Name: 3rd Most Common Venue, dtype: int64
    
    French Restaurant    5
    Deli / Bodega        2
    Greek Restaurant     1
    Food Truck           1
    Wine Bar             1
    Name: 4th Most Common Venue, dtype: int64
    
    Deli / Bodega        5
    Dessert Shop         2
    Convenience Store    1
    Food Truck           1
    Wine Bar             1
    Name: 5th Most Common Venue, dtype: int64
    


##### We can see, for example, that:

##### In the 1st cluster, the 1st most common venue category constitutes of (the first 5 only for simplicity): 10 Greek Restaurants, 7 Hotels, 6 Cafés, 6 Bakeries, 5 Resorts

##### In the 2nd cluster, the 1st most common venue category constitutes of (only 3 exist here): 7 Taverns, 2 Bars, 1 German Restaurant

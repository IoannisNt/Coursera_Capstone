# Week 3 peer graded assignment: Segmenting & clustering neighborhoods in Toronto

### What needs to be done for the purposes of the assignment constitutes of three parts. For simplification purposes, all three parts are going to be accessible through this particular notebook, in order to better track the assignment progress.

#### Part 1

##### At first we are going to import the necessary libraries to conduct the analysis.


```python
import pandas as pd
import numpy as np
import requests
import urllib.request
import time
```


```python
! pip install bs4
#The ! tells the notebook to execute the cell as a shell command.

#Beautiful Soup is a Python library for pulling data out of HTML and XML files. 
#It works with your favorite parser to provide idiomatic ways of navigating, searching, and modifying the parse tree. 
#It commonly saves programmers hours or days of work.
```

    Requirement already satisfied: bs4 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (0.0.1)
    Requirement already satisfied: beautifulsoup4 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from bs4) (4.8.1)
    Requirement already satisfied: soupsieve>=1.2 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from beautifulsoup4->bs4) (1.9.5)



```python
from bs4 import BeautifulSoup
```


```python
url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
response = requests.get(url)
print('The response print: ', response)
print('if the response print: <Response [200]>, this means that it went through!')

soup = BeautifulSoup(response.content, 'html.parser')
```

    The response print:  <Response [200]>
    if the response print: <Response [200]>, this means that it went through!


##### Let's create the table which is going to constitute our dataframe.


```python
table = soup.find('table')
table_headers = table.select('th')
table_rows = table.select('tr')
table_data = table.select('td')

col1 = []
col2 = []
col3 = []
colhead = []

for i in range(0, len(table_data), 3):
    col1.append(table_data[i].text.strip())
    col2.append(table_data[i+1].text.strip())
    col3.append(table_data[i+2].text.strip())
    
for h in table_headers:
    colhead.append(h.text.strip())

df_table = pd.DataFrame(data=[col1, col2, col3]).transpose()
df_table.columns = colhead
print('The dimensions of out dataframe are: ', df_table.shape)
df_table.head()
```

    The dimensions of out dataframe are:  (287, 3)





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
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
  </tbody>
</table>
</div>



##### Change the column names of the newly created dataframe in order for them to match the assignment requirements.


```python
df_table.rename(columns={'Postcode':'PostalCode', 'Neighbourhood' : 'Neighborhood'}, inplace=True)
df_table.head()
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
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Heights</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor</td>
    </tr>
  </tbody>
</table>
</div>



##### Proceed with what it's necessary in order to fulfill the assignment requirements.


```python
print('It appears we come across some missing data...But at what amount?')
print('')
df_table.replace('Not assigned', np.nan, inplace = True)

missing_data = df_table.isnull()

for col in missing_data.columns.values.tolist():
    print(col)
    print(missing_data[col].value_counts())
    print('')
```

    It appears we come across some missing data...But at what amount?
    
    PostalCode
    False    210
    Name: PostalCode, dtype: int64
    
    Borough
    False    210
    Name: Borough, dtype: int64
    
    Neighborhood
    False    210
    Name: Neighborhood, dtype: int64
    



```python
print('Nearly one third of our data seems to be missing from two out of three columns...What is the best practice?')
print('')
print('We have been told to "Only process the cells that have an assigned borough. Ignore cells with a borough that is "Not assigned".')
print('We will drop the rows where a borough is not assigned.')

df_table.dropna(subset = ['Borough'], axis = 0, inplace = True)
#axis=0 means that the deletion takes place each iterative row.

df_table.reset_index(drop = True, inplace = True)

print('')
print('The new dimensions of the df are: ', df_table.shape)
df_table.head()
```

    Nearly one third of our data seems to be missing from two out of three columns...What is the best practice?
    
    We have been told to "Only process the cells that have an assigned borough. Ignore cells with a borough that is "Not assigned".
    We will drop the rows where a borough is not assigned.
    
    The new dimensions of the df are:  (210, 3)





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
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Heights</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('In which columns does the Neighbourhood column present a NaN value?')
print('')

null_columns = df_table.columns[df_table.isnull().any()]

null_cols = []

for i in list(null_columns):
    null_cols.append(i)
    
null_cols
```

    In which columns does the Neighbourhood column present a NaN value?
    





    []




```python
print('In which rows, for the columns we have already found, does the Neighbourhood column present a NaN value?')
print('')
for i in null_cols:
    print(df_table[df_table[i].isnull()][null_columns])
```

    In which rows, for the columns we have already found, does the Neighbourhood column present a NaN value?
    



```python
df_table['Neighborhood'].replace(np.nan, df_table.iloc[5][1], inplace=True)

print('...and a check...:', df_table.iloc[5][2])
```

    ...and a check...: Queen's Park


##### After all these steps, it is time to reach the finalized dataframe, to conclude the Part 1 of this assignment.


```python
print('And now for the last requirement...')
print('')
df_table2 = df_table.groupby(['PostalCode', 'Borough'])['Neighborhood'].apply(', '.join).reset_index()

print('Here the dimensions of our table are: ', df_table2.shape)
df_table2.head(12)
```

    And now for the last requirement...
    
    Here the dimensions of our table are:  (103, 3)





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
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M1J</td>
      <td>Scarborough</td>
      <td>Scarborough Village</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1K</td>
      <td>Scarborough</td>
      <td>East Birchmount Park, Ionview, Kennedy Park</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M1L</td>
      <td>Scarborough</td>
      <td>Clairlea, Golden Mile, Oakridge</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1M</td>
      <td>Scarborough</td>
      <td>Cliffcrest, Cliffside, Scarborough Village West</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1N</td>
      <td>Scarborough</td>
      <td>Birch Cliff, Cliffside West</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M1P</td>
      <td>Scarborough</td>
      <td>Dorset Park, Scarborough Town Centre, Wexford ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M1R</td>
      <td>Scarborough</td>
      <td>Maryvale, Wexford</td>
    </tr>
  </tbody>
</table>
</div>



#### Part 2

##### We are going to have to install geocoder in order to be able to proceed.


```python
! pip install geocoder
```

    Collecting geocoder
    [?25l  Downloading https://files.pythonhosted.org/packages/4f/6b/13166c909ad2f2d76b929a4227c952630ebaf0d729f6317eb09cbceccbab/geocoder-1.38.1-py2.py3-none-any.whl (98kB)
    [K     |████████████████████████████████| 102kB 18.8MB/s ta 0:00:01
    [?25hCollecting ratelim (from geocoder)
      Downloading https://files.pythonhosted.org/packages/f2/98/7e6d147fd16a10a5f821db6e25f192265d6ecca3d82957a4fdd592cad49c/ratelim-0.1.6-py2.py3-none-any.whl
    Requirement already satisfied: requests in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from geocoder) (2.22.0)
    Collecting future (from geocoder)
    [?25l  Downloading https://files.pythonhosted.org/packages/45/0b/38b06fd9b92dc2b68d58b75f900e97884c45bedd2ff83203d933cf5851c9/future-0.18.2.tar.gz (829kB)
    [K     |████████████████████████████████| 829kB 33.8MB/s eta 0:00:01
    [?25hRequirement already satisfied: six in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from geocoder) (1.13.0)
    Collecting click (from geocoder)
    [?25l  Downloading https://files.pythonhosted.org/packages/fa/37/45185cb5abbc30d7257104c434fe0b07e5a195a6847506c074527aa599ec/Click-7.0-py2.py3-none-any.whl (81kB)
    [K     |████████████████████████████████| 81kB 15.1MB/s eta 0:00:01
    [?25hRequirement already satisfied: decorator in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from ratelim->geocoder) (4.4.1)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->geocoder) (1.25.7)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->geocoder) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->geocoder) (2.8)
    Requirement already satisfied: certifi>=2017.4.17 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->geocoder) (2019.11.28)
    Building wheels for collected packages: future
      Building wheel for future (setup.py) ... [?25ldone
    [?25h  Stored in directory: /home/jupyterlab/.cache/pip/wheels/8b/99/a0/81daf51dcd359a9377b110a8a886b3895921802d2fc1b2397e
    Successfully built future
    Installing collected packages: ratelim, future, click, geocoder
    Successfully installed click-7.0 future-0.18.2 geocoder-1.38.1 ratelim-0.1.6



```python
import geocoder
```

##### Now let's reach the csv file in order to proceed.


```python
# The given csv file is used for the lon and lat values
!wget -O GeoCord.csv http://cocl.us/Geospatial_data/
```

    --2019-12-10 12:10:29--  http://cocl.us/Geospatial_data/
    Resolving cocl.us (cocl.us)... 169.48.113.194
    Connecting to cocl.us (cocl.us)|169.48.113.194|:80... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://cocl.us/Geospatial_data/ [following]
    --2019-12-10 12:10:29--  https://cocl.us/Geospatial_data/
    Connecting to cocl.us (cocl.us)|169.48.113.194|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://ibm.box.com/shared/static/9afzr83pps4pwf2smjjcf1y5mvgb18rr.csv [following]
    --2019-12-10 12:10:30--  https://ibm.box.com/shared/static/9afzr83pps4pwf2smjjcf1y5mvgb18rr.csv
    Resolving ibm.box.com (ibm.box.com)... 107.152.26.197, 107.152.27.197
    Connecting to ibm.box.com (ibm.box.com)|107.152.26.197|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: /public/static/9afzr83pps4pwf2smjjcf1y5mvgb18rr.csv [following]
    --2019-12-10 12:10:31--  https://ibm.box.com/public/static/9afzr83pps4pwf2smjjcf1y5mvgb18rr.csv
    Reusing existing connection to ibm.box.com:443.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://ibm.ent.box.com/public/static/9afzr83pps4pwf2smjjcf1y5mvgb18rr.csv [following]
    --2019-12-10 12:10:31--  https://ibm.ent.box.com/public/static/9afzr83pps4pwf2smjjcf1y5mvgb18rr.csv
    Resolving ibm.ent.box.com (ibm.ent.box.com)... 107.152.27.211, 107.152.26.211
    Connecting to ibm.ent.box.com (ibm.ent.box.com)|107.152.27.211|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://public.boxcloud.com/d/1/b1!IYwxCpgJN4hGRSJCASLgAyZbwvF_8fPY77RObL7svVSZWhkqp8MWXheboysyYshtFUhrS8zAsiKtZkPgVroXGyNMWUwT4Az5E3borwZVCyjWtY8sa0RzB36W7SL_ro0OyRukVex7mkybwDY0SzyDi0FqNbGebt5AC6hODyyqJF3oRCpTzAmVXqTOkFeKzBBJkLN8CF2WXLqyTPml3_sMhJ-QbMADFLzxGDqbcK-ncIVhbHNQwrn39HWtchm0JxiZriPNrTdtCLMO_ICdzcyyPIMZS-cIUJBRoqaoSgLwWV-Rz4f2GsRhOWwndx3lHrxcfzjwuhGMDuDgLAzxYtXnN_Q7hlRpZjLQH1EZaoYu_Bl3fJBgfwpjp07ulq7ieUSGuFFf0rQMQOFUUyzvaTIvI-7x-O-NE2ga_NZMJ0V8_T1lLWodM8ljOzOP151SRsRWXHl2WDNVrEojxlA6JFvjxTQhHhO1AqMIt5rqPX418n2N7mKIL41USGZ4jaTe48Pp2GVb6emQ586-dVltR49stXy-EuRfFLDJ4UcbA2piUI19oPU4s9fe2Kvy43A1tHCKzQGvVmEZrhx_V3eaf6Us_aP1Z8Pl5pC3dtwo2WRRAIENRVHtGM7gRhzIrQ-_o8BTiLj7OOCgIBFv6gWZV2SOJRh6UJiqcVCQ3aqNTBCL8TfjwquY0pu49cdlknIol1qYR6IFHTtQQalWPdtjrP8a1_VVaV4zifKxEi24OUOnCwr9uI3TANQxZRraInR9IQ6dkF5vhQmBJXywS2iMgMISNgo972hNsdkftI6A1hW28JgHl5XeOOfrhHIJ2DEHagYZER4ggd07WrCPdNNgqig5eAE7l-jj-vohmePJjisYB3Mw4QoTChcRr1Eo4wTtnonZcINjQBJZFCo3F_CshKhfBbxJrw5HItWTPdW-n_skbqLJW9IjBOzs-dfx3RnujynGalvS2l04S1Fc2qlhsPs2-p9WCK-3VOLp5bxeVTDqw1mn_PFEJyLq03ldO2JyuV7JAedDTD-KSTQ5W9LUpKDlMkOBtNZQ00PW-reCQ73vmxHSYtWGvzHQr-Mb8kSEzUG0OnSE_psDhDDT7g3lRmlb2xFbCd1hUCz0ZF9ZbyVui-8l7SiyHLiyqkMNOwSGpH72KSAGi5HsXOfIHGyZFYqgyIDRURdKlB1qwbUu43zTPhjxVM0o04gVEmFESetMO-flHd3q9fUOOlCXl8WgqWfXH6O2xaNwJ7I_ucuoK-5Rn0RVKBd-Ke8OifS-fc9XH98rxxI8xgpOu9ug9_7vR9VtqXGKaVUWXey0Noz810VWV4w06JyhGr0AtAP4RcLmEw3XNORw-oIDesRi4rbcMlaDxUfA2grTqnC10DPM3BamQ-abGy782GmrFh1Y0LK8/download [following]
    --2019-12-10 12:10:31--  https://public.boxcloud.com/d/1/b1!IYwxCpgJN4hGRSJCASLgAyZbwvF_8fPY77RObL7svVSZWhkqp8MWXheboysyYshtFUhrS8zAsiKtZkPgVroXGyNMWUwT4Az5E3borwZVCyjWtY8sa0RzB36W7SL_ro0OyRukVex7mkybwDY0SzyDi0FqNbGebt5AC6hODyyqJF3oRCpTzAmVXqTOkFeKzBBJkLN8CF2WXLqyTPml3_sMhJ-QbMADFLzxGDqbcK-ncIVhbHNQwrn39HWtchm0JxiZriPNrTdtCLMO_ICdzcyyPIMZS-cIUJBRoqaoSgLwWV-Rz4f2GsRhOWwndx3lHrxcfzjwuhGMDuDgLAzxYtXnN_Q7hlRpZjLQH1EZaoYu_Bl3fJBgfwpjp07ulq7ieUSGuFFf0rQMQOFUUyzvaTIvI-7x-O-NE2ga_NZMJ0V8_T1lLWodM8ljOzOP151SRsRWXHl2WDNVrEojxlA6JFvjxTQhHhO1AqMIt5rqPX418n2N7mKIL41USGZ4jaTe48Pp2GVb6emQ586-dVltR49stXy-EuRfFLDJ4UcbA2piUI19oPU4s9fe2Kvy43A1tHCKzQGvVmEZrhx_V3eaf6Us_aP1Z8Pl5pC3dtwo2WRRAIENRVHtGM7gRhzIrQ-_o8BTiLj7OOCgIBFv6gWZV2SOJRh6UJiqcVCQ3aqNTBCL8TfjwquY0pu49cdlknIol1qYR6IFHTtQQalWPdtjrP8a1_VVaV4zifKxEi24OUOnCwr9uI3TANQxZRraInR9IQ6dkF5vhQmBJXywS2iMgMISNgo972hNsdkftI6A1hW28JgHl5XeOOfrhHIJ2DEHagYZER4ggd07WrCPdNNgqig5eAE7l-jj-vohmePJjisYB3Mw4QoTChcRr1Eo4wTtnonZcINjQBJZFCo3F_CshKhfBbxJrw5HItWTPdW-n_skbqLJW9IjBOzs-dfx3RnujynGalvS2l04S1Fc2qlhsPs2-p9WCK-3VOLp5bxeVTDqw1mn_PFEJyLq03ldO2JyuV7JAedDTD-KSTQ5W9LUpKDlMkOBtNZQ00PW-reCQ73vmxHSYtWGvzHQr-Mb8kSEzUG0OnSE_psDhDDT7g3lRmlb2xFbCd1hUCz0ZF9ZbyVui-8l7SiyHLiyqkMNOwSGpH72KSAGi5HsXOfIHGyZFYqgyIDRURdKlB1qwbUu43zTPhjxVM0o04gVEmFESetMO-flHd3q9fUOOlCXl8WgqWfXH6O2xaNwJ7I_ucuoK-5Rn0RVKBd-Ke8OifS-fc9XH98rxxI8xgpOu9ug9_7vR9VtqXGKaVUWXey0Noz810VWV4w06JyhGr0AtAP4RcLmEw3XNORw-oIDesRi4rbcMlaDxUfA2grTqnC10DPM3BamQ-abGy782GmrFh1Y0LK8/download
    Resolving public.boxcloud.com (public.boxcloud.com)... 107.152.24.200
    Connecting to public.boxcloud.com (public.boxcloud.com)|107.152.24.200|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 2891 (2.8K) [text/csv]
    Saving to: ‘GeoCord.csv’
    
    GeoCord.csv         100%[===================>]   2.82K  --.-KB/s    in 0s      
    
    2019-12-10 12:10:32 (82.5 MB/s) - ‘GeoCord.csv’ saved [2891/2891]
    



```python
#importing the csv into a pandas dataframe
Coordinates = pd.read_csv('GeoCord.csv')
Coordinates.head()
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
      <th>Postal Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>




```python
#I want to combine the two dataframes together, thus the column which will be used for this combination to take place, 
#PostalCode, should have the same name in both of these dataframes
Coordinates.rename(columns = {'Postal Code':"PostalCode"}, inplace = True)
Coordinates.head()
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
      <th>PostalCode</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



##### Let the dataframe combination take place.


```python
df_table3 = pd.concat([df_table2, Coordinates], axis = 1)

print('The dimensions of the new dataframe are the following:', df_table3.shape)
print('')
df_table3.head(12)
```

    The dimensions of the new dataframe are the following: (103, 6)
    





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
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>PostalCode</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M1J</td>
      <td>Scarborough</td>
      <td>Scarborough Village</td>
      <td>M1J</td>
      <td>43.744734</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1K</td>
      <td>Scarborough</td>
      <td>East Birchmount Park, Ionview, Kennedy Park</td>
      <td>M1K</td>
      <td>43.727929</td>
      <td>-79.262029</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M1L</td>
      <td>Scarborough</td>
      <td>Clairlea, Golden Mile, Oakridge</td>
      <td>M1L</td>
      <td>43.711112</td>
      <td>-79.284577</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1M</td>
      <td>Scarborough</td>
      <td>Cliffcrest, Cliffside, Scarborough Village West</td>
      <td>M1M</td>
      <td>43.716316</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1N</td>
      <td>Scarborough</td>
      <td>Birch Cliff, Cliffside West</td>
      <td>M1N</td>
      <td>43.692657</td>
      <td>-79.264848</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M1P</td>
      <td>Scarborough</td>
      <td>Dorset Park, Scarborough Town Centre, Wexford ...</td>
      <td>M1P</td>
      <td>43.757410</td>
      <td>-79.273304</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M1R</td>
      <td>Scarborough</td>
      <td>Maryvale, Wexford</td>
      <td>M1R</td>
      <td>43.750072</td>
      <td>-79.295849</td>
    </tr>
  </tbody>
</table>
</div>



#### Part 3


```python
!conda install -c conda-forge geopy --yes
!conda install -c conda-forge folium=0.5.0 --yes
```

    Solving environment: done
    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.11
      latest version: 4.7.12
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    
    
    # All requested packages already installed.
    
    Solving environment: done
    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.11
      latest version: 4.7.12
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    
    
    # All requested packages already installed.
    


##### We only want to work with boroughs which contain the word "Toronto".


```python
df_table3['Borough'].value_counts()
```




    North York          24
    Downtown Toronto    19
    Scarborough         17
    Etobicoke           11
    Central Toronto      9
    West Toronto         6
    York                 5
    East York            5
    East Toronto         5
    Mississauga          1
    Queen's Park         1
    Name: Borough, dtype: int64



##### So now we create a dataframe, only containing the boroughs which have something to do with Toronto.


```python
df_toronto = df_table3[df_table3['Borough'].str.contains('Toronto')]
df_toronto.head()
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
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>PostalCode</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>M4E</td>
      <td>43.676357</td>
      <td>-79.293031</td>
    </tr>
    <tr>
      <th>41</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>M4K</td>
      <td>43.679557</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>42</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>M4L</td>
      <td>43.668999</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>43</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>M4M</td>
      <td>43.659526</td>
      <td>-79.340923</td>
    </tr>
    <tr>
      <th>44</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>M4N</td>
      <td>43.728020</td>
      <td>-79.388790</td>
    </tr>
  </tbody>
</table>
</div>



##### Which are the coordinates of Toronto?


```python
from geopy.geocoders import Nominatim
address = 'Toronto, Canada'
geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of Toronto are 43.653963, -79.387207.


##### Now let's visualize the map.


```python
# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means
from sklearn.cluster import KMeans

import folium # map rendering library
```


```python
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_toronto['Latitude'], df_toronto['Longitude'], df_toronto['Borough'], df_toronto['Neighborhood']):
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
        parse_html=False).add_to(map_toronto)  
    
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF8yNTlhYzEyY2IyNjk0MTc0ODE4YTdiM2E1NzM3MTRjYiA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF8yNTlhYzEyY2IyNjk0MTc0ODE4YTdiM2E1NzM3MTRjYicsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzOTYzLC03OS4zODcyMDddLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTEsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzJmZWU3OWRmYzNmOTRjMzc5ZGVlMjA3ZGNmMmU2ZmZkID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82Y2RiYTdmZDUyNWI0YWExYjQ4MjFiNDAxMGQ1NGE1OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3NjM1NzM5OTk5OTk5LC03OS4yOTMwMzEyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI1NzFhOGJhMTQ2NjQzY2Y4N2FiNzBhZWU2MWYwMDgzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VjNWVlNDRjY2Y4MDQzYWFhZWMzMTAzNmNkMTM4MmNiID0gJCgnPGRpdiBpZD0iaHRtbF9lYzVlZTQ0Y2NmODA0M2FhYWVjMzEwMzZjZDEzODJjYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEJlYWNoZXMsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjU3MWE4YmExNDY2NDNjZjg3YWI3MGFlZTYxZjAwODMuc2V0Q29udGVudChodG1sX2VjNWVlNDRjY2Y4MDQzYWFhZWMzMTAzNmNkMTM4MmNiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZjZGJhN2ZkNTI1YjRhYTFiNDgyMWI0MDEwZDU0YTU5LmJpbmRQb3B1cChwb3B1cF8yNTcxYThiYTE0NjY0M2NmODdhYjcwYWVlNjFmMDA4Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZGJmMWVhM2ZlMjM0MmNmYmQzYzg2YjJhNjUzYWJhZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU1NzEsLTc5LjM1MjE4OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yNTlhYzEyY2IyNjk0MTc0ODE4YTdiM2E1NzM3MTRjYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wY2MxNjY5NWNhOTE0YzZjOGUyZjk5N2Y4MjM1ZTM4ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zYjg2MGM1MmRiZDg0MmM1YmMzYzRjMDk2MGYxN2JhOCA9ICQoJzxkaXYgaWQ9Imh0bWxfM2I4NjBjNTJkYmQ4NDJjNWJjM2M0YzA5NjBmMTdiYTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBEYW5mb3J0aCBXZXN0LCBSaXZlcmRhbGUsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGNjMTY2OTVjYTkxNGM2YzhlMmY5OTdmODIzNWUzOGQuc2V0Q29udGVudChodG1sXzNiODYwYzUyZGJkODQyYzViYzNjNGMwOTYwZjE3YmE4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzlkYmYxZWEzZmUyMzQyY2ZiZDNjODZiMmE2NTNhYmFmLmJpbmRQb3B1cChwb3B1cF8wY2MxNjY5NWNhOTE0YzZjOGUyZjk5N2Y4MjM1ZTM4ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YWJkMDkzMmQ2NGM0ODQyOTZhODI1ZTY5MTBkMGQ0MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2ODk5ODUsLTc5LjMxNTU3MTU5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUxYThmMTE4ZmEwOTRkYmNiNzcxMDliNTk1OGU4YzllID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzljNDEwZWIwYzk4MzQ3NGI5OTg2MWQyMTI4ODFjNDE1ID0gJCgnPGRpdiBpZD0iaHRtbF85YzQxMGViMGM5ODM0NzRiOTk4NjFkMjEyODgxYzQxNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEJlYWNoZXMgV2VzdCwgSW5kaWEgQmF6YWFyLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUxYThmMTE4ZmEwOTRkYmNiNzcxMDliNTk1OGU4YzllLnNldENvbnRlbnQoaHRtbF85YzQxMGViMGM5ODM0NzRiOTk4NjFkMjEyODgxYzQxNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YWJkMDkzMmQ2NGM0ODQyOTZhODI1ZTY5MTBkMGQ0MS5iaW5kUG9wdXAocG9wdXBfNTFhOGYxMThmYTA5NGRiY2I3NzEwOWI1OTU4ZThjOWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTlmOGY3OTQyMTliNDM3Y2JkYzBhZDIxNDljOTQ2MDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTk1MjU1LC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjI4OWUyZWZhN2E2NDJlYzk4NTViNWZjNGRlMjBjMGQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2E1NDRhMGE0YzBhNDdmNjgxYzY5Y2UxMzk2ZDY0MzIgPSAkKCc8ZGl2IGlkPSJodG1sX2NhNTQ0YTBhNGMwYTQ3ZjY4MWM2OWNlMTM5NmQ2NDMyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdHVkaW8gRGlzdHJpY3QsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjI4OWUyZWZhN2E2NDJlYzk4NTViNWZjNGRlMjBjMGQuc2V0Q29udGVudChodG1sX2NhNTQ0YTBhNGMwYTQ3ZjY4MWM2OWNlMTM5NmQ2NDMyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk5ZjhmNzk0MjE5YjQzN2NiZGMwYWQyMTQ5Yzk0NjAwLmJpbmRQb3B1cChwb3B1cF9iMjg5ZTJlZmE3YTY0MmVjOTg1NWI1ZmM0ZGUyMGMwZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYzYxM2EyMWY4MTY0OWY5YTZhZDQ4NjQ5Mzc4YWUyYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODAyMDUsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGY1ZDBhNDk4YmQ1NGEyMzg1MzFmY2U3M2I1OGQ0MDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzU5YjhjYTE4ZjMwNDA2MGIyMzk4M2M2ZWFiNjQ3ZmIgPSAkKCc8ZGl2IGlkPSJodG1sXzM1OWI4Y2ExOGYzMDQwNjBiMjM5ODNjNmVhYjY0N2ZiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MYXdyZW5jZSBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRmNWQwYTQ5OGJkNTRhMjM4NTMxZmNlNzNiNThkNDA3LnNldENvbnRlbnQoaHRtbF8zNTliOGNhMThmMzA0MDYwYjIzOTgzYzZlYWI2NDdmYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zYzYxM2EyMWY4MTY0OWY5YTZhZDQ4NjQ5Mzc4YWUyYi5iaW5kUG9wdXAocG9wdXBfNGY1ZDBhNDk4YmQ1NGEyMzg1MzFmY2U3M2I1OGQ0MDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjE3ODQzOGU0OWQ1NGVjMWJlY2I4ZTRlYmZhZDFiODYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTI3NTExLC03OS4zOTAxOTc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk5MTU2ZjViOGY5MTQwMzM4MzE2ZDEwODViMjdlZTczID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y2YzYyYzRlZjk5ZjQxNTZhNGUzMWIxNzAxMjI1OTRmID0gJCgnPGRpdiBpZD0iaHRtbF9mNmM2MmM0ZWY5OWY0MTU2YTRlMzFiMTcwMTIyNTk0ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSBOb3J0aCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85OTE1NmY1YjhmOTE0MDMzODMxNmQxMDg1YjI3ZWU3My5zZXRDb250ZW50KGh0bWxfZjZjNjJjNGVmOTlmNDE1NmE0ZTMxYjE3MDEyMjU5NGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjE3ODQzOGU0OWQ1NGVjMWJlY2I4ZTRlYmZhZDFiODYuYmluZFBvcHVwKHBvcHVwXzk5MTU2ZjViOGY5MTQwMzM4MzE2ZDEwODViMjdlZTczKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzgxZDY3MmQ5OTQ1MzQ0OGJhYzM1OTFkY2MxNTlkMWU2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE1MzgzNCwtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGY1Zjk0YTUxMTNjNDNkZTllYjQ2MWEzMzAxYzQzMzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTg3ZjE3OTlkY2FmNDM0ZThkMjFjYjkyZmYyNzg1Y2QgPSAkKCc8ZGl2IGlkPSJodG1sXzk4N2YxNzk5ZGNhZjQzNGU4ZDIxY2I5MmZmMjc4NWNkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBUb3JvbnRvIFdlc3QsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGY1Zjk0YTUxMTNjNDNkZTllYjQ2MWEzMzAxYzQzMzMuc2V0Q29udGVudChodG1sXzk4N2YxNzk5ZGNhZjQzNGU4ZDIxY2I5MmZmMjc4NWNkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzgxZDY3MmQ5OTQ1MzQ0OGJhYzM1OTFkY2MxNTlkMWU2LmJpbmRQb3B1cChwb3B1cF9kZjVmOTRhNTExM2M0M2RlOWViNDYxYTMzMDFjNDMzMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iZjk2NzBmMTcyMmU0MjdkOTFmYWUwMjBmYzVkMTExYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNDMyNDQsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTVlYWU1NWNmMTQyNGVlNTkwMzBmODlmOGFmM2UxZjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGI1NTQ0NDI0NWMyNDljMDgzZWFlNzJlMDMwMDVlMmYgPSAkKCc8ZGl2IGlkPSJodG1sX2RiNTU0NDQyNDVjMjQ5YzA4M2VhZTcyZTAzMDA1ZTJmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EYXZpc3ZpbGxlLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E1ZWFlNTVjZjE0MjRlZTU5MDMwZjg5ZjhhZjNlMWY2LnNldENvbnRlbnQoaHRtbF9kYjU1NDQ0MjQ1YzI0OWMwODNlYWU3MmUwMzAwNWUyZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iZjk2NzBmMTcyMmU0MjdkOTFmYWUwMjBmYzVkMTExYS5iaW5kUG9wdXAocG9wdXBfYTVlYWU1NWNmMTQyNGVlNTkwMzBmODlmOGFmM2UxZjYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWEzZDI1N2RmMjJhNDJkMjgwNGZiMWZiY2NjZjMyMTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yNTlhYzEyY2IyNjk0MTc0ODE4YTdiM2E1NzM3MTRjYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lZDU4MmZiZjFjODY0ZjhmODAxMGQ5MDcxODIzZTRlYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yODNkMDhkOTRkZmU0ZWVhYWRjODQ1M2ZhMjQyY2I1MiA9ICQoJzxkaXYgaWQ9Imh0bWxfMjgzZDA4ZDk0ZGZlNGVlYWFkYzg0NTNmYTI0MmNiNTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vb3JlIFBhcmssIFN1bW1lcmhpbGwgRWFzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lZDU4MmZiZjFjODY0ZjhmODAxMGQ5MDcxODIzZTRlYS5zZXRDb250ZW50KGh0bWxfMjgzZDA4ZDk0ZGZlNGVlYWFkYzg0NTNmYTI0MmNiNTIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWEzZDI1N2RmMjJhNDJkMjgwNGZiMWZiY2NjZjMyMTQuYmluZFBvcHVwKHBvcHVwX2VkNTgyZmJmMWM4NjRmOGY4MDEwZDkwNzE4MjNlNGVhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYwNjUxNDRkYmNmZTRlYmViMDgyMTM0NTc0ZmQxYTE1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg2NDEyMjk5OTk5OTksLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjBlY2MwOWI1MjI2NGY5ZWExMmRhNmRjYzJiODk1YTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2VlNzM3Y2NjZmFiNGI1ZWI4MTQzMWZiMGU5OWRmNGQgPSAkKCc8ZGl2IGlkPSJodG1sXzNlZTczN2NjY2ZhYjRiNWViODE0MzFmYjBlOTlkZjRkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZWVyIFBhcmssIEZvcmVzdCBIaWxsIFNFLCBSYXRobmVsbHksIFNvdXRoIEhpbGwsIFN1bW1lcmhpbGwgV2VzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMGVjYzA5YjUyMjY0ZjllYTEyZGE2ZGNjMmI4OTVhMS5zZXRDb250ZW50KGh0bWxfM2VlNzM3Y2NjZmFiNGI1ZWI4MTQzMWZiMGU5OWRmNGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjA2NTE0NGRiY2ZlNGViZWIwODIxMzQ1NzRmZDFhMTUuYmluZFBvcHVwKHBvcHVwX2IwZWNjMDliNTIyNjRmOWVhMTJkYTZkY2MyYjg5NWExKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJmZWUwNWI3Zjc1MDRlMTc4NWIxMjgxZmQwNWJjZjRmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTYyNiwtNzkuMzc3NTI5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDVjNzdiNTA5YTFlNDY4Njg0YjlmMGUyNmE3ZTI4ZmUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGYyYWRhZGVmNTMwNGE0MjljZmU4ODk1MzNhZGE3YjkgPSAkKCc8ZGl2IGlkPSJodG1sX2RmMmFkYWRlZjUzMDRhNDI5Y2ZlODg5NTMzYWRhN2I5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlZGFsZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDVjNzdiNTA5YTFlNDY4Njg0YjlmMGUyNmE3ZTI4ZmUuc2V0Q29udGVudChodG1sX2RmMmFkYWRlZjUzMDRhNDI5Y2ZlODg5NTMzYWRhN2I5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJmZWUwNWI3Zjc1MDRlMTc4NWIxMjgxZmQwNWJjZjRmLmJpbmRQb3B1cChwb3B1cF8wNWM3N2I1MDlhMWU0Njg2ODRiOWYwZTI2YTdlMjhmZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MDI5YWMzNTYyNmI0OTQ4YmI4ZmIxOTI5YTFhZWRlZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yNTlhYzEyY2IyNjk0MTc0ODE4YTdiM2E1NzM3MTRjYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84M2QwOWI5Yjg2ZTk0YTM3OWUxNmJkM2NmNjU1NDlhMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lOTYyYjY0NmNmZTY0NzQ3YTNmNDI1NWY5NzBlZWUwMCA9ICQoJzxkaXYgaWQ9Imh0bWxfZTk2MmI2NDZjZmU2NDc0N2EzZjQyNTVmOTcwZWVlMDAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhYmJhZ2V0b3duLCBTdC4gSmFtZXMgVG93biwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODNkMDliOWI4NmU5NGEzNzllMTZiZDNjZjY1NTQ5YTAuc2V0Q29udGVudChodG1sX2U5NjJiNjQ2Y2ZlNjQ3NDdhM2Y0MjU1Zjk3MGVlZTAwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQwMjlhYzM1NjI2YjQ5NDhiYjhmYjE5MjlhMWFlZGVmLmJpbmRQb3B1cChwb3B1cF84M2QwOWI5Yjg2ZTk0YTM3OWUxNmJkM2NmNjU1NDlhMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hYjIwMmZmY2VjODM0MTU4YmE4NmE2ZjZmYzQzYjZhMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2NTg1OTksLTc5LjM4MzE1OTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhiOTk5Mzg1OTI4YjQ1MTM4NGYxY2IyZjQ3MDAzMDVhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhhM2EzODhjZGNmOTQ0ZTJhOGFlODg5NWI0NGQzMWFmID0gJCgnPGRpdiBpZD0iaHRtbF84YTNhMzg4Y2RjZjk0NGUyYThhZTg4OTViNDRkMzFhZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2h1cmNoIGFuZCBXZWxsZXNsZXksIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhiOTk5Mzg1OTI4YjQ1MTM4NGYxY2IyZjQ3MDAzMDVhLnNldENvbnRlbnQoaHRtbF84YTNhMzg4Y2RjZjk0NGUyYThhZTg4OTViNDRkMzFhZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hYjIwMmZmY2VjODM0MTU4YmE4NmE2ZjZmYzQzYjZhMS5iaW5kUG9wdXAocG9wdXBfOGI5OTkzODU5MjhiNDUxMzg0ZjFjYjJmNDcwMDMwNWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTgxZTc5ZDM1NDBlNGMyMmFlODMwZjExMWQzZWQyM2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTQyNTk5LC03OS4zNjA2MzU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzAyMTc3NmY1YzJlMzQ4NzViYmU3YmFiNTQ3MGMxMDYwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBmYmVkNmZjOWY0OTQyYzU5YjExOTI0MDlkMzI1ZTNmID0gJCgnPGRpdiBpZD0iaHRtbF8wZmJlZDZmYzlmNDk0MmM1OWIxMTkyNDA5ZDMyNWUzZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMjE3NzZmNWMyZTM0ODc1YmJlN2JhYjU0NzBjMTA2MC5zZXRDb250ZW50KGh0bWxfMGZiZWQ2ZmM5ZjQ5NDJjNTliMTE5MjQwOWQzMjVlM2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTgxZTc5ZDM1NDBlNGMyMmFlODMwZjExMWQzZWQyM2UuYmluZFBvcHVwKHBvcHVwXzAyMTc3NmY1YzJlMzQ4NzViYmU3YmFiNTQ3MGMxMDYwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM5ZjE1OTNiZDI1ZjRhZjU4MTUzZDAxMWM2YmI2ZTY1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjVmY2M1Mzc5MTJkNDQ5NWIzYWFiNWI1OTNlZDViNDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDExYWFiNDgwNWZhNDdlYjgzODUxY2NhMThkMjczNzcgPSAkKCc8ZGl2IGlkPSJodG1sXzQxMWFhYjQ4MDVmYTQ3ZWI4Mzg1MWNjYTE4ZDI3Mzc3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SeWVyc29uLCBHYXJkZW4gRGlzdHJpY3QsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I1ZmNjNTM3OTEyZDQ0OTViM2FhYjViNTkzZWQ1YjQzLnNldENvbnRlbnQoaHRtbF80MTFhYWI0ODA1ZmE0N2ViODM4NTFjY2ExOGQyNzM3Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zOWYxNTkzYmQyNWY0YWY1ODE1M2QwMTFjNmJiNmU2NS5iaW5kUG9wdXAocG9wdXBfYjVmY2M1Mzc5MTJkNDQ5NWIzYWFiNWI1OTNlZDViNDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjhlY2U2MjhkYTJmNDZmNWIxMWViNGM5NGYyZmMzZjkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE0OTM5LC03OS4zNzU0MTc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZiNzRmNDg5OTZlMTQ5ODM4Y2I5ZjI2OWJkNDBkZTEzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg0NGRkZjFkZTU0YzQwZTRiMzM5NTk4M2FjNjkwNjkyID0gJCgnPGRpdiBpZD0iaHRtbF84NDRkZGYxZGU1NGM0MGU0YjMzOTU5ODNhYzY5MDY5MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZiNzRmNDg5OTZlMTQ5ODM4Y2I5ZjI2OWJkNDBkZTEzLnNldENvbnRlbnQoaHRtbF84NDRkZGYxZGU1NGM0MGU0YjMzOTU5ODNhYzY5MDY5Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iOGVjZTYyOGRhMmY0NmY1YjExZWI0Yzk0ZjJmYzNmOS5iaW5kUG9wdXAocG9wdXBfZmI3NGY0ODk5NmUxNDk4MzhjYjlmMjY5YmQ0MGRlMTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTlmNzk0N2U4ZjBhNDMwM2I3MWFmZGQ3YzEzNWEwMTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDQ3NzA3OTk5OTk5OTYsLTc5LjM3MzMwNjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2FmOTg3MGRhNGUxNGQyNThmYzQ3MTg0ZjBjNmNkZDYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGIyMzI1Yjg2ZWJkNGVlODlkNTI2NTJkNWVjNDc0YzYgPSAkKCc8ZGl2IGlkPSJodG1sX2RiMjMyNWI4NmViZDRlZTg5ZDUyNjUyZDVlYzQ3NGM2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZXJjenkgUGFyaywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2FmOTg3MGRhNGUxNGQyNThmYzQ3MTg0ZjBjNmNkZDYuc2V0Q29udGVudChodG1sX2RiMjMyNWI4NmViZDRlZTg5ZDUyNjUyZDVlYzQ3NGM2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk5Zjc5NDdlOGYwYTQzMDNiNzFhZmRkN2MxMzVhMDEyLmJpbmRQb3B1cChwb3B1cF9jYWY5ODcwZGE0ZTE0ZDI1OGZjNDcxODRmMGM2Y2RkNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mZDMwNjJjOGNhY2M0Njc2ODM0NzhiMGE5NTZlMzJlYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1Nzk1MjQsLTc5LjM4NzM4MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjkzZmJlMTFjYmM4NDBjYTljMGQxYTRhZDE0MTQyY2MgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjZlMDI2ZjcwYjdmNDAwYmFlMWM3OThiNTkyN2E4YTggPSAkKCc8ZGl2IGlkPSJodG1sX2Y2ZTAyNmY3MGI3ZjQwMGJhZTFjNzk4YjU5MjdhOGE4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIEJheSBTdHJlZXQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY5M2ZiZTExY2JjODQwY2E5YzBkMWE0YWQxNDE0MmNjLnNldENvbnRlbnQoaHRtbF9mNmUwMjZmNzBiN2Y0MDBiYWUxYzc5OGI1OTI3YThhOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mZDMwNjJjOGNhY2M0Njc2ODM0NzhiMGE5NTZlMzJlYi5iaW5kUG9wdXAocG9wdXBfNjkzZmJlMTFjYmM4NDBjYTljMGQxYTRhZDE0MTQyY2MpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjNiNmFmYTk2YWFmNDQwMWE1YTBjYzYwNmUwMmVlN2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA1NzEyMDAwMDAwMSwtNzkuMzg0NTY3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yNTlhYzEyY2IyNjk0MTc0ODE4YTdiM2E1NzM3MTRjYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81Mzk5NzBmMDNhYzM0YWI1YWFmMzRlZWZlODAyNDY2ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81YmQ3YjU1NmZiNzE0M2JkOGQzOGE3YmMyZTk2NzI4ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfNWJkN2I1NTZmYjcxNDNiZDhkMzhhN2JjMmU5NjcyOGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFkZWxhaWRlLCBLaW5nLCBSaWNobW9uZCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTM5OTcwZjAzYWMzNGFiNWFhZjM0ZWVmZTgwMjQ2NmUuc2V0Q29udGVudChodG1sXzViZDdiNTU2ZmI3MTQzYmQ4ZDM4YTdiYzJlOTY3MjhmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYzYjZhZmE5NmFhZjQ0MDFhNWEwY2M2MDZlMDJlZTdlLmJpbmRQb3B1cChwb3B1cF81Mzk5NzBmMDNhYzM0YWI1YWFmMzRlZWZlODAyNDY2ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wZDViOTA2ZTg2ZGE0ZDMxOTVlYmM3NjlmZDUxOTkxMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MDgxNTcsLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I3OGY3YTI3YjhhNzQxYTZiMjdjM2EzNjJhNjdhNDI4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I1ZjdlZDY4ZDJhNzRkYzc5MTljMWRmYmVkMjEyYTNkID0gJCgnPGRpdiBpZD0iaHRtbF9iNWY3ZWQ2OGQyYTc0ZGM3OTE5YzFkZmJlZDIxMmEzZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250IEVhc3QsIFRvcm9udG8gSXNsYW5kcywgVW5pb24gU3RhdGlvbiwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjc4ZjdhMjdiOGE3NDFhNmIyN2MzYTM2MmE2N2E0Mjguc2V0Q29udGVudChodG1sX2I1ZjdlZDY4ZDJhNzRkYzc5MTljMWRmYmVkMjEyYTNkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBkNWI5MDZlODZkYTRkMzE5NWViYzc2OWZkNTE5OTEzLmJpbmRQb3B1cChwb3B1cF9iNzhmN2EyN2I4YTc0MWE2YjI3YzNhMzYyYTY3YTQyOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMjFiMDcyMDg4OGY0ZWVjODQxNDJhMzJlYjRkZjFmZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkyZDRmYjU1YzJlYjQwMWFiMTAwYTNmNzUxMWFiMGM4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZjZGVmMGU4OGM5ZjQ1Yzg5NTBjNGMyZGY2ODhiNTU1ID0gJCgnPGRpdiBpZD0iaHRtbF9mY2RlZjBlODhjOWY0NWM4OTUwYzRjMmRmNjg4YjU1NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGVzaWduIEV4Y2hhbmdlLCBUb3JvbnRvIERvbWluaW9uIENlbnRyZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTJkNGZiNTVjMmViNDAxYWIxMDBhM2Y3NTExYWIwYzguc2V0Q29udGVudChodG1sX2ZjZGVmMGU4OGM5ZjQ1Yzg5NTBjNGMyZGY2ODhiNTU1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QyMWIwNzIwODg4ZjRlZWM4NDE0MmEzMmViNGRmMWZmLmJpbmRQb3B1cChwb3B1cF85MmQ0ZmI1NWMyZWI0MDFhYjEwMGEzZjc1MTFhYjBjOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kZjg3NWJiZjE1MmE0NzZiYWI2OGI5ODBlZWY2ZTY3NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODE5ODUsLTc5LjM3OTgxNjkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUyODcyYzFmYTBjMjQzZTBhMTJmNTM5OGY2NGI0OTQ0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzViMzk4OTFhMjY5ZjQxYWY5MmFjOGY3MzUwOTYyNDc5ID0gJCgnPGRpdiBpZD0iaHRtbF81YjM5ODkxYTI2OWY0MWFmOTJhYzhmNzM1MDk2MjQ3OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q29tbWVyY2UgQ291cnQsIFZpY3RvcmlhIEhvdGVsLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81Mjg3MmMxZmEwYzI0M2UwYTEyZjUzOThmNjRiNDk0NC5zZXRDb250ZW50KGh0bWxfNWIzOTg5MWEyNjlmNDFhZjkyYWM4ZjczNTA5NjI0NzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGY4NzViYmYxNTJhNDc2YmFiNjhiOTgwZWVmNmU2NzcuYmluZFBvcHVwKHBvcHVwXzUyODcyYzFmYTBjMjQzZTBhMTJmNTM5OGY2NGI0OTQ0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzgzZDY1MGFmNGExODRhYTNiMTJkOGJjNDViNDZjNThmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzExNjk0OCwtNzkuNDE2OTM1NTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTBiOTFkM2I4ZGRjNDYxYmFhNGNlMGYxYmFjYTljMzYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTI4MTBmMDhkNTIyNDJkMjk0NDE5YjFhOThiZjljMDIgPSAkKCc8ZGl2IGlkPSJodG1sX2EyODEwZjA4ZDUyMjQyZDI5NDQxOWIxYTk4YmY5YzAyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlbGF3biwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMGI5MWQzYjhkZGM0NjFiYWE0Y2UwZjFiYWNhOWMzNi5zZXRDb250ZW50KGh0bWxfYTI4MTBmMDhkNTIyNDJkMjk0NDE5YjFhOThiZjljMDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODNkNjUwYWY0YTE4NGFhM2IxMmQ4YmM0NWI0NmM1OGYuYmluZFBvcHVwKHBvcHVwX2EwYjkxZDNiOGRkYzQ2MWJhYTRjZTBmMWJhY2E5YzM2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhiZDA0Njg1ZTIzYjQ0NDZhNDk1MTkxNTJhYjIxMGI5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk2OTQ3NiwtNzkuNDExMzA3MjAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzI2Mjk5MjlkZWY0NDhlN2EwYzc0ODZiZmFhODIxZDkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDdkMGM3MGU2MjMyNGZlOTk5Y2EyOWYwNGE0MjRkNGQgPSAkKCc8ZGl2IGlkPSJodG1sXzA3ZDBjNzBlNjIzMjRmZTk5OWNhMjlmMDRhNDI0ZDRkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Gb3Jlc3QgSGlsbCBOb3J0aCwgRm9yZXN0IEhpbGwgV2VzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jMjYyOTkyOWRlZjQ0OGU3YTBjNzQ4NmJmYWE4MjFkOS5zZXRDb250ZW50KGh0bWxfMDdkMGM3MGU2MjMyNGZlOTk5Y2EyOWYwNGE0MjRkNGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGJkMDQ2ODVlMjNiNDQ0NmE0OTUxOTE1MmFiMjEwYjkuYmluZFBvcHVwKHBvcHVwX2MyNjI5OTI5ZGVmNDQ4ZTdhMGM3NDg2YmZhYTgyMWQ5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M4ZjYwMzRhMzFjMTQ2ZTZiMjEyNDBkYWRkYzI2NmFjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjcyNzA5NywtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTU1Yjk3OTU2ZmRlNDljNGIzZTg0OWFiNjNjZDVmNmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzk1NmZkYmNjMWRlNDBmYmFmMmY1ZDM5NzI5NDNiZTggPSAkKCc8ZGl2IGlkPSJodG1sX2M5NTZmZGJjYzFkZTQwZmJhZjJmNWQzOTcyOTQzYmU4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgQW5uZXgsIE5vcnRoIE1pZHRvd24sIFlvcmt2aWxsZSwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NTViOTc5NTZmZGU0OWM0YjNlODQ5YWI2M2NkNWY2Yy5zZXRDb250ZW50KGh0bWxfYzk1NmZkYmNjMWRlNDBmYmFmMmY1ZDM5NzI5NDNiZTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzhmNjAzNGEzMWMxNDZlNmIyMTI0MGRhZGRjMjY2YWMuYmluZFBvcHVwKHBvcHVwXzU1NWI5Nzk1NmZkZTQ5YzRiM2U4NDlhYjYzY2Q1ZjZjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M5ZGM2N2ZiMGE0ZDQxZmI5YTljZjkxM2FhODY0ZTgxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNjk1NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yNTlhYzEyY2IyNjk0MTc0ODE4YTdiM2E1NzM3MTRjYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hOGQ3YjljOGMxNzg0ODc3Yjc0MDk3NWM2ODc3ZjljNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80M2RjZjU4YmE4OTU0MzlkODE5MDQ5MmQ4NTkwMWQ2MSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDNkY2Y1OGJhODk1NDM5ZDgxOTA0OTJkODU5MDFkNjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvcmQsIFVuaXZlcnNpdHkgb2YgVG9yb250bywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYThkN2I5YzhjMTc4NDg3N2I3NDA5NzVjNjg3N2Y5Yzcuc2V0Q29udGVudChodG1sXzQzZGNmNThiYTg5NTQzOWQ4MTkwNDkyZDg1OTAxZDYxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M5ZGM2N2ZiMGE0ZDQxZmI5YTljZjkxM2FhODY0ZTgxLmJpbmRQb3B1cChwb3B1cF9hOGQ3YjljOGMxNzg0ODc3Yjc0MDk3NWM2ODc3ZjljNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82Y2Q4MWNkNTM4OTA0NjI4ODQ4M2YwZDVjY2IwYjQxNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzIwNTcsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjQ5YjY3NTNmYzljNGUzOGFkZWE4YzAxMDQ2ZjE1NWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODUwMWMzMzRjMjdkNGI0ZmExMzc4YmU1OTBiZDcxOGUgPSAkKCc8ZGl2IGlkPSJodG1sXzg1MDFjMzM0YzI3ZDRiNGZhMTM3OGJlNTkwYmQ3MThlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaGluYXRvd24sIEdyYW5nZSBQYXJrLCBLZW5zaW5ndG9uIE1hcmtldCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjQ5YjY3NTNmYzljNGUzOGFkZWE4YzAxMDQ2ZjE1NWIuc2V0Q29udGVudChodG1sXzg1MDFjMzM0YzI3ZDRiNGZhMTM3OGJlNTkwYmQ3MThlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZjZDgxY2Q1Mzg5MDQ2Mjg4NDgzZjBkNWNjYjBiNDE1LmJpbmRQb3B1cChwb3B1cF9mNDliNjc1M2ZjOWM0ZTM4YWRlYThjMDEwNDZmMTU1Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYTRkOTIxZWU0Yzc0MWZlYTUyNTM2NDRiMjhmMGUwMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjBkZTdjMjMyYmU1NGY3ZWFkZjdhMzAwMjNhODdjNjUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTQ0ZDU0OWRlODYyNGNhZDk0ZDhmMmZkZWVlNjhiNGQgPSAkKCc8ZGl2IGlkPSJodG1sX2U0NGQ1NDlkZTg2MjRjYWQ5NGQ4ZjJmZGVlZTY4YjRkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DTiBUb3dlciwgQmF0aHVyc3QgUXVheSwgSXNsYW5kIGFpcnBvcnQsIEhhcmJvdXJmcm9udCBXZXN0LCBLaW5nIGFuZCBTcGFkaW5hLCBSYWlsd2F5IExhbmRzLCBTb3V0aCBOaWFnYXJhLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MGRlN2MyMzJiZTU0ZjdlYWRmN2EzMDAyM2E4N2M2NS5zZXRDb250ZW50KGh0bWxfZTQ0ZDU0OWRlODYyNGNhZDk0ZDhmMmZkZWVlNjhiNGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2E0ZDkyMWVlNGM3NDFmZWE1MjUzNjQ0YjI4ZjBlMDMuYmluZFBvcHVwKHBvcHVwXzYwZGU3YzIzMmJlNTRmN2VhZGY3YTMwMDIzYTg3YzY1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2EyMGJhNjBiZTllYTRiMmI4ZGE4NDMzNGZhMDE0ZWY4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmFlMThlOGMzNjBjNDg5N2IzNmZmZjZmMDJlYTVkYmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmRlYmY0MmQ3MGRjNDZmOTliNzcwNzY0ZmI1ZDE5MGIgPSAkKCc8ZGl2IGlkPSJodG1sX2JkZWJmNDJkNzBkYzQ2Zjk5Yjc3MDc2NGZiNWQxOTBiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdG4gQSBQTyBCb3hlcyAyNSBUaGUgRXNwbGFuYWRlLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iYWUxOGU4YzM2MGM0ODk3YjM2ZmZmNmYwMmVhNWRiZi5zZXRDb250ZW50KGh0bWxfYmRlYmY0MmQ3MGRjNDZmOTliNzcwNzY0ZmI1ZDE5MGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTIwYmE2MGJlOWVhNGIyYjhkYTg0MzM0ZmEwMTRlZjguYmluZFBvcHVwKHBvcHVwX2JhZTE4ZThjMzYwYzQ4OTdiMzZmZmY2ZjAyZWE1ZGJmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E0YzQ0ZTAzN2QzZDQ1Nzg5ZWRjYmE2Yzk0YzYwY2EyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yNTlhYzEyY2IyNjk0MTc0ODE4YTdiM2E1NzM3MTRjYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85MjA0NzliNDcwZDc0NmU1OWE5OTBkZDNiYmRiOTNiMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNmRmMTAzYTRiODc0ZDkzYTAzYTRkOTBjOTAxNDE2YiA9ICQoJzxkaXYgaWQ9Imh0bWxfZTZkZjEwM2E0Yjg3NGQ5M2EwM2E0ZDkwYzkwMTQxNmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZpcnN0IENhbmFkaWFuIFBsYWNlLCBVbmRlcmdyb3VuZCBjaXR5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MjA0NzliNDcwZDc0NmU1OWE5OTBkZDNiYmRiOTNiMi5zZXRDb250ZW50KGh0bWxfZTZkZjEwM2E0Yjg3NGQ5M2EwM2E0ZDkwYzkwMTQxNmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTRjNDRlMDM3ZDNkNDU3ODllZGNiYTZjOTRjNjBjYTIuYmluZFBvcHVwKHBvcHVwXzkyMDQ3OWI0NzBkNzQ2ZTU5YTk5MGRkM2JiZGI5M2IyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFkNzQ1ODkwZmY1NjQ3NmRiNjkyZDc3Mzk0MDMyYzY3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5NTQyLC03OS40MjI1NjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk2YjFiODdkZTY2NjRmNmFiMzJlZjVhN2M1ODY3NGU4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M4MWVmMzBkZDI2NjRmM2Q4MDhkMTQxN2M4MjE4ZDRmID0gJCgnPGRpdiBpZD0iaHRtbF9jODFlZjMwZGQyNjY0ZjNkODA4ZDE0MTdjODIxOGQ0ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk2YjFiODdkZTY2NjRmNmFiMzJlZjVhN2M1ODY3NGU4LnNldENvbnRlbnQoaHRtbF9jODFlZjMwZGQyNjY0ZjNkODA4ZDE0MTdjODIxOGQ0Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xZDc0NTg5MGZmNTY0NzZkYjY5MmQ3NzM5NDAzMmM2Ny5iaW5kUG9wdXAocG9wdXBfOTZiMWI4N2RlNjY2NGY2YWIzMmVmNWE3YzU4Njc0ZTgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTQzYWY2ODlhNzU2NGEzZThmMWViMzQ2ODU5ZWFjYzUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjkwMDUxMDAwMDAwMSwtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yNTlhYzEyY2IyNjk0MTc0ODE4YTdiM2E1NzM3MTRjYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lZjRiYTNmM2Q4MGI0OGMxYWViYWI4NGIyZmEwOTM0YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mNjJiNTAyNWM0YmE0MWJlYTFjMzc0OWQwOTA3NTA5OSA9ICQoJzxkaXYgaWQ9Imh0bWxfZjYyYjUwMjVjNGJhNDFiZWExYzM3NDlkMDkwNzUwOTkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvdmVyY291cnQgVmlsbGFnZSwgRHVmZmVyaW4sIFdlc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWY0YmEzZjNkODBiNDhjMWFlYmFiODRiMmZhMDkzNGMuc2V0Q29udGVudChodG1sX2Y2MmI1MDI1YzRiYTQxYmVhMWMzNzQ5ZDA5MDc1MDk5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E0M2FmNjg5YTc1NjRhM2U4ZjFlYjM0Njg1OWVhY2M1LmJpbmRQb3B1cChwb3B1cF9lZjRiYTNmM2Q4MGI0OGMxYWViYWI4NGIyZmEwOTM0Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMTRiM2YyYzY5YzM0MzVlYjc3NjZlMDZjZGQ3MTI3NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yNTlhYzEyY2IyNjk0MTc0ODE4YTdiM2E1NzM3MTRjYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hZmVjYmUxNDIzMDM0ZjA1OGM2Njg0ZGNlZjJlMWExOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hOTM2NzNiZGU3MGQ0NzQ4YjlkNjJlMTFiMTA2MTQyMCA9ICQoJzxkaXYgaWQ9Imh0bWxfYTkzNjczYmRlNzBkNDc0OGI5ZDYyZTExYjEwNjE0MjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxpdHRsZSBQb3J0dWdhbCwgVHJpbml0eSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZmVjYmUxNDIzMDM0ZjA1OGM2Njg0ZGNlZjJlMWExOC5zZXRDb250ZW50KGh0bWxfYTkzNjczYmRlNzBkNDc0OGI5ZDYyZTExYjEwNjE0MjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDE0YjNmMmM2OWMzNDM1ZWI3NzY2ZTA2Y2RkNzEyNzYuYmluZFBvcHVwKHBvcHVwX2FmZWNiZTE0MjMwMzRmMDU4YzY2ODRkY2VmMmUxYTE4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM5NTk1Y2RjN2NiNDQzZDdhZjRmNjA5YWMyN2JlODMwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2ODQ3MiwtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODc5NDliYmVkNTkyNDE1NWJlNTgzMjM3NjQ2YTczYzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDkxNTQyOTM1NGRmNDIxYzhiNjQwOTNmNjRiZjlmNDMgPSAkKCc8ZGl2IGlkPSJodG1sXzQ5MTU0MjkzNTRkZjQyMWM4YjY0MDkzZjY0YmY5ZjQzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ccm9ja3RvbiwgRXhoaWJpdGlvbiBQbGFjZSwgUGFya2RhbGUgVmlsbGFnZSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84Nzk0OWJiZWQ1OTI0MTU1YmU1ODMyMzc2NDZhNzNjMS5zZXRDb250ZW50KGh0bWxfNDkxNTQyOTM1NGRmNDIxYzhiNjQwOTNmNjRiZjlmNDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzk1OTVjZGM3Y2I0NDNkN2FmNGY2MDlhYzI3YmU4MzAuYmluZFBvcHVwKHBvcHVwXzg3OTQ5YmJlZDU5MjQxNTViZTU4MzIzNzY0NmE3M2MxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M0ZWIzM2VkODIzYTQzYjI4YWQwNGEyNGRkMTRlY2ViID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYxNjA4MywtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTc0YzQ5MGEzOTYzNGVkNjg2YTFhNzc4NTg2ZWQ2MTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODkwYTc0ZjZmMjZiNDk2NWE3ZGY3MWFhOThhZTAxYmYgPSAkKCc8ZGl2IGlkPSJodG1sXzg5MGE3NGY2ZjI2YjQ5NjVhN2RmNzFhYTk4YWUwMWJmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IaWdoIFBhcmssIFRoZSBKdW5jdGlvbiBTb3V0aCwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85NzRjNDkwYTM5NjM0ZWQ2ODZhMWE3Nzg1ODZlZDYxNy5zZXRDb250ZW50KGh0bWxfODkwYTc0ZjZmMjZiNDk2NWE3ZGY3MWFhOThhZTAxYmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzRlYjMzZWQ4MjNhNDNiMjhhZDA0YTI0ZGQxNGVjZWIuYmluZFBvcHVwKHBvcHVwXzk3NGM0OTBhMzk2MzRlZDY4NmExYTc3ODU4NmVkNjE3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhhMmM2MjRhODU4YzQzNDE4ZTBmMDA1Y2NhZjlmMjFjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q5MzlkNDFjZGEyZDQyYjU5MDVmZWU3NmI0YTc0NjRmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgwZjdjNTEzOWJiNDQxZGFiNGFkNGVjMmQ2NTQzMmYzID0gJCgnPGRpdiBpZD0iaHRtbF84MGY3YzUxMzliYjQ0MWRhYjRhZDRlYzJkNjU0MzJmMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya2RhbGUsIFJvbmNlc3ZhbGxlcywgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kOTM5ZDQxY2RhMmQ0MmI1OTA1ZmVlNzZiNGE3NDY0Zi5zZXRDb250ZW50KGh0bWxfODBmN2M1MTM5YmI0NDFkYWI0YWQ0ZWMyZDY1NDMyZjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGEyYzYyNGE4NThjNDM0MThlMGYwMDVjY2FmOWYyMWMuYmluZFBvcHVwKHBvcHVwX2Q5MzlkNDFjZGEyZDQyYjU5MDVmZWU3NmI0YTc0NjRmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIxYjAyMTBjNjQ4NzQ0MjU5MjNmZTdkMjExODdlYjMxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNTcwNiwtNzkuNDg0NDQ5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8yNTlhYzEyY2IyNjk0MTc0ODE4YTdiM2E1NzM3MTRjYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNWJjNTQ5MzExMTk0NmNmYTU4ODk5ZjZhOWUyNTgyYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81YjZmNjRiN2ZiYjg0ZTU0YjIwZTJkZmI1ZTM0YzBlZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNWI2ZjY0YjdmYmI4NGU1NGIyMGUyZGZiNWUzNGMwZWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJ1bm55bWVkZSwgU3dhbnNlYSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zNWJjNTQ5MzExMTk0NmNmYTU4ODk5ZjZhOWUyNTgyYy5zZXRDb250ZW50KGh0bWxfNWI2ZjY0YjdmYmI4NGU1NGIyMGUyZGZiNWUzNGMwZWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjFiMDIxMGM2NDg3NDQyNTkyM2ZlN2QyMTE4N2ViMzEuYmluZFBvcHVwKHBvcHVwXzM1YmM1NDkzMTExOTQ2Y2ZhNTg4OTlmNmE5ZTI1ODJjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBiZTg4YTZlNGRhMTRhY2ViYmQxODdkYjBiZDI3M2Q2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNzQzOSwtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzI1OWFjMTJjYjI2OTQxNzQ4MThhN2IzYTU3MzcxNGNiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk1NjM4ZjE0NjM5NDQ1MjU5NmNkMzVmMjY1ZTE1ZjI4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgyZGE2NDU3MjA5YzQzYjFhZDk2NWVjYzI5NjE5ZTI0ID0gJCgnPGRpdiBpZD0iaHRtbF84MmRhNjQ1NzIwOWM0M2IxYWQ5NjVlY2MyOTYxOWUyNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QnVzaW5lc3MgUmVwbHkgTWFpbCBQcm9jZXNzaW5nIENlbnRyZSA5NjkgRWFzdGVybiwgRWFzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85NTYzOGYxNDYzOTQ0NTI1OTZjZDM1ZjI2NWUxNWYyOC5zZXRDb250ZW50KGh0bWxfODJkYTY0NTcyMDljNDNiMWFkOTY1ZWNjMjk2MTllMjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGJlODhhNmU0ZGExNGFjZWJiZDE4N2RiMGJkMjczZDYuYmluZFBvcHVwKHBvcHVwXzk1NjM4ZjE0NjM5NDQ1MjU5NmNkMzVmMjY1ZTE1ZjI4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE1ODY4MjJjMWE5MzRjOWQ4ZDBhMWJiZDMyNmI1NWI0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY3ODU1NiwtNzkuNTMyMjQyNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMjU5YWMxMmNiMjY5NDE3NDgxOGE3YjNhNTczNzE0Y2IpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGVmYWI1ZmUwYjIzNDBmMzhlYzY1ZWRmMTY0OTlkN2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODE5ODBjMjRhOTQ0NGFlZjlkOGZjMTI2YTIzNDM3NzQgPSAkKCc8ZGl2IGlkPSJodG1sXzgxOTgwYzI0YTk0NDRhZWY5ZDhmYzEyNmEyMzQzNzc0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5RdWVlbiYjMzk7cyBQYXJrLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kZWZhYjVmZTBiMjM0MGYzOGVjNjVlZGYxNjQ5OWQ3Yi5zZXRDb250ZW50KGh0bWxfODE5ODBjMjRhOTQ0NGFlZjlkOGZjMTI2YTIzNDM3NzQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTU4NjgyMmMxYTkzNGM5ZDhkMGExYmJkMzI2YjU1YjQuYmluZFBvcHVwKHBvcHVwX2RlZmFiNWZlMGIyMzQwZjM4ZWM2NWVkZjE2NDk5ZDdiKTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4=" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



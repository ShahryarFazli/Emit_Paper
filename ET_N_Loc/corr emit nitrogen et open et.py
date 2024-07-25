#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import warnings
import csv
from osgeo import gdal
import numpy as np
import math
import rasterio as rio
import xarray as xr
import holoviews as hv
import hvplot.xarray
import netCDF4 as nc
import sys
import rioxarray
import h5netcdf
import pandas as pd



# In[4]:


os.chdir(r'C:\Users\shahr\OneDrive\Desktop\New folder')


# In[5]:


os.listdir()


# In[6]:


ds = xr.open_dataset('Nitro_aug.tif')


# In[7]:


new_ds = xr.open_dataset('Average_ET_ROI_aug.tif')


# In[8]:


ds


# In[9]:


new_ds


# In[10]:


new_ds = new_ds['band_data'].isel(band=0)


# In[11]:


band_data = ds['band_data'].isel(band=0)


# In[12]:


#band_data = ds['band_data'].isel(band=0)


# In[13]:


#band_data = ds['band_data'].sel(band=1).values


# In[14]:


# Accessing latitude and longitude values from the dataset
latitudes = ds.coords['y'].values
longitudes = ds.coords['x'].values


# In[15]:


# Assuming the coordinates 'x' and 'y' are the centers of the pixels
x = ds.coords['x']
y = ds.coords['y']


# In[16]:


import pandas as pd
import numpy as np




# In[17]:


# Flatten the data array and coordinate arrays
band_data_flat = band_data.values.flatten()
x_flat = np.repeat(x.values, len(y))
y_flat = np.tile(y.values, len(x))



# In[18]:


df = pd.DataFrame({
    'x': x_flat,
    'y': y_flat,
    'Emit': band_data_flat
})


# In[19]:


df


# In[20]:


# Filter out NaN values if necessary
df = df.dropna(subset=['Emit'])
df = df[df['Emit'] != -9999]


# In[21]:


df


# In[22]:


def extract_value(row, data_array):
    try:
        result = data_array.sel(x=row['x'], y=row['y'], method='nearest').item()
        return result
    except Exception as e:
        print(f"Error extracting data for row {row.name}: {e}")
        return np.nan


# In[23]:


df['new'] = df.apply(lambda row: extract_value(row, new_ds), axis=1)


# In[24]:


# Filter the DataFrame to exclude rows where 'Emit' values are zero
df_nonzero_emit = df[df['Emit'] != 0]

# Plot scatter plot for non-zero 'Emit' values
plt.scatter(df_nonzero_emit['Emit'], df_nonzero_emit['new'], s=0.001)

# Add labels and title
plt.xlabel(' Emit Nitrogen Values')
plt.ylabel('ET Values')
plt.title('Scatter Plot ')

# Show plot
plt.show()


# In[29]:


# Step 1: Read the CSV file containing rice field locations
rice_locations_df = pd.read_csv('rice_locations.csv')  # Replace 'rice_locations.csv' with the actual file path

# Step 2: Find nearest or equal coordinates in rice locations dataset
def find_nearest_rice_location(row, rice_df):
    # Calculate distance for each rice location
    rice_df['distance'] = np.sqrt((rice_df['rice_lat'] - row['y'])**2 + (rice_df['rice_lon'] - row['x'])**2)
    # Find nearest rice location
    nearest_location = rice_df.loc[rice_df['distance'].idxmin()]
    return nearest_location['rice_lat'], nearest_location['rice_lon']

# Apply the function to the original DataFrame to find the nearest rice location for each point
df['rice_lat'], df['rice_lon'] = zip(*df.apply(lambda row: find_nearest_rice_location(row, rice_locations_df), axis=1))

# Step 3: Filter the original DataFrame based on matched rice field locations
matched_rice_df = df.merge(rice_locations_df, on=['rice_lat', 'rice_lon'], how='inner')

# Step 4: Plot the scatter plot with the filtered data
plt.scatter(matched_rice_df['Emit'], matched_rice_df['new'], s=0.001)

# Add labels and title
plt.xlabel('Emit Values')
plt.ylabel('New Values')
plt.title('Scatter Plot of Emit Values vs. New Values (Filtered by Rice Field Locations)')

# Show plot
plt.show()


# In[27]:


import os
import warnings
import csv
import numpy as np
import rasterio as rio
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# Set the working directory
os.chdir(r'C:\Users\shahr\OneDrive\Desktop\New folder')

# Step 1: Read the CSV file containing rice field locations
rice_locations_df = pd.read_csv('rice_locations.csv')  # Replace 'rice_locations.csv' with the actual file path

# Step 2: Define a function to find the closest or equal latitudes and longitudes from the rice CSV file
def find_closest_rice_locations(lat, lon, rice_locations_df, num_points=500):
    # Calculate distances
    rice_locations_df['distance'] = np.sqrt((rice_locations_df['rice_lat'] - lat)**2 + (rice_locations_df['rice_lon'] - lon)**2)
    # Sort by distance and select the closest points
    closest_points = rice_locations_df.sort_values(by='distance').head(num_points)
    return closest_points

# Step 3: Find the closest rice locations for each point in the original DataFrame
closest_rice_locations = []
for index, row in df.iterrows():
    closest_rice_location = find_closest_rice_locations(row['y'], row['x'], rice_locations_df)
    closest_rice_locations.append(closest_rice_location)

# Combine the closest rice locations into a single DataFrame
closest_rice_locations_df = pd.concat(closest_rice_locations)

# Step 4: Merge the closest rice locations DataFrame with the original DataFrame
matched_rice_df = df.merge(closest_rice_locations_df, on=['rice_lat', 'rice_lon'], how='inner')

# Step 5: Plot the scatter plot with the filtered data
plt.scatter(matched_rice_df['Emit'], matched_rice_df['new'], s=0.001)

# Add labels and title
plt.xlabel('Emit Values')
plt.ylabel('New Values')
plt.title('Scatter Plot of Emit Values vs. New Values (Filtered by Closest Rice Field Locations)')

# Show plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





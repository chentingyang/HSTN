import netCDF4
from netCDF4 import Dataset
import numpy as np
import sys
import os


subname = '_ITPCAS-CMFD_V0106_B-01_03hr_010deg_2017';
feature_names = ['LRad', 'Prec', 'Pres', 'SHum', 'SRad', 'Temp', 'Wind']
data = []
for feature in feature_names:
	feature_data = None
	for i in range(4, 12):
		if i < 10:
			i = '0' + str(i)
		else:
			i = str(i)
		name = feature + '/' + feature.lower() + subname + i + '.nc'

		
		nc_obj=Dataset(name)

		# t=(nc_obj.variables['time'][:])
		lat=(nc_obj.variables['lat'][:])
		lon=(nc_obj.variables['lon'][:])
		lrad = (nc_obj.variables[feature.lower()][:])

		print(name, lat[49], lon[452])

		if feature_data is None:
			feature_data = np.array(lrad[:, 49, 452])
		else:
			feature_data = np.concatenate((feature_data, lrad[:, 49, 452]), axis = 0)

	data.append(feature_data.reshape(-1, 1).repeat(6, axis = 0))

data = np.concatenate(data, axis = 1)
print(data.shape)
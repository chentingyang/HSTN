# HSTN
This is a Tensorflow implementation of Origin-Destination Traffic Prediction based on Hybrid Spatio-Temporal Network.

# Requirements
python==3.8.5

keras==2.2.0

pandas==1.2.3

tensorflow==2.2.0

numpy==1.19.2

# Training and Testing
cd NYC-OD

python HSTN_NY_train_test.py

or

cd HAIKOU-OD

python HSTN_HK_train_test.py

or

cd SHENZHEN-OD

python HSTN_SZ_train_test.py

# Data Source
The NYC OD data is provided from [Contextualized Spatial–Temporal Network for Taxi Origin-Destination Demand Prediction](https://ieeexplore.ieee.org/abstract/document/8720246) (https://github.com/liulingbo918/CSTN).

The Haikou Didi data was originally published at https://outreach.didichuxing.com/research/opendata/. The page is not available now. We are not authorized to republish the data. Users who are interested in the data may contact the original publisher to request via their homepage at https://outreach.didichuxing.com/.

The Shenzhen Metro OD data is our private data. If you want to access it for research only, please contact us (767278559@qq.com).

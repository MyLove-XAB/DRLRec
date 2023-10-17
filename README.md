# DRLRec

**UISA: User Information Separating Architecture for Commodity Recommendation Policy with Deep Reinforcement Learning**

Journal paper (under review)

## requirements
python=3.6  
gym=0.17.3  
torch=1.8.2  
wandb=0.12.2  
pandas=1.1.1  
numpy=1.19.2  

## Experiments on JDEnv Environment
The environment is based on JData. The e-commerce website is 
https://www.jd.com/, and the JData dataset can be found at 
https://www.kaggle.com/datasets/owincontext/jdata2016.

The files in the directory /JDataExp/data/ are part of environemnt.

Models are in the directory /JDataExp/data/Newmodel.

/JDataExp/model/DataProcessing.py and 
/JDataExp/model/DataProcessing.py are the codes for construct 
JDEnv. If you want to construct JDEnv from scratch, please 
download JData and put it in /JDataExp/data.

## Experiments on Virtual Taobao Environment
The environment cites the work of "Jing-Cheng Shi, Yang Yu, 
Qing Da, Shi-Yong Chen, and An-Xiang Zeng. Virtual-Taobao: 
Virtualizing real-world online retail environment for 
reinforcement learning. In: Proceedings of the 33rd AAAI 
Conference on Artificial Intelligence (AAAI’19), Honolulu, HI, 
2019.", the environment model is same as https://github.com/eyounx/VirtualTaobao.

Our models are in the directory /virtualTB/Newmodel.

/virtualTB/Newmodel/GetData.py generate the testdata.

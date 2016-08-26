import pandas as pd
import pandas as pd
from sklearn import linear_model, preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
Cov = pd.read_csv("C:/Users/hpatel/survey_detail_new.csv_Pieces/CombinedFile_3_1.csv", names = ["PNR" , "DOCUMENT_ID", "MPLVL", "CABIN", "DELAY_MINS", "O_AIRPORT", "EQUIP", "D_AIRPORT", "SEATNUM", "FLEET_GROUP", "ACT_DPRT_DTM", "FLNUM", "EXP_BAG", "EXP_CHKIN", "EXP_GATE", "EXP_LOUNGE", "EXP_TSA", "PILOT_SAT","CCE_SEAT_AREA" , "OBE_FA", "EXP_LOUNGE", 'FARE'] )

MC_BAG = Cov['EXP_BAG'].isnull().sum()
MC_CHKIN = Cov['EXP_CHKIN'].isnull().sum()
MC_GATE = Cov['EXP_GATE'].isnull().sum()
MC_TSA = Cov['EXP_TSA'].isnull().sum()
MC_PILOT = Cov['PILOT_SAT'].isnull().sum()
MC_CCE = Cov['CCE_SEAT_AREA'].isnull().sum()
MC_LOUNGE = Cov['EXP_LOUNGE'].isnull().sum()
MC_FA = Cov['OBE_FA'].isnull().sum()

print "MC_BAG: %d" %MC_BAG
print "MC_CHKIN: %d" %MC_CHKIN
print "MC_GATE: %d" %MC_GATE
print "MC_TSA: %d" %MC_TSA
print "MC_PILOT: %d" %MC_PILOT
print "MC_CCE %d" %MC_CCE
print "MC_LOUNGE:"  %MC_LOUNGE
print "MC_FA: %d" %MC_FA

Cov['chkin_mdedian'] = Cov['EXP_CHKIN'].fillna(Cov['EXP_CHKIN'].median())
Cov['bag_median'] = Cov['EXP_BAG'].fillna(Cov['EXP_BAG'].median())
Cov['gate_mdedian'] = Cov['EXP_CHKIN'].fillna(Cov['EXP_GATE'].median())
Cov['tsa_median'] = Cov['EXP_TSA'].fillna(Cov['EXP_TSA'].median())
Cov['pilot_mdedian'] = Cov['PILOT_SAT'].fillna(Cov['PILOT_SAT'].median())
Cov['fa_median'] = Cov['OBE_FA'].fillna(Cov['OBE_FA'].median())
Cov['Fare_Median'] = Cov['FARE'].fillna(Cov['FARE'].median())

Cov_complete = Cov.dropna()
Cov_complete.set_index("DOCUMENT_ID")



Cov_complete.loc[:,'SAT'] = (Cov['chkin_mdedian'] + Cov['fa_median'] + Cov_complete['bag_median'] + Cov_complete['tsa_median'] + Cov['gate_mdedian']+ Cov['pilot_mdedian']   )/6

Cov_complete.loc[:,"SAT"] = Cov_complete["SAT"].astype("int")
Cov_complete.head(10)

Cov_complete.ix[Cov_complete.DELAY_MINS >=60 , 'delay_recode'] = "Vlate"
Cov_complete.ix[Cov_complete.DELAY_MINS <60   , 'delay_recode'] = "Litlate"
Cov_complete.ix[Cov_complete.DELAY_MINS <=0 , 'delay_recode'] = "early"





MP_encoder = preprocessing.LabelEncoder().fit(Cov_complete["MPLVL"])
Cov_complete.loc[:,'mp_coded'] = MP_encoder.transform(Cov_complete["MPLVL"])

delay_encoder = preprocessing.LabelEncoder().fit(Cov_complete["delay_recode"])
Cov_complete.loc[:,'delay_coded'] = delay_encoder.transform(Cov_complete["delay_recode"])

ARPT_encoder = preprocessing.LabelEncoder().fit(Cov_complete["O_AIRPORT"])
Cov_complete.loc[:,'OAIRPORT_coded'] = ARPT_encoder.transform(Cov_complete["O_AIRPORT"])

cabin_encoder = preprocessing.LabelEncoder().fit(Cov_complete["CABIN"])
Cov_complete.loc[:,'cabin_coded'] = cabin_encoder.transform(Cov_complete["CABIN"])

fare_encoder = preprocessing.LabelEncoder().fit(Cov_complete["Fare_Median"])
Cov_complete.loc[:,'fare_coded'] = fare_encoder.transform(Cov_complete["Fare_Median"])




predictors = [ 'mp_coded', 'OAIRPORT_coded', 'fare_coded', 'delay_coded' , 'cabin_coded' ]
X = Cov_complete[predictors]
y = Cov_complete['SAT']


model = RandomForestClassifier(n_estimators = 20)
    
model.fit(X, y)

features = predictors
feature_importances = model.feature_importances_

features_df = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})
features_df.sort_values('Importance Score', inplace=True, ascending=False)

features_df.head()

from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(max_depth=2)

predictors1 = [ 'mp_coded', 'OAIRPORT_coded', 'delay_coded'  ]
X1 = Cov_complete[predictors1]
y = Cov_complete['SAT']
model1.fit(X1, y)

from sklearn.tree import export_graphviz
import StringIO
def build_tree_image(model1):
    dotfile = StringIO.StringIO()
    export_graphviz(model1,out_file = dotfile, feature_names = X1.columns)
    
    return dotfile.getvalue()
    
print build_tree_image(model1)

import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
import numpy as np
import pandas as pd

X1 = Cov_complete['SAT']
sat = pd.Series(X1)
satt.hist()

X2 = Cov_complete['delay_coded']
delay = pd.Series(X2)
delay.hist()


from sklearn.cluster import KMeans
df = Cov_complete[[ "delay_coded", "SAT","mp_coded"]]
#df.set_index("O_AIRPORT")
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(df)
k_means_labels = k_means.labels_
k_means_labels

clusters = pd.DataFrame(k_means_labels,index=df.index,columns=["Cluster"])
survey_with_cluster = pd.concat([df, clusters], axis=1)
g = sns.lmplot(x="delay_coded", y="SAT", hue="mp_coded", 
               fit_reg=False, 
               data=survey_with_cluster)

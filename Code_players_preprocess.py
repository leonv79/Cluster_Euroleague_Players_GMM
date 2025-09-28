import os
import pandas as pd
import numpy as np
os.chdir("D:/Downloads/Project_Basketball/Players/")

#player_stats_21_22 = pd.read_excel("Players stats_21_22.xlsx",sheet_name='PLAYERS_STATS')
player_stats_21_22 = pd.read_csv("Players stats_21_22.csv")


#Preprocessing part
#Make the datasets viable

#Data cleaning
#Remove NAs 
player_stats_21_22.columns
#player_stats_21_22 = player_stats_21_22[~np.isnan(player_stats_21_22['TM TAG'])]
player_stats_21_22 = player_stats_21_22[~pd.isnull(player_stats_21_22['TM TAG'])]
player_stats_21_22_no_na = player_stats_21_22.dropna(axis=1)

#Remove duplictes
distinct = list(set(player_stats_21_22_no_na['NAME']))
player_stats_21_22_v2 = player_stats_21_22_no_na[player_stats_21_22_no_na['NAME'].isin(distinct)]


#Remove columns that contain "-" values throughout
#player_stats_21_22_v2_test = player_stats_21_22_v2.loc[:, (player_stats_21_22_v2!= "-").any(axis=0)]

player_stats_21_22_v3 = player_stats_21_22_v2.loc[:, (player_stats_21_22_v2!= "-").any(axis=0)]


#time to turn numeric values to actual numeric. These columns are the ones after the "L" (losses) column
#split the dataset
player_stats_21_22_v4_part_1 = player_stats_21_22_v3.iloc[:,13:]

columns_to_exclude = player_stats_21_22_v4_part_1.columns
player_stats_21_22_v4_part_to_merge1 = player_stats_21_22_v3.loc[:,~player_stats_21_22_v3.columns.isin(columns_to_exclude)]




#we see there are columns that are either percentages (45% for reference) or contain non numeric values like "-" or "DNP"
player_stats_21_22_v4_part_1 = player_stats_21_22_v4_part_1.replace(["-","DNP"],np.nan)

''''
###NOT RUNNING####

for column in player_stats_21_22_v4_part_1.columns:
    player_stats_21_22_v4_part_1[player_stats_21_22_v4_part_1[column].str.contains("%")]
    
player_stats_21_22_v4_part_1[player_stats_21_22_v4_part_1[column].str.contains("%") for column in player_stats_21_22_v4_part_1.columns]
'''

#Personal note: You can manipulate the mask if you want to filter by rows or by columns by changing the index like this: mask.any(axis=1) or mask.any(axis=0)
#you mask if % exists, then with mask.any(axis=0) the code checks if in any row of a column there is a true value and keeps it else it drops
#the same happens but with rows if axis=1 


mask = player_stats_21_22_v4_part_1.applymap(lambda x: "%" in str(x))
#mask2 = player_stats_21_22_v4_part_1.loc[:,player_stats_21_22_v4_part_1.columns.str.contains("%")]
player_stats_21_22_v4_part_1_filter = player_stats_21_22_v4_part_1.loc[:,mask.any(axis=0)].replace(np.nan,"0%")


player_stats_21_22_v4_part_1_filter_1 = player_stats_21_22_v4_part_1_filter.apply(lambda x: x.str.replace("%",""))

'''
for inside the apply is redundant
player_stats_21_22_v4_part_1_filter_2 = player_stats_21_22_v4_part_1_filter_1.apply(lambda x: pd.to_numeric(player_stats_21_22_v4_part_1_filter_1[x],errors='') for x in player_stats_21_22_v4_part_1_filter_1.columns)
'''

player_stats_21_22_v4_final1 = player_stats_21_22_v4_part_1_filter_1.apply(pd.to_numeric)/100

#now for then non % columns
player_stats_21_22_v4_part_2_filter = player_stats_21_22_v4_part_1.loc[:,~(mask.any(axis=0))].replace(["DNP","-"],"0")
player_stats_21_22_v4_final2 = player_stats_21_22_v4_part_2_filter.apply(pd.to_numeric)

player_stats_21_22_v4_part_to_merge2 = pd.merge(player_stats_21_22_v4_final1,player_stats_21_22_v4_final2,left_index=True,right_index=True) 

player_stats_21_22_final = pd.merge(player_stats_21_22_v4_part_to_merge1,player_stats_21_22_v4_part_to_merge2,left_index=True,right_index=True) 
player_stats_21_22_final = player_stats_21_22_final.replace(np.nan,0)









#22_23 season


#player_stats_22_23 = pd.read_excel("Players stats_22_23.xlsx",sheet_name='PLAYERS_STATS')
player_stats_22_23 = pd.read_csv("Players stats_22_23.csv",  encoding='latin-1')


#Preprocessing part
#Make the datasets viable

#Data cleaning
#Remove NAs 
player_stats_22_23.columns
#player_stats_22_23 = player_stats_22_23[~np.isnan(player_stats_22_23['TM TAG'])]
player_stats_22_23 = player_stats_22_23[~pd.isnull(player_stats_22_23['TM TAG'])]
player_stats_22_23_no_na = player_stats_22_23.dropna(axis=1)

#Remove duplictes
distinct_22_23 = list(set(player_stats_22_23_no_na['NAME']))
player_stats_22_23_v2 = player_stats_22_23_no_na[player_stats_22_23_no_na['NAME'].isin(distinct_22_23)]


#Remove columns that contain "-" values throughout
#player_stats_22_23_v2_test = player_stats_22_23_v2.loc[:, (player_stats_22_23_v2!= "-").any(axis=0)]

player_stats_22_23_v3 = player_stats_22_23_v2.loc[:, (player_stats_22_23_v2!= "-").any(axis=0)]


#time to turn numeric values to actual numeric. These columns are the ones after the "L" (losses) column
#split the dataset
player_stats_22_23_v4_part_1 = player_stats_22_23_v3.iloc[:,13:]

columns_to_exclude = player_stats_22_23_v4_part_1.columns
player_stats_22_23_v4_part_to_merge1 = player_stats_22_23_v3.loc[:,~player_stats_22_23_v3.columns.isin(columns_to_exclude)]




#we see there are columns that are either percentages (45% for reference) or contain non numeric values like "-" or "DNP"
player_stats_22_23_v4_part_1 = player_stats_22_23_v4_part_1.replace(["-","DNP"],np.nan)

''''
###NOT RUNNING####

for column in player_stats_22_23_v4_part_1.columns:
    player_stats_22_23_v4_part_1[player_stats_22_23_v4_part_1[column].str.contains("%")]
    
player_stats_22_23_v4_part_1[player_stats_22_23_v4_part_1[column].str.contains("%") for column in player_stats_22_23_v4_part_1.columns]
'''

#Personal note: You can manipulate the mask if you want to filter by rows or by columns by changing the index like this: mask.any(axis=1) or mask.any(axis=0)
#you mask if % exists, then with mask.any(axis=0) the code checks if in any row of a column there is a true value and keeps it else it drops
#the same happens but with rows if axis=1 


mask = player_stats_22_23_v4_part_1.applymap(lambda x: "%" in str(x))
#mask2 = player_stats_22_23_v4_part_1.loc[:,player_stats_22_23_v4_part_1.columns.str.contains("%")]
player_stats_22_23_v4_part_1_filter = player_stats_22_23_v4_part_1.loc[:,mask.any(axis=0)].replace(np.nan,"0%")
player_stats_22_23_v4_part_1_filter = player_stats_22_23_v4_part_1.loc[:,mask.any(axis=0)].replace("#ÄÉÁÉÑ/0!","0%")

player_stats_22_23_v4_part_1_filter_1 = player_stats_22_23_v4_part_1_filter.apply(lambda x: x.str.replace("%",""))

'''
for inside the apply is redundant
player_stats_22_23_v4_part_1_filter_2 = player_stats_22_23_v4_part_1_filter_1.apply(lambda x: pd.to_numeric(player_stats_22_23_v4_part_1_filter_1[x],errors='') for x in player_stats_22_23_v4_part_1_filter_1.columns)
'''

player_stats_22_23_v4_final1 = player_stats_22_23_v4_part_1_filter_1.apply(pd.to_numeric)/100

#now for then non % columns
player_stats_22_23_v4_part_2_filter = player_stats_22_23_v4_part_1.loc[:,~(mask.any(axis=0))].replace(["DNP","-"],"0")
player_stats_22_23_v4_final2 = player_stats_22_23_v4_part_2_filter.apply(pd.to_numeric)

player_stats_22_23_v4_part_to_merge2 = pd.merge(player_stats_22_23_v4_final1,player_stats_22_23_v4_final2,left_index=True,right_index=True) 

player_stats_22_23_final = pd.merge(player_stats_22_23_v4_part_to_merge1,player_stats_22_23_v4_part_to_merge2,left_index=True,right_index=True) 
player_stats_22_23_final = player_stats_22_23_final.replace(np.nan,0)

    
    



#23_24 season


#player_stats_23_24 = pd.read_excel("Players stats_23_24.xlsx",sheet_name='PLAYERS_STATS')
player_stats_23_24 = pd.read_csv("Players stats_23_24.csv",  encoding='latin-1')


#Preprocessing part
#Make the datasets viable

#Data cleaning
#Remove NAs 
player_stats_23_24.columns
#player_stats_23_24 = player_stats_23_24[~np.isnan(player_stats_23_24['TM TAG'])]
player_stats_23_24 = player_stats_23_24[~pd.isnull(player_stats_23_24['NAME'])]
player_stats_23_24_no_na = player_stats_23_24.dropna(axis=1)

#Remove duplictes
distinct_23_24 = list(set(player_stats_23_24_no_na['NAME']))
player_stats_23_24_v2 = player_stats_23_24_no_na[player_stats_23_24_no_na['NAME'].isin(distinct_23_24)]


#Remove columns that contain "-" values throughout
#player_stats_23_24_v2_test = player_stats_23_24_v2.loc[:, (player_stats_23_24_v2!= "-").any(axis=0)]

player_stats_23_24_v3 = player_stats_23_24_v2.loc[:, (player_stats_23_24_v2!= "-").any(axis=0)]


#time to turn numeric values to actual numeric. These columns are the ones after the "L" (losses) column
#split the dataset
player_stats_23_24_v4_part_1 = player_stats_23_24_v3.iloc[:,7:]

columns_to_exclude = player_stats_23_24_v4_part_1.columns
player_stats_23_24_v4_part_to_merge1 = player_stats_23_24_v3.loc[:,~player_stats_23_24_v3.columns.isin(columns_to_exclude)]




#we see there are columns that are either percentages (45% for reference) or contain non numeric values like "-" or "DNP"
player_stats_23_24_v4_part_1 = player_stats_23_24_v4_part_1.replace(["-","DNP"],np.nan)

''''
###NOT RUNNING####

for column in player_stats_23_24_v4_part_1.columns:
    player_stats_23_24_v4_part_1[player_stats_23_24_v4_part_1[column].str.contains("%")]
    
player_stats_23_24_v4_part_1[player_stats_23_24_v4_part_1[column].str.contains("%") for column in player_stats_23_24_v4_part_1.columns]
'''

#Personal note: You can manipulate the mask if you want to filter by rows or by columns by changing the index like this: mask.any(axis=1) or mask.any(axis=0)
#you mask if % exists, then with mask.any(axis=0) the code checks if in any row of a column there is a true value and keeps it else it drops
#the same happens but with rows if axis=1 


mask = player_stats_23_24_v4_part_1.applymap(lambda x: "%" in str(x))
#mask2 = player_stats_23_24_v4_part_1.loc[:,player_stats_23_24_v4_part_1.columns.str.contains("%")]
player_stats_23_24_v4_part_1_filter = player_stats_23_24_v4_part_1.loc[:,mask.any(axis=0)].replace(np.nan,"0%")


player_stats_23_24_v4_part_1_filter_1 = player_stats_23_24_v4_part_1_filter.apply(lambda x: x.str.replace("%",""))

'''
for inside the apply is redundant
player_stats_23_24_v4_part_1_filter_2 = player_stats_23_24_v4_part_1_filter_1.apply(lambda x: pd.to_numeric(player_stats_23_24_v4_part_1_filter_1[x],errors='') for x in player_stats_23_24_v4_part_1_filter_1.columns)
'''

player_stats_23_24_v4_final1 = player_stats_23_24_v4_part_1_filter_1.apply(pd.to_numeric)/100

#now for then non % columns
player_stats_23_24_v4_part_2_filter = player_stats_23_24_v4_part_1.loc[:,~(mask.any(axis=0))].replace(["DNP","-"],"0")
player_stats_23_24_v4_final2 = player_stats_23_24_v4_part_2_filter.apply(pd.to_numeric)

player_stats_23_24_v4_part_to_merge2 = pd.merge(player_stats_23_24_v4_final1,player_stats_23_24_v4_final2,left_index=True,right_index=True) 

player_stats_23_24_final = pd.merge(player_stats_23_24_v4_part_to_merge1,player_stats_23_24_v4_part_to_merge2,left_index=True,right_index=True) 
player_stats_23_24_final = player_stats_23_24_final.replace(np.nan,0)

    
    


#24_25 season


#player_stats_24_25 = pd.read_excel("Players stats_24_25.xlsx",sheet_name='PLAYERS_STATS')
player_stats_24_25 = pd.read_csv("Players stats_24_25.csv",  encoding='latin-1')


#Preprocessing part
#Make the datasets viable

#Data cleaning
#Remove NAs 
player_stats_24_25.columns
#player_stats_24_25 = player_stats_24_25[~np.isnan(player_stats_24_25['TM TAG'])]
player_stats_24_25 = player_stats_24_25[~pd.isnull(player_stats_24_25['NAME'])]
player_stats_24_25_no_na = player_stats_24_25.dropna(axis=1)

#Remove duplictes
distinct_24_25 = list(set(player_stats_24_25_no_na['NAME']))
player_stats_24_25_v2 = player_stats_24_25_no_na[player_stats_24_25_no_na['NAME'].isin(distinct_24_25)]


#Remove columns that contain "-" values throughout
#player_stats_24_25_v2_test = player_stats_24_25_v2.loc[:, (player_stats_24_25_v2!= "-").any(axis=0)]

player_stats_24_25_v3 = player_stats_24_25_v2.loc[:, (player_stats_24_25_v2!= "-").any(axis=0)]


#time to turn numeric values to actual numeric. These columns are the ones after the "L" (losses) column
#split the dataset
player_stats_24_25_v4_part_1 = player_stats_24_25_v3.iloc[:,7:]

columns_to_exclude = player_stats_24_25_v4_part_1.columns
player_stats_24_25_v4_part_to_merge1 = player_stats_24_25_v3.loc[:,~player_stats_24_25_v3.columns.isin(columns_to_exclude)]




#we see there are columns that are either percentages (45% for reference) or contain non numeric values like "-" or "DNP"
player_stats_24_25_v4_part_1 = player_stats_24_25_v4_part_1.replace(["-","DNP"],np.nan)

''''
###NOT RUNNING####

for column in player_stats_24_25_v4_part_1.columns:
    player_stats_24_25_v4_part_1[player_stats_24_25_v4_part_1[column].str.contains("%")]
    
player_stats_24_25_v4_part_1[player_stats_24_25_v4_part_1[column].str.contains("%") for column in player_stats_24_25_v4_part_1.columns]
'''

#Personal note: You can manipulate the mask if you want to filter by rows or by columns by changing the index like this: mask.any(axis=1) or mask.any(axis=0)
#you mask if % exists, then with mask.any(axis=0) the code checks if in any row of a column there is a true value and keeps it else it drops
#the same happens but with rows if axis=1 


mask = player_stats_24_25_v4_part_1.applymap(lambda x: "%" in str(x))
#mask2 = player_stats_24_25_v4_part_1.loc[:,player_stats_24_25_v4_part_1.columns.str.contains("%")]
player_stats_24_25_v4_part_1_filter = player_stats_24_25_v4_part_1.loc[:,mask.any(axis=0)].replace(np.nan,"0%")


player_stats_24_25_v4_part_1_filter_1 = player_stats_24_25_v4_part_1_filter.apply(lambda x: x.str.replace("%",""))

'''
for inside the apply is redundant
player_stats_24_25_v4_part_1_filter_2 = player_stats_24_25_v4_part_1_filter_1.apply(lambda x: pd.to_numeric(player_stats_24_25_v4_part_1_filter_1[x],errors='') for x in player_stats_24_25_v4_part_1_filter_1.columns)
'''

player_stats_24_25_v4_final1 = player_stats_24_25_v4_part_1_filter_1.apply(pd.to_numeric)/100

#now for then non % columns
player_stats_24_25_v4_part_2_filter = player_stats_24_25_v4_part_1.loc[:,~(mask.any(axis=0))].replace(["DNP","-"],"0")
player_stats_24_25_v4_final2 = player_stats_24_25_v4_part_2_filter.apply(pd.to_numeric)

player_stats_24_25_v4_part_to_merge2 = pd.merge(player_stats_24_25_v4_final1,player_stats_24_25_v4_final2,left_index=True,right_index=True) 

player_stats_24_25_final = pd.merge(player_stats_24_25_v4_part_to_merge1,player_stats_24_25_v4_part_to_merge2,left_index=True,right_index=True) 
player_stats_24_25_final = player_stats_24_25_final.replace(np.nan,0)




#datasets we will work now are the _final
'''
player_stats_21_22_final
player_stats_22_23_final    
player_stats_23_24_final
player_stats_24_25_final


player_stats_21_22_final.head()
player_stats_22_23_final.head()    
player_stats_23_24_final.head()
player_stats_24_25_final.head()

player_stats_21_22_final.to_csv('player_stats_21_22_final.csv')
player_stats_22_23_final.to_csv('player_stats_22_23_final.csv')
player_stats_23_24_final.to_csv('player_stats_23_24_final.csv')
player_stats_24_25_final.to_csv('player_stats_24_25_final.csv')
'''



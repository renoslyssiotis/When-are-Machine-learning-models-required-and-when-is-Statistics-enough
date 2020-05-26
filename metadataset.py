import sys, os, pickle
from pathlib import PurePath
current_dir = os.path.realpath(__file__)
p = PurePath(current_dir)
sys.path.append(str(p.parents[0]))

import pandas as pd
#==============================================================================
#==========================Dataset 1===========================================
#==============================================================================
with open('./training/offline/avila/results_roc_avila_0.pickle', 'rb') as handle:
    results_roc_0 = pickle.load(handle)
with open('./training/offline/avila/results_prc_avila_0.pickle', 'rb') as handle:
    results_prc_0 = pickle.load(handle)
with open('./training/offline/avila/results_f1_avila_0.pickle', 'rb') as handle:
    results_f1_0 = pickle.load(handle)
with open('./training/offline/avila/metafeatures_0.pickle', 'rb') as handle:
    meta_features_0 = pickle.load(handle)
    
with open('./training/offline/avila/results_roc_avila_1.pickle', 'rb') as handle:
    results_roc_1 = pickle.load(handle)
with open('./training/offline/avila/results_prc_avila_1.pickle', 'rb') as handle:
    results_prc_1 = pickle.load(handle)
with open('./training/offline/avila/results_f1_avila_1.pickle', 'rb') as handle:
    results_f1_1 = pickle.load(handle)
with open('./training/offline/avila/metafeatures_1.pickle', 'rb') as handle:
    meta_features_1 = pickle.load(handle)    

with open('./training/offline/avila/results_roc_avila_2.pickle', 'rb') as handle:
    results_roc_2 = pickle.load(handle)
with open('./training/offline/avila/results_prc_avila_2.pickle', 'rb') as handle:
    results_prc_2 = pickle.load(handle)
with open('./training/offline/avila/results_f1_avila_2.pickle', 'rb') as handle:
    results_f1_2 = pickle.load(handle)
with open('./training/offline/avila/metafeatures_2.pickle', 'rb') as handle:
    meta_features_2 = pickle.load(handle)    
    
with open('./training/offline/avila/results_roc_avila_3.pickle', 'rb') as handle:
    results_roc_3 = pickle.load(handle)
with open('./training/offline/avila/results_prc_avila_3.pickle', 'rb') as handle:
    results_prc_3 = pickle.load(handle)
with open('./training/offline/avila/results_f1_avila_3.pickle', 'rb') as handle:
    results_f1_3 = pickle.load(handle)
with open('./training/offline/avila/metafeatures_3.pickle', 'rb') as handle:
    meta_features_3 = pickle.load(handle)    
    
with open('./training/offline/avila/results_roc_avila_4.pickle', 'rb') as handle:
    results_roc_4 = pickle.load(handle)
with open('./training/offline/avila/results_prc_avila_4.pickle', 'rb') as handle:
    results_prc_4 = pickle.load(handle)
with open('./training/offline/avila/results_f1_avila_4.pickle', 'rb') as handle:
    results_f1_4 = pickle.load(handle)
with open('./training/offline/avila/metafeatures_4.pickle', 'rb') as handle:
    meta_features_4 = pickle.load(handle)    
    
#==============================================================================
#==========================Dataset 2===========================================
#==============================================================================
with open('./training/offline/banknote/results_roc_banknote_0.pickle', 'rb') as handle:
    results_roc_5 = pickle.load(handle)
with open('./training/offline/banknote/results_prc_banknote_0.pickle', 'rb') as handle:
    results_prc_5 = pickle.load(handle)
with open('./training/offline/banknote/results_f1_banknote_0.pickle', 'rb') as handle:
    results_f1_5 = pickle.load(handle)
with open('./training/offline/banknote/metafeatures_banknote_0.pickle', 'rb') as handle:
    meta_features_5 = pickle.load(handle)
    
with open('./training/offline/banknote/results_roc_banknote_1.pickle', 'rb') as handle:
    results_roc_6 = pickle.load(handle)
with open('./training/offline/banknote/results_prc_banknote_1.pickle', 'rb') as handle:
    results_prc_6 = pickle.load(handle)
with open('./training/offline/banknote/results_f1_banknote_1.pickle', 'rb') as handle:
    results_f1_6 = pickle.load(handle)
with open('./training/offline/banknote/metafeatures_banknote_1.pickle', 'rb') as handle:
    meta_features_6 = pickle.load(handle)
    
with open('./training/offline/banknote/results_roc_banknote_2.pickle', 'rb') as handle:
    results_roc_7 = pickle.load(handle)
with open('./training/offline/banknote/results_prc_banknote_2.pickle', 'rb') as handle:
    results_prc_7 = pickle.load(handle)
with open('./training/offline/banknote/results_f1_banknote_2.pickle', 'rb') as handle:
    results_f1_7 = pickle.load(handle)
with open('./training/offline/banknote/metafeatures_banknote_2.pickle', 'rb') as handle:
    meta_features_7 = pickle.load(handle)
    
with open('./training/offline/banknote/results_roc_banknote_3.pickle', 'rb') as handle:
    results_roc_8 = pickle.load(handle)
with open('./training/offline/banknote/results_prc_banknote_3.pickle', 'rb') as handle:
    results_prc_8 = pickle.load(handle)
with open('./training/offline/banknote/results_f1_banknote_3.pickle', 'rb') as handle:
    results_f1_8 = pickle.load(handle)
with open('./training/offline/banknote/metafeatures_banknote_3.pickle', 'rb') as handle:
    meta_features_8 = pickle.load(handle)
    
with open('./training/offline/banknote/results_roc_banknote_4.pickle', 'rb') as handle:
    results_roc_9 = pickle.load(handle)
with open('./training/offline/banknote/results_prc_banknote_4.pickle', 'rb') as handle:
    results_prc_9 = pickle.load(handle)
with open('./training/offline/banknote/results_f1_banknote_4.pickle', 'rb') as handle:
    results_f1_9 = pickle.load(handle)
with open('./training/offline/banknote/metafeatures_banknote_4.pickle', 'rb') as handle:
    meta_features_9 = pickle.load(handle)
    
#==============================================================================
#==========================Dataset 3===========================================
#==============================================================================
with open('./training/offline/blood_transfusion/results_roc_blood_0.pickle', 'rb') as handle:
    results_roc_10 = pickle.load(handle)
with open('./training/offline/blood_transfusion/results_prc_blood_0.pickle', 'rb') as handle:
    results_prc_10 = pickle.load(handle)
with open('./training/offline/blood_transfusion/results_f1_blood_0.pickle', 'rb') as handle:
    results_f1_10 = pickle.load(handle)
with open('./training/offline/blood_transfusion/metafeatures_blood_0.pickle', 'rb') as handle:
    meta_features_10 = pickle.load(handle)    

with open('./training/offline/blood_transfusion/results_roc_blood_1.pickle', 'rb') as handle:
    results_roc_11 = pickle.load(handle)
with open('./training/offline/blood_transfusion/results_prc_blood_1.pickle', 'rb') as handle:
    results_prc_11 = pickle.load(handle)
with open('./training/offline/blood_transfusion/results_f1_blood_1.pickle', 'rb') as handle:
    results_f1_11 = pickle.load(handle)
with open('./training/offline/blood_transfusion/metafeatures_blood_1.pickle', 'rb') as handle:
    meta_features_11 = pickle.load(handle)    

with open('./training/offline/blood_transfusion/results_roc_blood_2.pickle', 'rb') as handle:
    results_roc_12 = pickle.load(handle)
with open('./training/offline/blood_transfusion/results_prc_blood_2.pickle', 'rb') as handle:
    results_prc_12 = pickle.load(handle)
with open('./training/offline/blood_transfusion/results_f1_blood_2.pickle', 'rb') as handle:
    results_f1_12 = pickle.load(handle)
with open('./training/offline/blood_transfusion/metafeatures_blood_2.pickle', 'rb') as handle:
    meta_features_12 = pickle.load(handle)    

with open('./training/offline/blood_transfusion/results_roc_blood_3.pickle', 'rb') as handle:
    results_roc_13 = pickle.load(handle)
with open('./training/offline/blood_transfusion/results_prc_blood_3.pickle', 'rb') as handle:
    results_prc_13 = pickle.load(handle)
with open('./training/offline/blood_transfusion/results_f1_blood_3.pickle', 'rb') as handle:
    results_f1_13 = pickle.load(handle)
with open('./training/offline/blood_transfusion/metafeatures_blood_3.pickle', 'rb') as handle:
    meta_features_13 = pickle.load(handle)    

with open('./training/offline/blood_transfusion/results_roc_blood_4.pickle', 'rb') as handle:
    results_roc_14 = pickle.load(handle)
with open('./training/offline/blood_transfusion/results_prc_blood_4.pickle', 'rb') as handle:
    results_prc_14 = pickle.load(handle)
with open('./training/offline/blood_transfusion/results_f1_blood_4.pickle', 'rb') as handle:
    results_f1_14 = pickle.load(handle)
with open('./training/offline/blood_transfusion/metafeatures_blood_4.pickle', 'rb') as handle:
    meta_features_14 = pickle.load(handle)        
    
#==============================================================================
#==========================Dataset 4===========================================
#==============================================================================
with open('./training/offline/breast_cancer/results_roc_breastCancer_0.pickle', 'rb') as handle:
    results_roc_15 = pickle.load(handle)
with open('./training/offline/breast_cancer/results_prc_breastCancer_0.pickle', 'rb') as handle:
    results_prc_15 = pickle.load(handle)
with open('./training/offline/breast_cancer/results_f1_breastCancer_0.pickle', 'rb') as handle:
    results_f1_15 = pickle.load(handle)
with open('./training/offline/breast_cancer/metafeatures_breastCancer_0.pickle', 'rb') as handle:
    meta_features_15 = pickle.load(handle)    

with open('./training/offline/breast_cancer/results_roc_breastCancer_1.pickle', 'rb') as handle:
    results_roc_16 = pickle.load(handle)
with open('./training/offline/breast_cancer/results_prc_breastCancer_1.pickle', 'rb') as handle:
    results_prc_16 = pickle.load(handle)
with open('./training/offline/breast_cancer/results_f1_breastCancer_1.pickle', 'rb') as handle:
    results_f1_16 = pickle.load(handle)
with open('./training/offline/breast_cancer/metafeatures_breastCancer_1.pickle', 'rb') as handle:
    meta_features_16 = pickle.load(handle)    
    
with open('./training/offline/breast_cancer/results_roc_breastCancer_2.pickle', 'rb') as handle:
    results_roc_17 = pickle.load(handle)
with open('./training/offline/breast_cancer/results_prc_breastCancer_2.pickle', 'rb') as handle:
    results_prc_17 = pickle.load(handle)
with open('./training/offline/breast_cancer/results_f1_breastCancer_2.pickle', 'rb') as handle:
    results_f1_17 = pickle.load(handle)
with open('./training/offline/breast_cancer/metafeatures_breastCancer_2.pickle', 'rb') as handle:
    meta_features_17 = pickle.load(handle)    
    
with open('./training/offline/breast_cancer/results_roc_breastCancer_3.pickle', 'rb') as handle:
    results_roc_18 = pickle.load(handle)
with open('./training/offline/breast_cancer/results_prc_breastCancer_3.pickle', 'rb') as handle:
    results_prc_18 = pickle.load(handle)
with open('./training/offline/breast_cancer/results_f1_breastCancer_3.pickle', 'rb') as handle:
    results_f1_18 = pickle.load(handle)
with open('./training/offline/breast_cancer/metafeatures_breastCancer_3.pickle', 'rb') as handle:
    meta_features_18 = pickle.load(handle)    
    
with open('./training/offline/breast_cancer/results_roc_breastCancer_4.pickle', 'rb') as handle:
    results_roc_19 = pickle.load(handle)
with open('./training/offline/breast_cancer/results_prc_breastCancer_4.pickle', 'rb') as handle:
    results_prc_19 = pickle.load(handle)
with open('./training/offline/breast_cancer/results_f1_breastCancer_4.pickle', 'rb') as handle:
    results_f1_19 = pickle.load(handle)
with open('./training/offline/breast_cancer/metafeatures_breastCancer_4.pickle', 'rb') as handle:
    meta_features_19 = pickle.load(handle)    
    
#==============================================================================
#==========================Dataset 5===========================================
#==============================================================================
with open('./training/offline/breast_cancer_coimba/results_roc_coimbra_0.pickle', 'rb') as handle:
    results_roc_20 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/results_prc_coimbra_0.pickle', 'rb') as handle:
    results_prc_20 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/results_f1_coimbra_0.pickle', 'rb') as handle:
    results_f1_20 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/metafeatures_coimbra_0.pickle', 'rb') as handle:
    meta_features_20 = pickle.load(handle)    
    
with open('./training/offline/breast_cancer_coimba/results_roc_coimbra_1.pickle', 'rb') as handle:
    results_roc_21 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/results_prc_coimbra_1.pickle', 'rb') as handle:
    results_prc_21 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/results_f1_coimbra_1.pickle', 'rb') as handle:
    results_f1_21 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/metafeatures_coimbra_1.pickle', 'rb') as handle:
    meta_features_21 = pickle.load(handle)    
    
with open('./training/offline/breast_cancer_coimba/results_roc_coimbra_2.pickle', 'rb') as handle:
    results_roc_22 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/results_prc_coimbra_2.pickle', 'rb') as handle:
    results_prc_22 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/results_f1_coimbra_2.pickle', 'rb') as handle:
    results_f1_22 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/metafeatures_coimbra_2.pickle', 'rb') as handle:
    meta_features_22 = pickle.load(handle)    
    
with open('./training/offline/breast_cancer_coimba/results_roc_coimbra_3.pickle', 'rb') as handle:
    results_roc_23 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/results_prc_coimbra_3.pickle', 'rb') as handle:
    results_prc_23 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/results_f1_coimbra_3.pickle', 'rb') as handle:
    results_f1_23 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/metafeatures_coimbra_3.pickle', 'rb') as handle:
    meta_features_23 = pickle.load(handle)    
    
with open('./training/offline/breast_cancer_coimba/results_roc_coimbra_4.pickle', 'rb') as handle:
    results_roc_24 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/results_prc_coimbra_4.pickle', 'rb') as handle:
    results_prc_24 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/results_f1_coimbra_4.pickle', 'rb') as handle:
    results_f1_24 = pickle.load(handle)
with open('./training/offline/breast_cancer_coimba/metafeatures_coimbra_4.pickle', 'rb') as handle:
    meta_features_24 = pickle.load(handle)    
    
#==============================================================================
#==========================Dataset 6===========================================
#==============================================================================
with open('./training/offline/breast_tissue/results_roc_breastTissue_0.pickle', 'rb') as handle:
    results_roc_25 = pickle.load(handle)
with open('./training/offline/breast_tissue/results_prc_breastTissue_0.pickle', 'rb') as handle:
    results_prc_25 = pickle.load(handle)
with open('./training/offline/breast_tissue/results_f1_breastTissue_0.pickle', 'rb') as handle:
    results_f1_25 = pickle.load(handle)
with open('./training/offline/breast_tissue/metafeatures_breastTissue_0.pickle', 'rb') as handle:
    meta_features_25 = pickle.load(handle)   
    
with open('./training/offline/breast_tissue/results_roc_breastTissue_1.pickle', 'rb') as handle:
    results_roc_26 = pickle.load(handle)
with open('./training/offline/breast_tissue/results_prc_breastTissue_1.pickle', 'rb') as handle:
    results_prc_26 = pickle.load(handle)
with open('./training/offline/breast_tissue/results_f1_breastTissue_1.pickle', 'rb') as handle:
    results_f1_26 = pickle.load(handle)
with open('./training/offline/breast_tissue/metafeatures_breastTissue_1.pickle', 'rb') as handle:
    meta_features_26 = pickle.load(handle)   
    
with open('./training/offline/breast_tissue/results_roc_breastTissue_2.pickle', 'rb') as handle:
    results_roc_27 = pickle.load(handle)
with open('./training/offline/breast_tissue/results_prc_breastTissue_2.pickle', 'rb') as handle:
    results_prc_27 = pickle.load(handle)
with open('./training/offline/breast_tissue/results_f1_breastTissue_2.pickle', 'rb') as handle:
    results_f1_27 = pickle.load(handle)
with open('./training/offline/breast_tissue/metafeatures_breastTissue_2.pickle', 'rb') as handle:
    meta_features_27 = pickle.load(handle)   
    
with open('./training/offline/breast_tissue/results_roc_breastTissue_3.pickle', 'rb') as handle:
    results_roc_28 = pickle.load(handle)
with open('./training/offline/breast_tissue/results_prc_breastTissue_3.pickle', 'rb') as handle:
    results_prc_28 = pickle.load(handle)
with open('./training/offline/breast_tissue/results_f1_breastTissue_3.pickle', 'rb') as handle:
    results_f1_28 = pickle.load(handle)
with open('./training/offline/breast_tissue/metafeatures_breastTissue_3.pickle', 'rb') as handle:
    meta_features_28 = pickle.load(handle)   
    
with open('./training/offline/breast_tissue/results_roc_breastTissue_4.pickle', 'rb') as handle:
    results_roc_29 = pickle.load(handle)
with open('./training/offline/breast_tissue/results_prc_breastTissue_4.pickle', 'rb') as handle:
    results_prc_29 = pickle.load(handle)
with open('./training/offline/breast_tissue/results_f1_breastTissue_4.pickle', 'rb') as handle:
    results_f1_29 = pickle.load(handle)
with open('./training/offline/breast_tissue/metafeatures_breastTissue_4.pickle', 'rb') as handle:
    meta_features_29 = pickle.load(handle)   
    
#==============================================================================
#==========================Dataset 7===========================================
#==============================================================================
with open('./training/offline/churn/results_roc_churn_0.pickle', 'rb') as handle:
    results_roc_30 = pickle.load(handle)
with open('./training/offline/churn/results_prc_churn_0.pickle', 'rb') as handle:
    results_prc_30 = pickle.load(handle)
with open('./training/offline/churn/results_f1_churn_0.pickle', 'rb') as handle:
    results_f1_30 = pickle.load(handle)
with open('./training/offline/churn/metafeatures_churn_0.pickle', 'rb') as handle:
    meta_features_30 = pickle.load(handle) 

with open('./training/offline/churn/results_roc_churn_1.pickle', 'rb') as handle:
    results_roc_31 = pickle.load(handle)
with open('./training/offline/churn/results_prc_churn_1.pickle', 'rb') as handle:
    results_prc_31 = pickle.load(handle)
with open('./training/offline/churn/results_f1_churn_1.pickle', 'rb') as handle:
    results_f1_31 = pickle.load(handle)
with open('./training/offline/churn/metafeatures_churn_1.pickle', 'rb') as handle:
    meta_features_31 = pickle.load(handle) 

with open('./training/offline/churn/results_roc_churn_2.pickle', 'rb') as handle:
    results_roc_32 = pickle.load(handle)
with open('./training/offline/churn/results_prc_churn_2.pickle', 'rb') as handle:
    results_prc_32 = pickle.load(handle)
with open('./training/offline/churn/results_f1_churn_2.pickle', 'rb') as handle:
    results_f1_32 = pickle.load(handle)
with open('./training/offline/churn/metafeatures_churn_2.pickle', 'rb') as handle:
    meta_features_32 = pickle.load(handle) 

with open('./training/offline/churn/results_roc_churn_3.pickle', 'rb') as handle:
    results_roc_33 = pickle.load(handle)
with open('./training/offline/churn/results_prc_churn_3.pickle', 'rb') as handle:
    results_prc_33 = pickle.load(handle)
with open('./training/offline/churn/results_f1_churn_3.pickle', 'rb') as handle:
    results_f1_33 = pickle.load(handle)
with open('./training/offline/churn/metafeatures_churn_3.pickle', 'rb') as handle:
    meta_features_33 = pickle.load(handle) 

with open('./training/offline/churn/results_roc_churn_4.pickle', 'rb') as handle:
    results_roc_34 = pickle.load(handle)
with open('./training/offline/churn/results_prc_churn_4.pickle', 'rb') as handle:
    results_prc_34 = pickle.load(handle)
with open('./training/offline/churn/results_f1_churn_4.pickle', 'rb') as handle:
    results_f1_34 = pickle.load(handle)
with open('./training/offline/churn/metafeatures_churn_4.pickle', 'rb') as handle:
    meta_features_34 = pickle.load(handle) 
    
#==============================================================================
#==========================Dataset 8===========================================
#==============================================================================
with open('./training/offline/cleveland/results_roc_cleveland_0.pickle', 'rb') as handle:
    results_roc_35 = pickle.load(handle)
with open('./training/offline/cleveland/results_prc_cleveland_0.pickle', 'rb') as handle:
    results_prc_35 = pickle.load(handle)
with open('./training/offline/cleveland/results_f1_cleveland_0.pickle', 'rb') as handle:
    results_f1_35 = pickle.load(handle)
with open('./training/offline/cleveland/metafeatures_cleveland_0.pickle', 'rb') as handle:
    meta_features_35 = pickle.load(handle) 
    
with open('./training/offline/cleveland/results_roc_cleveland_1.pickle', 'rb') as handle:
    results_roc_36 = pickle.load(handle)
with open('./training/offline/cleveland/results_prc_cleveland_1.pickle', 'rb') as handle:
    results_prc_36 = pickle.load(handle)
with open('./training/offline/cleveland/results_f1_cleveland_1.pickle', 'rb') as handle:
    results_f1_36 = pickle.load(handle)
with open('./training/offline/cleveland/metafeatures_cleveland_1.pickle', 'rb') as handle:
    meta_features_36 = pickle.load(handle) 
    
with open('./training/offline/cleveland/results_roc_cleveland_2.pickle', 'rb') as handle:
    results_roc_37 = pickle.load(handle)
with open('./training/offline/cleveland/results_prc_cleveland_2.pickle', 'rb') as handle:
    results_prc_37 = pickle.load(handle)
with open('./training/offline/cleveland/results_f1_cleveland_2.pickle', 'rb') as handle:
    results_f1_37 = pickle.load(handle)
with open('./training/offline/cleveland/metafeatures_cleveland_2.pickle', 'rb') as handle:
    meta_features_37 = pickle.load(handle) 
    
with open('./training/offline/cleveland/results_roc_cleveland_3.pickle', 'rb') as handle:
    results_roc_38 = pickle.load(handle)
with open('./training/offline/cleveland/results_prc_cleveland_3.pickle', 'rb') as handle:
    results_prc_38 = pickle.load(handle)
with open('./training/offline/cleveland/results_f1_cleveland_3.pickle', 'rb') as handle:
    results_f1_38 = pickle.load(handle)
with open('./training/offline/cleveland/metafeatures_cleveland_3.pickle', 'rb') as handle:
    meta_features_38 = pickle.load(handle) 
    
with open('./training/offline/cleveland/results_roc_cleveland_4.pickle', 'rb') as handle:
    results_roc_39 = pickle.load(handle)
with open('./training/offline/cleveland/results_prc_cleveland_4.pickle', 'rb') as handle:
    results_prc_39 = pickle.load(handle)
with open('./training/offline/cleveland/results_f1_cleveland_4.pickle', 'rb') as handle:
    results_f1_39 = pickle.load(handle)
with open('./training/offline/cleveland/metafeatures_cleveland_4.pickle', 'rb') as handle:
    meta_features_39 = pickle.load(handle) 
    
#==============================================================================
#==========================Dataset 9===========================================
#==============================================================================
with open('./training/offline/cryotherapy/results_roc_cryo_0.pickle', 'rb') as handle:
    results_roc_40 = pickle.load(handle)
with open('./training/offline/cryotherapy/results_prc_cryo_0.pickle', 'rb') as handle:
    results_prc_40 = pickle.load(handle)
with open('./training/offline/cryotherapy/results_f1_cryo_0.pickle', 'rb') as handle:
    results_f1_40 = pickle.load(handle)
with open('./training/offline/cryotherapy/metafeatures_cryo_0.pickle', 'rb') as handle:
    meta_features_40 = pickle.load(handle) 
    
with open('./training/offline/cryotherapy/results_roc_cryo_1.pickle', 'rb') as handle:
    results_roc_41 = pickle.load(handle)
with open('./training/offline/cryotherapy/results_prc_cryo_1.pickle', 'rb') as handle:
    results_prc_41 = pickle.load(handle)
with open('./training/offline/cryotherapy/results_f1_cryo_1.pickle', 'rb') as handle:
    results_f1_41 = pickle.load(handle)
with open('./training/offline/cryotherapy/metafeatures_cryo_1.pickle', 'rb') as handle:
    meta_features_41 = pickle.load(handle) 
    
with open('./training/offline/cryotherapy/results_roc_cryo_2.pickle', 'rb') as handle:
    results_roc_42 = pickle.load(handle)
with open('./training/offline/cryotherapy/results_prc_cryo_2.pickle', 'rb') as handle:
    results_prc_42 = pickle.load(handle)
with open('./training/offline/cryotherapy/results_f1_cryo_2.pickle', 'rb') as handle:
    results_f1_42 = pickle.load(handle)
with open('./training/offline/cryotherapy/metafeatures_cryo_2.pickle', 'rb') as handle:
    meta_features_42 = pickle.load(handle) 
    
with open('./training/offline/cryotherapy/results_roc_cryo_3.pickle', 'rb') as handle:
    results_roc_43 = pickle.load(handle)
with open('./training/offline/cryotherapy/results_prc_cryo_3.pickle', 'rb') as handle:
    results_prc_43 = pickle.load(handle)
with open('./training/offline/cryotherapy/results_f1_cryo_3.pickle', 'rb') as handle:
    results_f1_43 = pickle.load(handle)
with open('./training/offline/cryotherapy/metafeatures_cryo_3.pickle', 'rb') as handle:
    meta_features_43 = pickle.load(handle) 
    
with open('./training/offline/cryotherapy/results_roc_cryo_4.pickle', 'rb') as handle:
    results_roc_44 = pickle.load(handle)
with open('./training/offline/cryotherapy/results_prc_cryo_4.pickle', 'rb') as handle:
    results_prc_44 = pickle.load(handle)
with open('./training/offline/cryotherapy/results_f1_cryo_4.pickle', 'rb') as handle:
    results_f1_44 = pickle.load(handle)
with open('./training/offline/cryotherapy/metafeatures_cryo_4.pickle', 'rb') as handle:
    meta_features_44 = pickle.load(handle) 
    
#==============================================================================
#==========================Dataset 10==========================================
#==============================================================================
with open('./training/offline/dermatology/results_roc_derma_0.pickle', 'rb') as handle:
    results_roc_45 = pickle.load(handle)
with open('./training/offline/dermatology/results_prc_derma_0.pickle', 'rb') as handle:
    results_prc_45 = pickle.load(handle)
with open('./training/offline/dermatology/results_f1_derma_0.pickle', 'rb') as handle:
    results_f1_45 = pickle.load(handle)
with open('./training/offline/dermatology/metafeatures_derma_0.pickle', 'rb') as handle:
    meta_features_45 = pickle.load(handle) 
    
with open('./training/offline/dermatology/results_roc_derma_1.pickle', 'rb') as handle:
    results_roc_46 = pickle.load(handle)
with open('./training/offline/dermatology/results_prc_derma_1.pickle', 'rb') as handle:
    results_prc_46 = pickle.load(handle)
with open('./training/offline/dermatology/results_f1_derma_1.pickle', 'rb') as handle:
    results_f1_46 = pickle.load(handle)
with open('./training/offline/dermatology/metafeatures_derma_1.pickle', 'rb') as handle:
    meta_features_46 = pickle.load(handle) 
    
with open('./training/offline/dermatology/results_roc_derma_2.pickle', 'rb') as handle:
    results_roc_47 = pickle.load(handle)
with open('./training/offline/dermatology/results_prc_derma_2.pickle', 'rb') as handle:
    results_prc_47 = pickle.load(handle)
with open('./training/offline/dermatology/results_f1_derma_2.pickle', 'rb') as handle:
    results_f1_47 = pickle.load(handle)
with open('./training/offline/dermatology/metafeatures_derma_2.pickle', 'rb') as handle:
    meta_features_47 = pickle.load(handle) 
    
with open('./training/offline/dermatology/results_roc_derma_3.pickle', 'rb') as handle:
    results_roc_48 = pickle.load(handle)
with open('./training/offline/dermatology/results_prc_derma_3.pickle', 'rb') as handle:
    results_prc_48 = pickle.load(handle)
with open('./training/offline/dermatology/results_f1_derma_3.pickle', 'rb') as handle:
    results_f1_48 = pickle.load(handle)
with open('./training/offline/dermatology/metafeatures_derma_3.pickle', 'rb') as handle:
    meta_features_48 = pickle.load(handle) 
    
with open('./training/offline/dermatology/results_roc_derma_4.pickle', 'rb') as handle:
    results_roc_49 = pickle.load(handle)
with open('./training/offline/dermatology/results_prc_derma_4.pickle', 'rb') as handle:
    results_prc_49 = pickle.load(handle)
with open('./training/offline/dermatology/results_f1_derma_4.pickle', 'rb') as handle:
    results_f1_49 = pickle.load(handle)
with open('./training/offline/dermatology/metafeatures_derma_4.pickle', 'rb') as handle:
    meta_features_49 = pickle.load(handle) 
    
#==============================================================================
#==========================Dataset 11==========================================
#==============================================================================
with open('./training/offline/fertility/results_roc_fertility_0.pickle', 'rb') as handle:
    results_roc_50 = pickle.load(handle)
with open('./training/offline/fertility/results_prc_fertility_0.pickle', 'rb') as handle:
    results_prc_50 = pickle.load(handle)
with open('./training/offline/fertility/results_f1_fertility_0.pickle', 'rb') as handle:
    results_f1_50 = pickle.load(handle)
with open('./training/offline/fertility/metafeatures_fertility_0.pickle', 'rb') as handle:
    meta_features_50 = pickle.load(handle)    
    
with open('./training/offline/fertility/results_roc_fertility_1.pickle', 'rb') as handle:
    results_roc_51 = pickle.load(handle)
with open('./training/offline/fertility/results_prc_fertility_1.pickle', 'rb') as handle:
    results_prc_51 = pickle.load(handle)
with open('./training/offline/fertility/results_f1_fertility_1.pickle', 'rb') as handle:
    results_f1_51 = pickle.load(handle)
with open('./training/offline/fertility/metafeatures_fertility_1.pickle', 'rb') as handle:
    meta_features_51 = pickle.load(handle)    
    
with open('./training/offline/fertility/results_roc_fertility_2.pickle', 'rb') as handle:
    results_roc_52 = pickle.load(handle)
with open('./training/offline/fertility/results_prc_fertility_2.pickle', 'rb') as handle:
    results_prc_52 = pickle.load(handle)
with open('./training/offline/fertility/results_f1_fertility_2.pickle', 'rb') as handle:
    results_f1_52 = pickle.load(handle)
with open('./training/offline/fertility/metafeatures_fertility_2.pickle', 'rb') as handle:
    meta_features_52 = pickle.load(handle)    
    
with open('./training/offline/fertility/results_roc_fertility_3.pickle', 'rb') as handle:
    results_roc_53 = pickle.load(handle)
with open('./training/offline/fertility/results_prc_fertility_3.pickle', 'rb') as handle:
    results_prc_53 = pickle.load(handle)
with open('./training/offline/fertility/results_f1_fertility_3.pickle', 'rb') as handle:
    results_f1_53 = pickle.load(handle)
with open('./training/offline/fertility/metafeatures_fertility_3.pickle', 'rb') as handle:
    meta_features_53 = pickle.load(handle)    
    
with open('./training/offline/fertility/results_roc_fertility_4.pickle', 'rb') as handle:
    results_roc_54 = pickle.load(handle)
with open('./training/offline/fertility/results_prc_fertility_4.pickle', 'rb') as handle:
    results_prc_54 = pickle.load(handle)
with open('./training/offline/fertility/results_f1_fertility_4.pickle', 'rb') as handle:
    results_f1_54 = pickle.load(handle)
with open('./training/offline/fertility/metafeatures_fertility_4.pickle', 'rb') as handle:
    meta_features_54 = pickle.load(handle)    
    
#==============================================================================
#==========================Dataset 12==========================================
#==============================================================================
with open('./training/offline/frogs/results_roc_frogs_0.pickle', 'rb') as handle:
    results_roc_55 = pickle.load(handle)
with open('./training/offline/frogs/results_prc_frogs_0.pickle', 'rb') as handle:
    results_prc_55 = pickle.load(handle)
with open('./training/offline/frogs/results_f1_frogs_0.pickle', 'rb') as handle:
    results_f1_55 = pickle.load(handle)
with open('./training/offline/frogs/metafeatures_frogs_0.pickle', 'rb') as handle:
    meta_features_55 = pickle.load(handle)  
    
with open('./training/offline/frogs/results_roc_frogs_1.pickle', 'rb') as handle:
    results_roc_56 = pickle.load(handle)
with open('./training/offline/frogs/results_prc_frogs_1.pickle', 'rb') as handle:
    results_prc_56 = pickle.load(handle)
with open('./training/offline/frogs/results_f1_frogs_1.pickle', 'rb') as handle:
    results_f1_56 = pickle.load(handle)
with open('./training/offline/frogs/metafeatures_frogs_1.pickle', 'rb') as handle:
    meta_features_56 = pickle.load(handle)  
    
with open('./training/offline/frogs/results_roc_frogs_2.pickle', 'rb') as handle:
    results_roc_57 = pickle.load(handle)
with open('./training/offline/frogs/results_prc_frogs_2.pickle', 'rb') as handle:
    results_prc_57 = pickle.load(handle)
with open('./training/offline/frogs/results_f1_frogs_2.pickle', 'rb') as handle:
    results_f1_57 = pickle.load(handle)
with open('./training/offline/frogs/metafeatures_frogs_2.pickle', 'rb') as handle:
    meta_features_57 = pickle.load(handle)  
    
with open('./training/offline/frogs/results_roc_frogs_3.pickle', 'rb') as handle:
    results_roc_58 = pickle.load(handle)
with open('./training/offline/frogs/results_prc_frogs_3.pickle', 'rb') as handle:
    results_prc_58 = pickle.load(handle)
with open('./training/offline/frogs/results_f1_frogs_3.pickle', 'rb') as handle:
    results_f1_58 = pickle.load(handle)
with open('./training/offline/frogs/metafeatures_frogs_3.pickle', 'rb') as handle:
    meta_features_58 = pickle.load(handle)  
    
with open('./training/offline/frogs/results_roc_frogs_4.pickle', 'rb') as handle:
    results_roc_59 = pickle.load(handle)
with open('./training/offline/frogs/results_prc_frogs_4.pickle', 'rb') as handle:
    results_prc_59 = pickle.load(handle)
with open('./training/offline/frogs/results_f1_frogs_4.pickle', 'rb') as handle:
    results_f1_59 = pickle.load(handle)
with open('./training/offline/frogs/metafeatures_frogs_4.pickle', 'rb') as handle:
    meta_features_59 = pickle.load(handle)  
    
#==============================================================================
#==========================Dataset 13==========================================
#==============================================================================
with open('./training/offline/immunotherapy/results_roc_immunotherapy_0.pickle', 'rb') as handle:
    results_roc_60 = pickle.load(handle)
with open('./training/offline/immunotherapy/results_prc_immunotherapy_0.pickle', 'rb') as handle:
    results_prc_60 = pickle.load(handle)
with open('./training/offline/immunotherapy/results_f1_immunotherapy_0.pickle', 'rb') as handle:
    results_f1_60 = pickle.load(handle)
with open('./training/offline/immunotherapy/metafeatures_immunotherapy_0.pickle', 'rb') as handle:
    meta_features_60 = pickle.load(handle)   
    
with open('./training/offline/immunotherapy/results_roc_immunotherapy_1.pickle', 'rb') as handle:
    results_roc_61 = pickle.load(handle)
with open('./training/offline/immunotherapy/results_prc_immunotherapy_1.pickle', 'rb') as handle:
    results_prc_61 = pickle.load(handle)
with open('./training/offline/immunotherapy/results_f1_immunotherapy_1.pickle', 'rb') as handle:
    results_f1_61 = pickle.load(handle)
with open('./training/offline/immunotherapy/metafeatures_immunotherapy_1.pickle', 'rb') as handle:
    meta_features_61 = pickle.load(handle)   
    
with open('./training/offline/immunotherapy/results_roc_immunotherapy_2.pickle', 'rb') as handle:
    results_roc_62 = pickle.load(handle)
with open('./training/offline/immunotherapy/results_prc_immunotherapy_2.pickle', 'rb') as handle:
    results_prc_62 = pickle.load(handle)
with open('./training/offline/immunotherapy/results_f1_immunotherapy_2.pickle', 'rb') as handle:
    results_f1_62 = pickle.load(handle)
with open('./training/offline/immunotherapy/metafeatures_immunotherapy_2.pickle', 'rb') as handle:
    meta_features_62 = pickle.load(handle)   
    
with open('./training/offline/immunotherapy/results_roc_immunotherapy_3.pickle', 'rb') as handle:
    results_roc_63 = pickle.load(handle)
with open('./training/offline/immunotherapy/results_prc_immunotherapy_3.pickle', 'rb') as handle:
    results_prc_63 = pickle.load(handle)
with open('./training/offline/immunotherapy/results_f1_immunotherapy_3.pickle', 'rb') as handle:
    results_f1_63 = pickle.load(handle)
with open('./training/offline/immunotherapy/metafeatures_immunotherapy_3.pickle', 'rb') as handle:
    meta_features_63 = pickle.load(handle)   
    
with open('./training/offline/immunotherapy/results_roc_immunotherapy_4.pickle', 'rb') as handle:
    results_roc_64 = pickle.load(handle)
with open('./training/offline/immunotherapy/results_prc_immunotherapy_4.pickle', 'rb') as handle:
    results_prc_64 = pickle.load(handle)
with open('./training/offline/immunotherapy/results_f1_immunotherapy_4.pickle', 'rb') as handle:
    results_f1_64 = pickle.load(handle)
with open('./training/offline/immunotherapy/metafeatures_immunotherapy_4.pickle', 'rb') as handle:
    meta_features_64 = pickle.load(handle)   
    
#==============================================================================
#==========================Dataset 14==========================================
#==============================================================================
with open('./training/offline/indians/results_roc_indians_0.pickle', 'rb') as handle:
    results_roc_65 = pickle.load(handle)
with open('./training/offline/indians/results_prc_indians_0.pickle', 'rb') as handle:
    results_prc_65 = pickle.load(handle)
with open('./training/offline/indians/results_f1_indians_0.pickle', 'rb') as handle:
    results_f1_65 = pickle.load(handle)
with open('./training/offline/indians/metafeatures_indians_0.pickle', 'rb') as handle:
    meta_features_65 = pickle.load(handle)   
    
with open('./training/offline/indians/results_roc_indians_1.pickle', 'rb') as handle:
    results_roc_66 = pickle.load(handle)
with open('./training/offline/indians/results_prc_indians_1.pickle', 'rb') as handle:
    results_prc_66 = pickle.load(handle)
with open('./training/offline/indians/results_f1_indians_1.pickle', 'rb') as handle:
    results_f1_66 = pickle.load(handle)
with open('./training/offline/indians/metafeatures_indians_1.pickle', 'rb') as handle:
    meta_features_66 = pickle.load(handle)   
    
with open('./training/offline/indians/results_roc_indians_2.pickle', 'rb') as handle:
    results_roc_67 = pickle.load(handle)
with open('./training/offline/indians/results_prc_indians_2.pickle', 'rb') as handle:
    results_prc_67 = pickle.load(handle)
with open('./training/offline/indians/results_f1_indians_2.pickle', 'rb') as handle:
    results_f1_67 = pickle.load(handle)
with open('./training/offline/indians/metafeatures_indians_2.pickle', 'rb') as handle:
    meta_features_67 = pickle.load(handle)   
    
with open('./training/offline/indians/results_roc_indians_3.pickle', 'rb') as handle:
    results_roc_68 = pickle.load(handle)
with open('./training/offline/indians/results_prc_indians_3.pickle', 'rb') as handle:
    results_prc_68 = pickle.load(handle)
with open('./training/offline/indians/results_f1_indians_3.pickle', 'rb') as handle:
    results_f1_68 = pickle.load(handle)
with open('./training/offline/indians/metafeatures_indians_3.pickle', 'rb') as handle:
    meta_features_68 = pickle.load(handle)   
    
with open('./training/offline/indians/results_roc_indians_4.pickle', 'rb') as handle:
    results_roc_69 = pickle.load(handle)
with open('./training/offline/indians/results_prc_indians_4.pickle', 'rb') as handle:
    results_prc_69 = pickle.load(handle)
with open('./training/offline/indians/results_f1_indians_4.pickle', 'rb') as handle:
    results_f1_69 = pickle.load(handle)
with open('./training/offline/indians/metafeatures_indians_4.pickle', 'rb') as handle:
    meta_features_69 = pickle.load(handle)   
    
#==============================================================================
#==========================Dataset 15==========================================
#==============================================================================
with open('./training/offline/movies/results_roc_movies_0.pickle', 'rb') as handle:
    results_roc_70 = pickle.load(handle)
with open('./training/offline/movies/results_prc_movies_0.pickle', 'rb') as handle:
    results_prc_70 = pickle.load(handle)
with open('./training/offline/movies/results_f1_movies_0.pickle', 'rb') as handle:
    results_f1_70 = pickle.load(handle)
with open('./training/offline/movies/metafeatures_movies_0.pickle', 'rb') as handle:
    meta_features_70 = pickle.load(handle)    

with open('./training/offline/movies/results_roc_movies_1.pickle', 'rb') as handle:
    results_roc_71 = pickle.load(handle)
with open('./training/offline/movies/results_prc_movies_1.pickle', 'rb') as handle:
    results_prc_71 = pickle.load(handle)
with open('./training/offline/movies/results_f1_movies_1.pickle', 'rb') as handle:
    results_f1_71 = pickle.load(handle)
with open('./training/offline/movies/metafeatures_movies_1.pickle', 'rb') as handle:
    meta_features_71 = pickle.load(handle)    

with open('./training/offline/movies/results_roc_movies_2.pickle', 'rb') as handle:
    results_roc_72 = pickle.load(handle)
with open('./training/offline/movies/results_prc_movies_2.pickle', 'rb') as handle:
    results_prc_72 = pickle.load(handle)
with open('./training/offline/movies/results_f1_movies_2.pickle', 'rb') as handle:
    results_f1_72 = pickle.load(handle)
with open('./training/offline/movies/metafeatures_movies_2.pickle', 'rb') as handle:
    meta_features_72 = pickle.load(handle)    

with open('./training/offline/movies/results_roc_movies_3.pickle', 'rb') as handle:
    results_roc_73 = pickle.load(handle)
with open('./training/offline/movies/results_prc_movies_3.pickle', 'rb') as handle:
    results_prc_73 = pickle.load(handle)
with open('./training/offline/movies/results_f1_movies_3.pickle', 'rb') as handle:
    results_f1_73 = pickle.load(handle)
with open('./training/offline/movies/metafeatures_movies_3.pickle', 'rb') as handle:
    meta_features_73 = pickle.load(handle)    

with open('./training/offline/movies/results_roc_movies_4.pickle', 'rb') as handle:
    results_roc_74 = pickle.load(handle)
with open('./training/offline/movies/results_prc_movies_4.pickle', 'rb') as handle:
    results_prc_74 = pickle.load(handle)
with open('./training/offline/movies/results_f1_movies_4.pickle', 'rb') as handle:
    results_f1_74 = pickle.load(handle)
with open('./training/offline/movies/metafeatures_movies_4.pickle', 'rb') as handle:
    meta_features_74 = pickle.load(handle)    
    
#==============================================================================
#==========================Dataset 16==========================================
#==============================================================================
with open('./training/offline/skin_segmentation/results_roc_skin_0.pickle', 'rb') as handle:
    results_roc_75 = pickle.load(handle)
with open('./training/offline/skin_segmentation/results_prc_skin_0.pickle', 'rb') as handle:
    results_prc_75 = pickle.load(handle)
with open('./training/offline/skin_segmentation/results_f1_skin_0.pickle', 'rb') as handle:
    results_f1_75 = pickle.load(handle)
with open('./training/offline/skin_segmentation/metafeatures_skin_0.pickle', 'rb') as handle:
    meta_features_75 = pickle.load(handle)        

with open('./training/offline/skin_segmentation/results_roc_skin_1.pickle', 'rb') as handle:
    results_roc_76 = pickle.load(handle)
with open('./training/offline/skin_segmentation/results_prc_skin_1.pickle', 'rb') as handle:
    results_prc_76 = pickle.load(handle)
with open('./training/offline/skin_segmentation/results_f1_skin_1.pickle', 'rb') as handle:
    results_f1_76 = pickle.load(handle)
with open('./training/offline/skin_segmentation/metafeatures_skin_1.pickle', 'rb') as handle:
    meta_features_76 = pickle.load(handle)        

with open('./training/offline/skin_segmentation/results_roc_skin_2.pickle', 'rb') as handle:
    results_roc_77 = pickle.load(handle)
with open('./training/offline/skin_segmentation/results_prc_skin_2.pickle', 'rb') as handle:
    results_prc_77 = pickle.load(handle)
with open('./training/offline/skin_segmentation/results_f1_skin_2.pickle', 'rb') as handle:
    results_f1_77 = pickle.load(handle)
with open('./training/offline/skin_segmentation/metafeatures_skin_2.pickle', 'rb') as handle:
    meta_features_77 = pickle.load(handle)        

with open('./training/offline/skin_segmentation/results_roc_skin_3.pickle', 'rb') as handle:
    results_roc_78 = pickle.load(handle)
with open('./training/offline/skin_segmentation/results_prc_skin_3.pickle', 'rb') as handle:
    results_prc_78 = pickle.load(handle)
with open('./training/offline/skin_segmentation/results_f1_skin_3.pickle', 'rb') as handle:
    results_f1_78 = pickle.load(handle)
with open('./training/offline/skin_segmentation/metafeatures_skin_3.pickle', 'rb') as handle:
    meta_features_78 = pickle.load(handle)        

with open('./training/offline/skin_segmentation/results_roc_skin_4.pickle', 'rb') as handle:
    results_roc_79 = pickle.load(handle)
with open('./training/offline/skin_segmentation/results_prc_skin_4.pickle', 'rb') as handle:
    results_prc_79 = pickle.load(handle)
with open('./training/offline/skin_segmentation/results_f1_skin_4.pickle', 'rb') as handle:
    results_f1_79 = pickle.load(handle)
with open('./training/offline/skin_segmentation/metafeatures_skin_4.pickle', 'rb') as handle:
    meta_features_79 = pickle.load(handle)        

#==============================================================================
#==========================Dataset 17==========================================
#==============================================================================
with open('./training/offline/sonar/results_roc_sonar_0.pickle', 'rb') as handle:
    results_roc_80 = pickle.load(handle)
with open('./training/offline/sonar/results_prc_sonar_0.pickle', 'rb') as handle:
    results_prc_80 = pickle.load(handle)
with open('./training/offline/sonar/results_f1_sonar_0.pickle', 'rb') as handle:
    results_f1_80 = pickle.load(handle)
with open('./training/offline/sonar/metafeatures_sonar_0.pickle', 'rb') as handle:
    meta_features_80 = pickle.load(handle)   

with open('./training/offline/sonar/results_roc_sonar_1.pickle', 'rb') as handle:
    results_roc_81 = pickle.load(handle)
with open('./training/offline/sonar/results_prc_sonar_1.pickle', 'rb') as handle:
    results_prc_81 = pickle.load(handle)
with open('./training/offline/sonar/results_f1_sonar_1.pickle', 'rb') as handle:
    results_f1_81 = pickle.load(handle)
with open('./training/offline/sonar/metafeatures_sonar_1.pickle', 'rb') as handle:
    meta_features_81 = pickle.load(handle)   
    
with open('./training/offline/sonar/results_roc_sonar_2.pickle', 'rb') as handle:
    results_roc_82 = pickle.load(handle)
with open('./training/offline/sonar/results_prc_sonar_2.pickle', 'rb') as handle:
    results_prc_82 = pickle.load(handle)
with open('./training/offline/sonar/results_f1_sonar_2.pickle', 'rb') as handle:
    results_f1_82 = pickle.load(handle)
with open('./training/offline/sonar/metafeatures_sonar_2.pickle', 'rb') as handle:
    meta_features_82 = pickle.load(handle)   
    
with open('./training/offline/sonar/results_roc_sonar_3.pickle', 'rb') as handle:
    results_roc_83 = pickle.load(handle)
with open('./training/offline/sonar/results_prc_sonar_3.pickle', 'rb') as handle:
    results_prc_83 = pickle.load(handle)
with open('./training/offline/sonar/results_f1_sonar_3.pickle', 'rb') as handle:
    results_f1_83 = pickle.load(handle)
with open('./training/offline/sonar/metafeatures_sonar_3.pickle', 'rb') as handle:
    meta_features_83 = pickle.load(handle)   
    
with open('./training/offline/sonar/results_roc_sonar_4.pickle', 'rb') as handle:
    results_roc_84 = pickle.load(handle)
with open('./training/offline/sonar/results_prc_sonar_4.pickle', 'rb') as handle:
    results_prc_84 = pickle.load(handle)
with open('./training/offline/sonar/results_f1_sonar_4.pickle', 'rb') as handle:
    results_f1_84 = pickle.load(handle)
with open('./training/offline/sonar/metafeatures_sonar_4.pickle', 'rb') as handle:
    meta_features_84 = pickle.load(handle)   

#==============================================================================
#==========================Dataset 18==========================================
#==============================================================================
with open('./training/offline/spam/results_roc_spam_0.pickle', 'rb') as handle:
    results_roc_85 = pickle.load(handle)
with open('./training/offline/spam/results_prc_spam_0.pickle', 'rb') as handle:
    results_prc_85 = pickle.load(handle)
with open('./training/offline/spam/results_f1_spam_0.pickle', 'rb') as handle:
    results_f1_85 = pickle.load(handle)
with open('./training/offline/spam/metafeatures_spam_0.pickle', 'rb') as handle:
    meta_features_85 = pickle.load(handle)   

with open('./training/offline/spam/results_roc_spam_1.pickle', 'rb') as handle:
    results_roc_86 = pickle.load(handle)
with open('./training/offline/spam/results_prc_spam_1.pickle', 'rb') as handle:
    results_prc_86 = pickle.load(handle)
with open('./training/offline/spam/results_f1_spam_1.pickle', 'rb') as handle:
    results_f1_86 = pickle.load(handle)
with open('./training/offline/spam/metafeatures_spam_1.pickle', 'rb') as handle:
    meta_features_86 = pickle.load(handle)   
    
with open('./training/offline/spam/results_roc_spam_2.pickle', 'rb') as handle:
    results_roc_87 = pickle.load(handle)
with open('./training/offline/spam/results_prc_spam_2.pickle', 'rb') as handle:
    results_prc_87 = pickle.load(handle)
with open('./training/offline/spam/results_f1_spam_2.pickle', 'rb') as handle:
    results_f1_87 = pickle.load(handle)
with open('./training/offline/spam/metafeatures_spam_2.pickle', 'rb') as handle:
    meta_features_87 = pickle.load(handle)   
    
with open('./training/offline/spam/results_roc_spam_3.pickle', 'rb') as handle:
    results_roc_88 = pickle.load(handle)
with open('./training/offline/spam/results_prc_spam_3.pickle', 'rb') as handle:
    results_prc_88 = pickle.load(handle)
with open('./training/offline/spam/results_f1_spam_3.pickle', 'rb') as handle:
    results_f1_88 = pickle.load(handle)
with open('./training/offline/spam/metafeatures_spam_3.pickle', 'rb') as handle:
    meta_features_88 = pickle.load(handle)   
    
with open('./training/offline/spam/results_roc_spam_4.pickle', 'rb') as handle:
    results_roc_89 = pickle.load(handle)
with open('./training/offline/spam/results_prc_spam_4.pickle', 'rb') as handle:
    results_prc_89 = pickle.load(handle)
with open('./training/offline/spam/results_f1_spam_4.pickle', 'rb') as handle:
    results_f1_89 = pickle.load(handle)
with open('./training/offline/spam/metafeatures_spam_4.pickle', 'rb') as handle:
    meta_features_89 = pickle.load(handle)   

#==============================================================================
#==========================Dataset 19==========================================
#==============================================================================
with open('./training/offline/thoracic/results_roc_thoracic_0.pickle', 'rb') as handle:
    results_roc_90 = pickle.load(handle)
with open('./training/offline/thoracic/results_prc_thoracic_0.pickle', 'rb') as handle:
    results_prc_90 = pickle.load(handle)
with open('./training/offline/thoracic/results_f1_thoracic_0.pickle', 'rb') as handle:
    results_f1_90 = pickle.load(handle)
with open('./training/offline/thoracic/metafeatures_thoracic_0.pickle', 'rb') as handle:
    meta_features_90 = pickle.load(handle)     

with open('./training/offline/thoracic/results_roc_thoracic_1.pickle', 'rb') as handle:
    results_roc_91 = pickle.load(handle)
with open('./training/offline/thoracic/results_prc_thoracic_1.pickle', 'rb') as handle:
    results_prc_91 = pickle.load(handle)
with open('./training/offline/thoracic/results_f1_thoracic_1.pickle', 'rb') as handle:
    results_f1_91 = pickle.load(handle)
with open('./training/offline/thoracic/metafeatures_thoracic_1.pickle', 'rb') as handle:
    meta_features_91 = pickle.load(handle)     

with open('./training/offline/thoracic/results_roc_thoracic_2.pickle', 'rb') as handle:
    results_roc_92 = pickle.load(handle)
with open('./training/offline/thoracic/results_prc_thoracic_2.pickle', 'rb') as handle:
    results_prc_92 = pickle.load(handle)
with open('./training/offline/thoracic/results_f1_thoracic_2.pickle', 'rb') as handle:
    results_f1_92 = pickle.load(handle)
with open('./training/offline/thoracic/metafeatures_thoracic_2.pickle', 'rb') as handle:
    meta_features_92 = pickle.load(handle)     

with open('./training/offline/thoracic/results_roc_thoracic_3.pickle', 'rb') as handle:
    results_roc_93 = pickle.load(handle)
with open('./training/offline/thoracic/results_prc_thoracic_3.pickle', 'rb') as handle:
    results_prc_93 = pickle.load(handle)
with open('./training/offline/thoracic/results_f1_thoracic_3.pickle', 'rb') as handle:
    results_f1_93 = pickle.load(handle)
with open('./training/offline/thoracic/metafeatures_thoracic_3.pickle', 'rb') as handle:
    meta_features_93 = pickle.load(handle)     

with open('./training/offline/thoracic/results_roc_thoracic_4.pickle', 'rb') as handle:
    results_roc_94 = pickle.load(handle)
with open('./training/offline/thoracic/results_prc_thoracic_4.pickle', 'rb') as handle:
    results_prc_94 = pickle.load(handle)
with open('./training/offline/thoracic/results_f1_thoracic_4.pickle', 'rb') as handle:
    results_f1_94 = pickle.load(handle)
with open('./training/offline/thoracic/metafeatures_thoracic_4.pickle', 'rb') as handle:
    meta_features_94 = pickle.load(handle)     
    
#==============================================================================
#==========================Dataset 20==========================================
#==============================================================================
with open('./training/offline/vertebral/results_roc_vertebral_0.pickle', 'rb') as handle:
    results_roc_95 = pickle.load(handle)
with open('./training/offline/vertebral/results_prc_vertebral_0.pickle', 'rb') as handle:
    results_prc_95 = pickle.load(handle)
with open('./training/offline/vertebral/results_f1_vertebral_0.pickle', 'rb') as handle:
    results_f1_95 = pickle.load(handle)
with open('./training/offline/vertebral/metafeatures_vertebral_0.pickle', 'rb') as handle:
    meta_features_95 = pickle.load(handle)  

with open('./training/offline/vertebral/results_roc_vertebral_1.pickle', 'rb') as handle:
    results_roc_96 = pickle.load(handle)
with open('./training/offline/vertebral/results_prc_vertebral_1.pickle', 'rb') as handle:
    results_prc_96 = pickle.load(handle)
with open('./training/offline/vertebral/results_f1_vertebral_1.pickle', 'rb') as handle:
    results_f1_96 = pickle.load(handle)
with open('./training/offline/vertebral/metafeatures_vertebral_1.pickle', 'rb') as handle:
    meta_features_96 = pickle.load(handle)  

with open('./training/offline/vertebral/results_roc_vertebral_2.pickle', 'rb') as handle:
    results_roc_97 = pickle.load(handle)
with open('./training/offline/vertebral/results_prc_vertebral_2.pickle', 'rb') as handle:
    results_prc_97 = pickle.load(handle)
with open('./training/offline/vertebral/results_f1_vertebral_2.pickle', 'rb') as handle:
    results_f1_97 = pickle.load(handle)
with open('./training/offline/vertebral/metafeatures_vertebral_2.pickle', 'rb') as handle:
    meta_features_97 = pickle.load(handle)  

with open('./training/offline/vertebral/results_roc_vertebral_3.pickle', 'rb') as handle:
    results_roc_98 = pickle.load(handle)
with open('./training/offline/vertebral/results_prc_vertebral_3.pickle', 'rb') as handle:
    results_prc_98 = pickle.load(handle)
with open('./training/offline/vertebral/results_f1_vertebral_3.pickle', 'rb') as handle:
    results_f1_98 = pickle.load(handle)
with open('./training/offline/vertebral/metafeatures_vertebral_3.pickle', 'rb') as handle:
    meta_features_98 = pickle.load(handle)  

with open('./training/offline/vertebral/results_roc_vertebral_4.pickle', 'rb') as handle:
    results_roc_99 = pickle.load(handle)
with open('./training/offline/vertebral/results_prc_vertebral_4.pickle', 'rb') as handle:
    results_prc_99 = pickle.load(handle)
with open('./training/offline/vertebral/results_f1_vertebral_4.pickle', 'rb') as handle:
    results_f1_99 = pickle.load(handle)
with open('./training/offline/vertebral/metafeatures_vertebral_4.pickle', 'rb') as handle:
    meta_features_99 = pickle.load(handle)  

#==============================================================================
#==========================Dataset 21==========================================
#==============================================================================
with open('./training/offline/nba_rookies/results_roc_NBA_0.pickle', 'rb') as handle:
    results_roc_100 = pickle.load(handle)
with open('./training/offline/nba_rookies/results_prc_NBA_0.pickle', 'rb') as handle:
    results_prc_100 = pickle.load(handle)
with open('./training/offline/nba_rookies/results_f1_NBA_0.pickle', 'rb') as handle:
    results_f1_100 = pickle.load(handle)
with open('./training/offline/nba_rookies/metafeatures_NBA_0.pickle', 'rb') as handle:
    meta_features_100 = pickle.load(handle)  

with open('./training/offline/nba_rookies/results_roc_NBA_1.pickle', 'rb') as handle:
    results_roc_101 = pickle.load(handle)
with open('./training/offline/nba_rookies/results_prc_NBA_1.pickle', 'rb') as handle:
    results_prc_101 = pickle.load(handle)
with open('./training/offline/nba_rookies/results_f1_NBA_1.pickle', 'rb') as handle:
    results_f1_101 = pickle.load(handle)
with open('./training/offline/nba_rookies/metafeatures_NBA_1.pickle', 'rb') as handle:
    meta_features_101 = pickle.load(handle)  
    
with open('./training/offline/nba_rookies/results_roc_NBA_2.pickle', 'rb') as handle:
    results_roc_102 = pickle.load(handle)
with open('./training/offline/nba_rookies/results_prc_NBA_2.pickle', 'rb') as handle:
    results_prc_102 = pickle.load(handle)
with open('./training/offline/nba_rookies/results_f1_NBA_2.pickle', 'rb') as handle:
    results_f1_102 = pickle.load(handle)
with open('./training/offline/nba_rookies/metafeatures_NBA_2.pickle', 'rb') as handle:
    meta_features_102 = pickle.load(handle)  
    
with open('./training/offline/nba_rookies/results_roc_NBA_3.pickle', 'rb') as handle:
    results_roc_103 = pickle.load(handle)
with open('./training/offline/nba_rookies/results_prc_NBA_3.pickle', 'rb') as handle:
    results_prc_103 = pickle.load(handle)
with open('./training/offline/nba_rookies/results_f1_NBA_3.pickle', 'rb') as handle:
    results_f1_103 = pickle.load(handle)
with open('./training/offline/nba_rookies/metafeatures_NBA_3.pickle', 'rb') as handle:
    meta_features_103 = pickle.load(handle)  
    
with open('./training/offline/nba_rookies/results_roc_NBA_4.pickle', 'rb') as handle:
    results_roc_104 = pickle.load(handle)
with open('./training/offline/nba_rookies/results_prc_NBA_4.pickle', 'rb') as handle:
    results_prc_104 = pickle.load(handle)
with open('./training/offline/nba_rookies/results_f1_NBA_4.pickle', 'rb') as handle:
    results_f1_104 = pickle.load(handle)
with open('./training/offline/nba_rookies/metafeatures_NBA_4.pickle', 'rb') as handle:
    meta_features_104 = pickle.load(handle)  

#==============================================================================
#==========================Dataset 22==========================================
#==============================================================================
with open('./training/offline/surgical/results_roc_surgical_0.pickle', 'rb') as handle:
    results_roc_105 = pickle.load(handle)
with open('./training/offline/surgical/results_prc_surgical_0.pickle', 'rb') as handle:
    results_prc_105 = pickle.load(handle)
with open('./training/offline/surgical/results_f1_surgical_0.pickle', 'rb') as handle:
    results_f1_105 = pickle.load(handle)
with open('./training/offline/surgical/metafeatures_surgical_0.pickle', 'rb') as handle:
    meta_features_105 = pickle.load(handle)  

with open('./training/offline/surgical/results_roc_surgical_1.pickle', 'rb') as handle:
    results_roc_106 = pickle.load(handle)
with open('./training/offline/surgical/results_prc_surgical_1.pickle', 'rb') as handle:
    results_prc_106 = pickle.load(handle)
with open('./training/offline/surgical/results_f1_surgical_1.pickle', 'rb') as handle:
    results_f1_106 = pickle.load(handle)
with open('./training/offline/surgical/metafeatures_surgical_1.pickle', 'rb') as handle:
    meta_features_106 = pickle.load(handle)  
    
with open('./training/offline/surgical/results_roc_surgical_2.pickle', 'rb') as handle:
    results_roc_107 = pickle.load(handle)
with open('./training/offline/surgical/results_prc_surgical_2.pickle', 'rb') as handle:
    results_prc_107 = pickle.load(handle)
with open('./training/offline/surgical/results_f1_surgical_2.pickle', 'rb') as handle:
    results_f1_107 = pickle.load(handle)
with open('./training/offline/surgical/metafeatures_surgical_2.pickle', 'rb') as handle:
    meta_features_107 = pickle.load(handle)  
    
with open('./training/offline/surgical/results_roc_surgical_3.pickle', 'rb') as handle:
    results_roc_108 = pickle.load(handle)
with open('./training/offline/surgical/results_prc_surgical_3.pickle', 'rb') as handle:
    results_prc_108 = pickle.load(handle)
with open('./training/offline/surgical/results_f1_surgical_3.pickle', 'rb') as handle:
    results_f1_108 = pickle.load(handle)
with open('./training/offline/surgical/metafeatures_surgical_3.pickle', 'rb') as handle:
    meta_features_108 = pickle.load(handle)  
    
with open('./training/offline/surgical/results_roc_surgical_4.pickle', 'rb') as handle:
    results_roc_109 = pickle.load(handle)
with open('./training/offline/surgical/results_prc_surgical_4.pickle', 'rb') as handle:
    results_prc_109 = pickle.load(handle)
with open('./training/offline/surgical/results_f1_surgical_4.pickle', 'rb') as handle:
    results_f1_109 = pickle.load(handle)
with open('./training/offline/surgical/metafeatures_surgical_4.pickle', 'rb') as handle:
    meta_features_109 = pickle.load(handle)  
    
#==============================================================================
#==========================Dataset 23==========================================
#==============================================================================
with open('./training/offline/banana/results_roc_banana_0.pickle', 'rb') as handle:
    results_roc_110 = pickle.load(handle)
with open('./training/offline/banana/results_prc_banana_0.pickle', 'rb') as handle:
    results_prc_110 = pickle.load(handle)
with open('./training/offline/banana/results_f1_banana_0.pickle', 'rb') as handle:
    results_f1_110 = pickle.load(handle)
with open('./training/offline/banana/metafeatures_banana_0.pickle', 'rb') as handle:
    meta_features_110 = pickle.load(handle)  

with open('./training/offline/banana/results_roc_banana_1.pickle', 'rb') as handle:
    results_roc_111 = pickle.load(handle)
with open('./training/offline/banana/results_prc_banana_1.pickle', 'rb') as handle:
    results_prc_111 = pickle.load(handle)
with open('./training/offline/banana/results_f1_banana_1.pickle', 'rb') as handle:
    results_f1_111 = pickle.load(handle)
with open('./training/offline/banana/metafeatures_banana_1.pickle', 'rb') as handle:
    meta_features_111 = pickle.load(handle)  
    
with open('./training/offline/banana/results_roc_banana_2.pickle', 'rb') as handle:
    results_roc_112 = pickle.load(handle)
with open('./training/offline/banana/results_prc_banana_2.pickle', 'rb') as handle:
    results_prc_112 = pickle.load(handle)
with open('./training/offline/banana/results_f1_banana_2.pickle', 'rb') as handle:
    results_f1_112 = pickle.load(handle)
with open('./training/offline/banana/metafeatures_banana_2.pickle', 'rb') as handle:
    meta_features_112 = pickle.load(handle) 
    
with open('./training/offline/banana/results_roc_banana_3.pickle', 'rb') as handle:
    results_roc_113 = pickle.load(handle)
with open('./training/offline/banana/results_prc_banana_3.pickle', 'rb') as handle:
    results_prc_113 = pickle.load(handle)
with open('./training/offline/banana/results_f1_banana_3.pickle', 'rb') as handle:
    results_f1_113 = pickle.load(handle)
with open('./training/offline/banana/metafeatures_banana_3.pickle', 'rb') as handle:
    meta_features_113 = pickle.load(handle)  
    
with open('./training/offline/banana/results_roc_banana_4.pickle', 'rb') as handle:
    results_roc_114 = pickle.load(handle)
with open('./training/offline/banana/results_prc_banana_4.pickle', 'rb') as handle:
    results_prc_114 = pickle.load(handle)
with open('./training/offline/banana/results_f1_banana_4.pickle', 'rb') as handle:
    results_f1_114 = pickle.load(handle)
with open('./training/offline/banana/metafeatures_banana_4.pickle', 'rb') as handle:
    meta_features_114 = pickle.load(handle)  

#==============================================================================
#==========================Dataset 24==========================================
#==============================================================================
with open('./training/offline/german_number/results_roc_german_1.pickle', 'rb') as handle:
    results_roc_115 = pickle.load(handle)
with open('./training/offline/german_number/results_prc_german_1.pickle', 'rb') as handle:
    results_prc_115 = pickle.load(handle)
with open('./training/offline/german_number/results_f1_german_1.pickle', 'rb') as handle:
    results_f1_115 = pickle.load(handle)
with open('./training/offline/german_number/metafeatures_german_1.pickle', 'rb') as handle:
    meta_features_115 = pickle.load(handle)  
    
with open('./training/offline/german_number/results_roc_german_2.pickle', 'rb') as handle:
    results_roc_116 = pickle.load(handle)
with open('./training/offline/german_number/results_prc_german_2.pickle', 'rb') as handle:
    results_prc_116 = pickle.load(handle)
with open('./training/offline/german_number/results_f1_german_2.pickle', 'rb') as handle:
    results_f1_116 = pickle.load(handle)
with open('./training/offline/german_number/metafeatures_german_2.pickle', 'rb') as handle:
    meta_features_116 = pickle.load(handle) 
    
with open('./training/offline/german_number/results_roc_german_3.pickle', 'rb') as handle:
    results_roc_117 = pickle.load(handle)
with open('./training/offline/german_number/results_prc_german_3.pickle', 'rb') as handle:
    results_prc_117 = pickle.load(handle)
with open('./training/offline/german_number/results_f1_german_3.pickle', 'rb') as handle:
    results_f1_117 = pickle.load(handle)
with open('./training/offline/german_number/metafeatures_german_3.pickle', 'rb') as handle:
    meta_features_117 = pickle.load(handle)  
    
with open('./training/offline/german_number/results_roc_german_4.pickle', 'rb') as handle:
    results_roc_118 = pickle.load(handle)
with open('./training/offline/german_number/results_prc_german_4.pickle', 'rb') as handle:
    results_prc_118 = pickle.load(handle)
with open('./training/offline/german_number/results_f1_german_4.pickle', 'rb') as handle:
    results_f1_118 = pickle.load(handle)
with open('./training/offline/german_number/metafeatures_german_4.pickle', 'rb') as handle:
    meta_features_118 = pickle.load(handle)  

#==============================================================================
#==========================Dataset 25==========================================
#==============================================================================
with open('./training/offline/colon_cancer/results_roc_colon_0.pickle', 'rb') as handle:
    results_roc_119 = pickle.load(handle)
with open('./training/offline/colon_cancer/results_prc_colon_0.pickle', 'rb') as handle:
    results_prc_119 = pickle.load(handle)
with open('./training/offline/colon_cancer/results_f1_colon_0.pickle', 'rb') as handle:
    results_f1_119 = pickle.load(handle)
with open('./training/offline/colon_cancer/metafeatures_colon_0.pickle', 'rb') as handle:
    meta_features_119 = pickle.load(handle)
    
with open('./training/offline/colon_cancer/results_roc_colon_1.pickle', 'rb') as handle:
    results_roc_120 = pickle.load(handle)
with open('./training/offline/colon_cancer/results_prc_colon_1.pickle', 'rb') as handle:
    results_prc_120 = pickle.load(handle)
with open('./training/offline/colon_cancer/results_f1_colon_1.pickle', 'rb') as handle:
    results_f1_120 = pickle.load(handle)
with open('./training/offline/colon_cancer/metafeatures_colon_1.pickle', 'rb') as handle:
    meta_features_120 = pickle.load(handle)  
    
with open('./training/offline/colon_cancer/results_roc_colon_2.pickle', 'rb') as handle:
    results_roc_121 = pickle.load(handle)
with open('./training/offline/colon_cancer/results_prc_colon_2.pickle', 'rb') as handle:
    results_prc_121 = pickle.load(handle)
with open('./training/offline/colon_cancer/results_f1_colon_2.pickle', 'rb') as handle:
    results_f1_121 = pickle.load(handle)
with open('./training/offline/colon_cancer/metafeatures_colon_2.pickle', 'rb') as handle:
    meta_features_121 = pickle.load(handle) 
    
with open('./training/offline/colon_cancer/results_roc_colon_3.pickle', 'rb') as handle:
    results_roc_122 = pickle.load(handle)
with open('./training/offline/colon_cancer/results_prc_colon_3.pickle', 'rb') as handle:
    results_prc_122 = pickle.load(handle)
with open('./training/offline/colon_cancer/results_f1_colon_3.pickle', 'rb') as handle:
    results_f1_122 = pickle.load(handle)
with open('./training/offline/colon_cancer/metafeatures_colon_3.pickle', 'rb') as handle:
    meta_features_122 = pickle.load(handle)  
    
with open('./training/offline/colon_cancer/results_roc_colon_4.pickle', 'rb') as handle:
    results_roc_123 = pickle.load(handle)
with open('./training/offline/colon_cancer/results_prc_colon_4.pickle', 'rb') as handle:
    results_prc_123 = pickle.load(handle)
with open('./training/offline/colon_cancer/results_f1_colon_4.pickle', 'rb') as handle:
    results_f1_123 = pickle.load(handle)
with open('./training/offline/colon_cancer/metafeatures_colon_4.pickle', 'rb') as handle:
    meta_features_123 = pickle.load(handle)  

#==============================================================================
#==========================Dataset 26==========================================
#==============================================================================
with open('./training/offline/diabetes/results_roc_diabetes_0.pickle', 'rb') as handle:
    results_roc_124 = pickle.load(handle)
with open('./training/offline/diabetes/results_prc_diabates_0.pickle', 'rb') as handle:
    results_prc_124 = pickle.load(handle)
with open('./training/offline/diabetes/results_f1_diabates_0.pickle', 'rb') as handle:
    results_f1_124 = pickle.load(handle)
with open('./training/offline/diabetes/metafeatures_diabates_0.pickle', 'rb') as handle:
    meta_features_124 = pickle.load(handle)
    
with open('./training/offline/diabetes/results_roc_diabetes_1.pickle', 'rb') as handle:
    results_roc_125 = pickle.load(handle)
with open('./training/offline/diabetes/results_prc_diabates_1.pickle', 'rb') as handle:
    results_prc_125 = pickle.load(handle)
with open('./training/offline/diabetes/results_f1_diabates_1.pickle', 'rb') as handle:
    results_f1_125 = pickle.load(handle)
with open('./training/offline/diabetes/metafeatures_diabates_1.pickle', 'rb') as handle:
    meta_features_125 = pickle.load(handle)  
    
with open('./training/offline/diabetes/results_roc_diabetes_2.pickle', 'rb') as handle:
    results_roc_126 = pickle.load(handle)
with open('./training/offline/diabetes/results_prc_diabates_2.pickle', 'rb') as handle:
    results_prc_126 = pickle.load(handle)
with open('./training/offline/diabetes/results_f1_diabates_2.pickle', 'rb') as handle:
    results_f1_126 = pickle.load(handle)
with open('./training/offline/diabetes/metafeatures_diabates_2.pickle', 'rb') as handle:
    meta_features_126 = pickle.load(handle) 
    
with open('./training/offline/diabetes/results_roc_diabetes_3.pickle', 'rb') as handle:
    results_roc_127 = pickle.load(handle)
with open('./training/offline/diabetes/results_prc_diabates_3.pickle', 'rb') as handle:
    results_prc_127 = pickle.load(handle)
with open('./training/offline/diabetes/results_f1_diabates_3.pickle', 'rb') as handle:
    results_f1_127 = pickle.load(handle)
with open('./training/offline/diabetes/metafeatures_diabates_3.pickle', 'rb') as handle:
    meta_features_127 = pickle.load(handle)  
    
with open('./training/offline/diabetes/results_roc_diabetes_4.pickle', 'rb') as handle:
    results_roc_128 = pickle.load(handle)
with open('./training/offline/diabetes/results_prc_diabates_4.pickle', 'rb') as handle:
    results_prc_128 = pickle.load(handle)
with open('./training/offline/diabetes/results_f1_diabates_4.pickle', 'rb') as handle:
    results_f1_128 = pickle.load(handle)
with open('./training/offline/diabetes/metafeatures_diabates_4.pickle', 'rb') as handle:
    meta_features_128 = pickle.load(handle)  
    
#==============================================================================
#==========================Dataset 27==========================================
#==============================================================================
with open('./training/offline/pulsar_star/results_roc_pulsar_0.pickle', 'rb') as handle:
    results_roc_129 = pickle.load(handle)
with open('./training/offline/pulsar_star/results_prc_pulsar_0.pickle', 'rb') as handle:
    results_prc_129 = pickle.load(handle)
with open('./training/offline/pulsar_star/results_f1_pulsar_0.pickle', 'rb') as handle:
    results_f1_129 = pickle.load(handle)
with open('./training/offline/pulsar_star/metafeatures_pulsar_0.pickle', 'rb') as handle:
    meta_features_129 = pickle.load(handle)
    
with open('./training/offline/pulsar_star/results_roc_pulsar_1.pickle', 'rb') as handle:
    results_roc_130 = pickle.load(handle)
with open('./training/offline/pulsar_star/results_prc_pulsar_1.pickle', 'rb') as handle:
    results_prc_130 = pickle.load(handle)
with open('./training/offline/pulsar_star/results_f1_pulsar_1.pickle', 'rb') as handle:
    results_f1_130 = pickle.load(handle)
with open('./training/offline/pulsar_star/metafeatures_pulsar_1.pickle', 'rb') as handle:
    meta_features_130 = pickle.load(handle)  
    
with open('./training/offline/pulsar_star/results_roc_pulsar_2.pickle', 'rb') as handle:
    results_roc_131 = pickle.load(handle)
with open('./training/offline/pulsar_star/results_prc_pulsar_2.pickle', 'rb') as handle:
    results_prc_131 = pickle.load(handle)
with open('./training/offline/pulsar_star/results_f1_pulsar_2.pickle', 'rb') as handle:
    results_f1_131 = pickle.load(handle)
with open('./training/offline/pulsar_star/metafeatures_pulsar_2.pickle', 'rb') as handle:
    meta_features_131 = pickle.load(handle) 
    
with open('./training/offline/pulsar_star/results_roc_pulsar_3.pickle', 'rb') as handle:
    results_roc_132 = pickle.load(handle)
with open('./training/offline/pulsar_star/results_prc_pulsar_3.pickle', 'rb') as handle:
    results_prc_132 = pickle.load(handle)
with open('./training/offline/pulsar_star/results_f1_pulsar_3.pickle', 'rb') as handle:
    results_f1_132 = pickle.load(handle)
with open('./training/offline/pulsar_star/metafeatures_pulsar_3.pickle', 'rb') as handle:
    meta_features_132 = pickle.load(handle)  
    
with open('./training/offline/pulsar_star/results_roc_pulsar_4.pickle', 'rb') as handle:
    results_roc_133 = pickle.load(handle)
with open('./training/offline/pulsar_star/results_prc_pulsar_4.pickle', 'rb') as handle:
    results_prc_133 = pickle.load(handle)
with open('./training/offline/pulsar_star/results_f1_pulsar_4.pickle', 'rb') as handle:
    results_f1_133 = pickle.load(handle)
with open('./training/offline/pulsar_star/metafeatures_pulsar_4.pickle', 'rb') as handle:
    meta_features_133 = pickle.load(handle)  

#==============================================================================
#==========================Dataset 28==========================================
#==============================================================================
with open('./training/offline/abalone/results_roc_abalone_0.pickle', 'rb') as handle:
    results_roc_134 = pickle.load(handle)
with open('./training/offline/abalone/results_prc_abalone_0.pickle', 'rb') as handle:
    results_prc_134 = pickle.load(handle)
with open('./training/offline/abalone/results_f1_abalone_0.pickle', 'rb') as handle:
    results_f1_134 = pickle.load(handle)
with open('./training/offline/abalone/metafeatures_abalone_0.pickle', 'rb') as handle:
    meta_features_134 = pickle.load(handle)
    
with open('./training/offline/abalone/results_roc_abalone_1.pickle', 'rb') as handle:
    results_roc_135 = pickle.load(handle)
with open('./training/offline/abalone/results_prc_abalone_1.pickle', 'rb') as handle:
    results_prc_135 = pickle.load(handle)
with open('./training/offline/abalone/results_f1_abalone_1.pickle', 'rb') as handle:
    results_f1_135 = pickle.load(handle)
with open('./training/offline/abalone/metafeatures_abalone_1.pickle', 'rb') as handle:
    meta_features_135 = pickle.load(handle)
    
with open('./training/offline/abalone/results_roc_abalone_2.pickle', 'rb') as handle:
    results_roc_136 = pickle.load(handle)
with open('./training/offline/abalone/results_prc_abalone_2.pickle', 'rb') as handle:
    results_prc_136 = pickle.load(handle)
with open('./training/offline/abalone/results_f1_abalone_2.pickle', 'rb') as handle:
    results_f1_136 = pickle.load(handle)
with open('./training/offline/abalone/metafeatures_abalone_2.pickle', 'rb') as handle:
    meta_features_136 = pickle.load(handle)
    
with open('./training/offline/abalone/results_roc_abalone_3.pickle', 'rb') as handle:
    results_roc_137 = pickle.load(handle)
with open('./training/offline/abalone/results_prc_abalone_3.pickle', 'rb') as handle:
    results_prc_137 = pickle.load(handle)
with open('./training/offline/abalone/results_f1_abalone_3.pickle', 'rb') as handle:
    results_f1_137 = pickle.load(handle)
with open('./training/offline/abalone/metafeatures_abalone_3.pickle', 'rb') as handle:
    meta_features_137 = pickle.load(handle)
    
with open('./training/offline/abalone/results_roc_abalone_4.pickle', 'rb') as handle:
    results_roc_138 = pickle.load(handle)
with open('./training/offline/abalone/results_prc_abalone_4.pickle', 'rb') as handle:
    results_prc_138 = pickle.load(handle)
with open('./training/offline/abalone/results_f1_abalone_4.pickle', 'rb') as handle:
    results_f1_138 = pickle.load(handle)
with open('./training/offline/abalone/metafeatures_abalone_4.pickle', 'rb') as handle:
    meta_features_138 = pickle.load(handle)


#==============================================================================
#==========================Dataset 29==========================================
#==============================================================================
with open('./training/offline/seeds/results_roc_seeds_0.pickle', 'rb') as handle:
    results_roc_139 = pickle.load(handle)
with open('./training/offline/seeds/results_prc_seeds_0.pickle', 'rb') as handle:
    results_prc_139 = pickle.load(handle)
with open('./training/offline/seeds/results_f1_seeds_0.pickle', 'rb') as handle:
    results_f1_139 = pickle.load(handle)
with open('./training/offline/seeds/metafeatures_seeds_0.pickle', 'rb') as handle:
    meta_features_139 = pickle.load(handle)

with open('./training/offline/seeds/results_roc_seeds_1.pickle', 'rb') as handle:
    results_roc_140 = pickle.load(handle)
with open('./training/offline/seeds/results_prc_seeds_1.pickle', 'rb') as handle:
    results_prc_140 = pickle.load(handle)
with open('./training/offline/seeds/results_f1_seeds_1.pickle', 'rb') as handle:
    results_f1_140 = pickle.load(handle)
with open('./training/offline/seeds/metafeatures_seeds_1.pickle', 'rb') as handle:
    meta_features_140 = pickle.load(handle)
    
with open('./training/offline/seeds/results_roc_seeds_2.pickle', 'rb') as handle:
    results_roc_141 = pickle.load(handle)
with open('./training/offline/seeds/results_prc_seeds_2.pickle', 'rb') as handle:
    results_prc_141 = pickle.load(handle)
with open('./training/offline/seeds/results_f1_seeds_2.pickle', 'rb') as handle:
    results_f1_141 = pickle.load(handle)
with open('./training/offline/seeds/metafeatures_seeds_2.pickle', 'rb') as handle:
    meta_features_141 = pickle.load(handle)

with open('./training/offline/seeds/results_roc_seeds_3.pickle', 'rb') as handle:
    results_roc_142 = pickle.load(handle)
with open('./training/offline/seeds/results_prc_seeds_3.pickle', 'rb') as handle:
    results_prc_142 = pickle.load(handle)
with open('./training/offline/seeds/results_f1_seeds_3.pickle', 'rb') as handle:
    results_f1_142 = pickle.load(handle)
with open('./training/offline/seeds/metafeatures_seeds_3.pickle', 'rb') as handle:
    meta_features_142 = pickle.load(handle)

with open('./training/offline/seeds/results_roc_seeds_4.pickle', 'rb') as handle:
    results_roc_143 = pickle.load(handle)
with open('./training/offline/seeds/results_prc_seeds_4.pickle', 'rb') as handle:
    results_prc_143 = pickle.load(handle)
with open('./training/offline/seeds/results_f1_seeds_4.pickle', 'rb') as handle:
    results_f1_143 = pickle.load(handle)
with open('./training/offline/seeds/metafeatures_seeds_4.pickle', 'rb') as handle:
    meta_features_143 = pickle.load(handle)

#==============================================================================
#==========================Dataset 30==========================================
#==============================================================================
with open('./training/offline/hepatitis/results_roc_hepatitis_0.pickle', 'rb') as handle:
    results_roc_144 = pickle.load(handle)
with open('./training/offline/hepatitis/results_prc_hepatitis_0.pickle', 'rb') as handle:
    results_prc_144 = pickle.load(handle)
with open('./training/offline/hepatitis/results_f1_hepatitis_0.pickle', 'rb') as handle:
    results_f1_144 = pickle.load(handle)
with open('./training/offline/hepatitis/metafeatures_hepatitis_0.pickle', 'rb') as handle:
    meta_features_144 = pickle.load(handle)

with open('./training/offline/hepatitis/results_roc_hepatitis_1.pickle', 'rb') as handle:
    results_roc_145 = pickle.load(handle)
with open('./training/offline/hepatitis/results_prc_hepatitis_1.pickle', 'rb') as handle:
    results_prc_145 = pickle.load(handle)
with open('./training/offline/hepatitis/results_f1_hepatitis_1.pickle', 'rb') as handle:
    results_f1_145 = pickle.load(handle)
with open('./training/offline/hepatitis/metafeatures_hepatitis_1.pickle', 'rb') as handle:
    meta_features_145 = pickle.load(handle)

with open('./training/offline/hepatitis/results_roc_hepatitis_2.pickle', 'rb') as handle:
    results_roc_146 = pickle.load(handle)
with open('./training/offline/hepatitis/results_prc_hepatitis_2.pickle', 'rb') as handle:
    results_prc_146 = pickle.load(handle)
with open('./training/offline/hepatitis/results_f1_hepatitis_2.pickle', 'rb') as handle:
    results_f1_146 = pickle.load(handle)
with open('./training/offline/hepatitis/metafeatures_hepatitis_2.pickle', 'rb') as handle:
    meta_features_146 = pickle.load(handle)
    
with open('./training/offline/hepatitis/results_roc_hepatitis_3.pickle', 'rb') as handle:
    results_roc_147 = pickle.load(handle)
with open('./training/offline/hepatitis/results_prc_hepatitis_3.pickle', 'rb') as handle:
    results_prc_147 = pickle.load(handle)
with open('./training/offline/hepatitis/results_f1_hepatitis_3.pickle', 'rb') as handle:
    results_f1_147 = pickle.load(handle)
with open('./training/offline/hepatitis/metafeatures_hepatitis_3.pickle', 'rb') as handle:
    meta_features_147 = pickle.load(handle)
    
with open('./training/offline/hepatitis/results_roc_hepatitis_4.pickle', 'rb') as handle:
    results_roc_148 = pickle.load(handle)
with open('./training/offline/hepatitis/results_prc_hepatitis_4.pickle', 'rb') as handle:
    results_prc_148 = pickle.load(handle)
with open('./training/offline/hepatitis/results_f1_hepatitis_4.pickle', 'rb') as handle:
    results_f1_148 = pickle.load(handle)
with open('./training/offline/hepatitis/metafeatures_hepatitis_4.pickle', 'rb') as handle:
    meta_features_148 = pickle.load(handle)

#==============================================================================
#==========================Dataset 31==========================================
#==============================================================================
with open('./training/offline/eye/results_roc_eye_0.pickle', 'rb') as handle:
    results_roc_149 = pickle.load(handle)
with open('./training/offline/eye/results_prc_eye_0.pickle', 'rb') as handle:
    results_prc_149 = pickle.load(handle)
with open('./training/offline/eye/results_f1_eye_0.pickle', 'rb') as handle:
    results_f1_149 = pickle.load(handle)
with open('./training/offline/eye/metafeatures_eye_0.pickle', 'rb') as handle:
    meta_features_149 = pickle.load(handle)

with open('./training/offline/eye/results_roc_eye_1.pickle', 'rb') as handle:
    results_roc_150 = pickle.load(handle)
with open('./training/offline/eye/results_prc_eye_1.pickle', 'rb') as handle:
    results_prc_150 = pickle.load(handle)
with open('./training/offline/eye/results_f1_eye_1.pickle', 'rb') as handle:
    results_f1_150 = pickle.load(handle)
with open('./training/offline/eye/metafeatures_eye_1.pickle', 'rb') as handle:
    meta_features_150 = pickle.load(handle)

with open('./training/offline/eye/results_roc_eye_2.pickle', 'rb') as handle:
    results_roc_151 = pickle.load(handle)
with open('./training/offline/eye/results_prc_eye_2.pickle', 'rb') as handle:
    results_prc_151 = pickle.load(handle)
with open('./training/offline/eye/results_f1_eye_2.pickle', 'rb') as handle:
    results_f1_151 = pickle.load(handle)
with open('./training/offline/eye/metafeatures_eye_2.pickle', 'rb') as handle:
    meta_features_151 = pickle.load(handle)
    
with open('./training/offline/eye/results_roc_eye_3.pickle', 'rb') as handle:
    results_roc_152 = pickle.load(handle)
with open('./training/offline/eye/results_prc_eye_3.pickle', 'rb') as handle:
    results_prc_152 = pickle.load(handle)
with open('./training/offline/eye/results_f1_eye_3.pickle', 'rb') as handle:
    results_f1_152 = pickle.load(handle)
with open('./training/offline/eye/metafeatures_eye_3.pickle', 'rb') as handle:
    meta_features_152 = pickle.load(handle)
    
with open('./training/offline/eye/results_roc_eye_4.pickle', 'rb') as handle:
    results_roc_153 = pickle.load(handle)
with open('./training/offline/eye/results_prc_eye_4.pickle', 'rb') as handle:
    results_prc_153 = pickle.load(handle)
with open('./training/offline/eye/results_f1_eye_4.pickle', 'rb') as handle:
    results_f1_153 = pickle.load(handle)
with open('./training/offline/eye/metafeatures_eye_4.pickle', 'rb') as handle:
    meta_features_153 = pickle.load(handle)

#==============================================================================
#==========================Dataset 32==========================================
#==============================================================================
with open('./training/offline/ecoli/results_roc_ecoli_0.pickle', 'rb') as handle:
    results_roc_154 = pickle.load(handle)
with open('./training/offline/ecoli/results_prc_ecoli_0.pickle', 'rb') as handle:
    results_prc_154 = pickle.load(handle)
with open('./training/offline/ecoli/results_f1_ecoli_0.pickle', 'rb') as handle:
    results_f1_154 = pickle.load(handle)
with open('./training/offline/ecoli/metafeatures_ecoli_0.pickle', 'rb') as handle:
    meta_features_154 = pickle.load(handle)

with open('./training/offline/ecoli/results_roc_ecoli_1.pickle', 'rb') as handle:
    results_roc_155 = pickle.load(handle)
with open('./training/offline/ecoli/results_prc_ecoli_1.pickle', 'rb') as handle:
    results_prc_155 = pickle.load(handle)
with open('./training/offline/ecoli/results_f1_ecoli_1.pickle', 'rb') as handle:
    results_f1_155 = pickle.load(handle)
with open('./training/offline/ecoli/metafeatures_ecoli_1.pickle', 'rb') as handle:
    meta_features_155 = pickle.load(handle)

with open('./training/offline/ecoli/results_roc_ecoli_2.pickle', 'rb') as handle:
    results_roc_156 = pickle.load(handle)
with open('./training/offline/ecoli/results_prc_ecoli_2.pickle', 'rb') as handle:
    results_prc_156 = pickle.load(handle)
with open('./training/offline/ecoli/results_f1_ecoli_2.pickle', 'rb') as handle:
    results_f1_156 = pickle.load(handle)
with open('./training/offline/ecoli/metafeatures_ecoli_2.pickle', 'rb') as handle:
    meta_features_156 = pickle.load(handle)
    
with open('./training/offline/ecoli/results_roc_ecoli_3.pickle', 'rb') as handle:
    results_roc_157 = pickle.load(handle)
with open('./training/offline/ecoli/results_prc_ecoli_3.pickle', 'rb') as handle:
    results_prc_157 = pickle.load(handle)
with open('./training/offline/ecoli/results_f1_ecoli_3.pickle', 'rb') as handle:
    results_f1_157 = pickle.load(handle)
with open('./training/offline/ecoli/metafeatures_ecoli_3.pickle', 'rb') as handle:
    meta_features_157 = pickle.load(handle)
    
with open('./training/offline/ecoli/results_roc_ecoli_4.pickle', 'rb') as handle:
    results_roc_158 = pickle.load(handle)
with open('./training/offline/ecoli/results_prc_ecoli_4.pickle', 'rb') as handle:
    results_prc_158 = pickle.load(handle)
with open('./training/offline/ecoli/results_f1_ecoli_4.pickle', 'rb') as handle:
    results_f1_158 = pickle.load(handle)
with open('./training/offline/ecoli/metafeatures_ecoli_4.pickle', 'rb') as handle:
    meta_features_158 = pickle.load(handle)

#==============================================================================
#==========================Dataset 33==========================================
#==============================================================================
with open('./training/offline/drug_consumption/results_roc_drug_0.pickle', 'rb') as handle:
    results_roc_159 = pickle.load(handle)
with open('./training/offline/drug_consumption/results_prc_drug_0.pickle', 'rb') as handle:
    results_prc_159 = pickle.load(handle)
with open('./training/offline/drug_consumption/results_f1_drug_0.pickle', 'rb') as handle:
    results_f1_159 = pickle.load(handle)
with open('./training/offline/drug_consumption/metafeatures_drug_0.pickle', 'rb') as handle:
    meta_features_159 = pickle.load(handle)

with open('./training/offline/drug_consumption/results_roc_drug_1.pickle', 'rb') as handle:
    results_roc_160 = pickle.load(handle)
with open('./training/offline/drug_consumption/results_prc_drug_1.pickle', 'rb') as handle:
    results_prc_160 = pickle.load(handle)
with open('./training/offline/drug_consumption/results_f1_drug_1.pickle', 'rb') as handle:
    results_f1_160 = pickle.load(handle)
with open('./training/offline/drug_consumption/metafeatures_drug_1.pickle', 'rb') as handle:
    meta_features_160 = pickle.load(handle)

with open('./training/offline/drug_consumption/results_roc_drug_2.pickle', 'rb') as handle:
    results_roc_161 = pickle.load(handle)
with open('./training/offline/drug_consumption/results_prc_drug_2.pickle', 'rb') as handle:
    results_prc_161 = pickle.load(handle)
with open('./training/offline/drug_consumption/results_f1_drug_2.pickle', 'rb') as handle:
    results_f1_161 = pickle.load(handle)
with open('./training/offline/drug_consumption/metafeatures_drug_2.pickle', 'rb') as handle:
    meta_features_161 = pickle.load(handle)
    
with open('./training/offline/drug_consumption/results_roc_drug_3.pickle', 'rb') as handle:
    results_roc_162 = pickle.load(handle)
with open('./training/offline/drug_consumption/results_prc_drug_3.pickle', 'rb') as handle:
    results_prc_162 = pickle.load(handle)
with open('./training/offline/drug_consumption/results_f1_drug_3.pickle', 'rb') as handle:
    results_f1_162 = pickle.load(handle)
with open('./training/offline/drug_consumption/metafeatures_drug_3.pickle', 'rb') as handle:
    meta_features_162 = pickle.load(handle)
    
with open('./training/offline/drug_consumption/results_roc_drug_4.pickle', 'rb') as handle:
    results_roc_163 = pickle.load(handle)
with open('./training/offline/drug_consumption/results_prc_drug_4.pickle', 'rb') as handle:
    results_prc_163 = pickle.load(handle)
with open('./training/offline/drug_consumption/results_f1_drug_4.pickle', 'rb') as handle:
    results_f1_163 = pickle.load(handle)
with open('./training/offline/drug_consumption/metafeatures_drug_4.pickle', 'rb') as handle:
    meta_features_163 = pickle.load(handle)


#==============================================================================
#==========================Dataset 34==========================================
#==============================================================================
with open('./training/offline/mammographic/results_roc_mammographic_0.pickle', 'rb') as handle:
    results_roc_164 = pickle.load(handle)
with open('./training/offline/mammographic/results_prc_mammographic_0.pickle', 'rb') as handle:
    results_prc_164 = pickle.load(handle)
with open('./training/offline/mammographic/results_f1_mammographic_0.pickle', 'rb') as handle:
    results_f1_164 = pickle.load(handle)
with open('./training/offline/mammographic/metafeatures_mammographic_0.pickle', 'rb') as handle:
    meta_features_164 = pickle.load(handle)

with open('./training/offline/mammographic/results_roc_mammographic_1.pickle', 'rb') as handle:
    results_roc_165 = pickle.load(handle)
with open('./training/offline/mammographic/results_prc_mammographic_1.pickle', 'rb') as handle:
    results_prc_165 = pickle.load(handle)
with open('./training/offline/mammographic/results_f1_mammographic_1.pickle', 'rb') as handle:
    results_f1_165 = pickle.load(handle)
with open('./training/offline/mammographic/metafeatures_mammographic_1.pickle', 'rb') as handle:
    meta_features_165 = pickle.load(handle)

with open('./training/offline/mammographic/results_roc_mammographic_2.pickle', 'rb') as handle:
    results_roc_166 = pickle.load(handle)
with open('./training/offline/mammographic/results_prc_mammographic_2.pickle', 'rb') as handle:
    results_prc_166 = pickle.load(handle)
with open('./training/offline/mammographic/results_f1_mammographic_2.pickle', 'rb') as handle:
    results_f1_166 = pickle.load(handle)
with open('./training/offline/mammographic/metafeatures_mammographic_2.pickle', 'rb') as handle:
    meta_features_166 = pickle.load(handle)
    
with open('./training/offline/mammographic/results_roc_mammographic_3.pickle', 'rb') as handle:
    results_roc_167 = pickle.load(handle)
with open('./training/offline/mammographic/results_prc_mammographic_3.pickle', 'rb') as handle:
    results_prc_167 = pickle.load(handle)
with open('./training/offline/mammographic/results_f1_mammographic_3.pickle', 'rb') as handle:
    results_f1_167 = pickle.load(handle)
with open('./training/offline/mammographic/metafeatures_mammographic_3.pickle', 'rb') as handle:
    meta_features_167 = pickle.load(handle)
    
with open('./training/offline/mammographic/results_roc_mammographic_4.pickle', 'rb') as handle:
    results_roc_168 = pickle.load(handle)
with open('./training/offline/mammographic/results_prc_mammographic_4.pickle', 'rb') as handle:
    results_prc_168 = pickle.load(handle)
with open('./training/offline/mammographic/results_f1_mammographic_4.pickle', 'rb') as handle:
    results_f1_168 = pickle.load(handle)
with open('./training/offline/mammographic/metafeatures_mammographic_4.pickle', 'rb') as handle:
    meta_features_168 = pickle.load(handle)

#==============================================================================
#==========================Dataset 35==========================================
#==============================================================================
with open('./training/offline/parkinsons/results_roc_parkinsons_0.pickle', 'rb') as handle:
    results_roc_169 = pickle.load(handle)
with open('./training/offline/parkinsons/results_prc_parkinsons_0.pickle', 'rb') as handle:
    results_prc_169 = pickle.load(handle)
with open('./training/offline/parkinsons/results_f1_parkinsons_0.pickle', 'rb') as handle:
    results_f1_169 = pickle.load(handle)
with open('./training/offline/parkinsons/metafeatures_parkinsons_0.pickle', 'rb') as handle:
    meta_features_169 = pickle.load(handle)

with open('./training/offline/parkinsons/results_roc_parkinsons_1.pickle', 'rb') as handle:
    results_roc_170 = pickle.load(handle)
with open('./training/offline/parkinsons/results_prc_parkinsons_1.pickle', 'rb') as handle:
    results_prc_170 = pickle.load(handle)
with open('./training/offline/parkinsons/results_f1_parkinsons_1.pickle', 'rb') as handle:
    results_f1_170 = pickle.load(handle)
with open('./training/offline/parkinsons/metafeatures_parkinsons_1.pickle', 'rb') as handle:
    meta_features_170 = pickle.load(handle)

with open('./training/offline/parkinsons/results_roc_parkinsons_2.pickle', 'rb') as handle:
    results_roc_171 = pickle.load(handle)
with open('./training/offline/parkinsons/results_prc_parkinsons_2.pickle', 'rb') as handle:
    results_prc_171 = pickle.load(handle)
with open('./training/offline/parkinsons/results_f1_parkinsons_2.pickle', 'rb') as handle:
    results_f1_171 = pickle.load(handle)
with open('./training/offline/parkinsons/metafeatures_parkinsons_2.pickle', 'rb') as handle:
    meta_features_171 = pickle.load(handle)
    
with open('./training/offline/parkinsons/results_roc_parkinsons_3.pickle', 'rb') as handle:
    results_roc_172 = pickle.load(handle)
with open('./training/offline/parkinsons/results_prc_parkinsons_3.pickle', 'rb') as handle:
    results_prc_172 = pickle.load(handle)
with open('./training/offline/parkinsons/results_f1_parkinsons_3.pickle', 'rb') as handle:
    results_f1_172 = pickle.load(handle)
with open('./training/offline/parkinsons/metafeatures_parkinsons_3.pickle', 'rb') as handle:
    meta_features_172 = pickle.load(handle)
    
with open('./training/offline/parkinsons/results_roc_parkinsons_4.pickle', 'rb') as handle:
    results_roc_173 = pickle.load(handle)
with open('./training/offline/parkinsons/results_prc_parkinsons_4.pickle', 'rb') as handle:
    results_prc_173 = pickle.load(handle)
with open('./training/offline/parkinsons/results_f1_parkinsons_4.pickle', 'rb') as handle:
    results_f1_173 = pickle.load(handle)
with open('./training/offline/parkinsons/metafeatures_parkinsons_4.pickle', 'rb') as handle:
    meta_features_173 = pickle.load(handle)
    
    
#==============================================================================
#==========================Dataset 36==========================================
#==============================================================================
with open('./training/offline/mice_protein/results_roc_mice_0.pickle', 'rb') as handle:
    results_roc_174 = pickle.load(handle)
with open('./training/offline/mice_protein/results_prc_mice_0.pickle', 'rb') as handle:
    results_prc_174 = pickle.load(handle)
with open('./training/offline/mice_protein/results_f1_mice_0.pickle', 'rb') as handle:
    results_f1_174 = pickle.load(handle)
with open('./training/offline/mice_protein/metafeatures_mice_0.pickle', 'rb') as handle:
    meta_features_174 = pickle.load(handle)
    
with open('./training/offline/mice_protein/results_roc_mice_1.pickle', 'rb') as handle:
    results_roc_175 = pickle.load(handle)
with open('./training/offline/mice_protein/results_prc_mice_1.pickle', 'rb') as handle:
    results_prc_175 = pickle.load(handle)
with open('./training/offline/mice_protein/results_f1_mice_1.pickle', 'rb') as handle:
    results_f1_175 = pickle.load(handle)
with open('./training/offline/mice_protein/metafeatures_mice_1.pickle', 'rb') as handle:
    meta_features_175 = pickle.load(handle)
    
with open('./training/offline/mice_protein/results_roc_mice_2.pickle', 'rb') as handle:
    results_roc_176 = pickle.load(handle)
with open('./training/offline/mice_protein/results_prc_mice_2.pickle', 'rb') as handle:
    results_prc_176 = pickle.load(handle)
with open('./training/offline/mice_protein/results_f1_mice_2.pickle', 'rb') as handle:
    results_f1_176 = pickle.load(handle)
with open('./training/offline/mice_protein/metafeatures_mice_2.pickle', 'rb') as handle:
    meta_features_176 = pickle.load(handle)
    
with open('./training/offline/mice_protein/results_roc_mice_3.pickle', 'rb') as handle:
    results_roc_177 = pickle.load(handle)
with open('./training/offline/mice_protein/results_prc_mice_3.pickle', 'rb') as handle:
    results_prc_177 = pickle.load(handle)
with open('./training/offline/mice_protein/results_f1_mice_3.pickle', 'rb') as handle:
    results_f1_177 = pickle.load(handle)
with open('./training/offline/mice_protein/metafeatures_mice_3.pickle', 'rb') as handle:
    meta_features_177 = pickle.load(handle)
    
with open('./training/offline/mice_protein/results_roc_mice_4.pickle', 'rb') as handle:
    results_roc_178 = pickle.load(handle)
with open('./training/offline/mice_protein/results_prc_mice_4.pickle', 'rb') as handle:
    results_prc_178 = pickle.load(handle)
with open('./training/offline/mice_protein/results_f1_mice_4.pickle', 'rb') as handle:
    results_f1_178 = pickle.load(handle)
with open('./training/offline/mice_protein/metafeatures_mice_4.pickle', 'rb') as handle:
    meta_features_178 = pickle.load(handle)


#==============================================================================
#==========================Dataset 37==========================================
#==============================================================================
with open('./training/offline/alcohol/results_roc_alcohol_0.pickle', 'rb') as handle:
    results_roc_179 = pickle.load(handle)
with open('./training/offline/alcohol/results_prc_alcohol_0.pickle', 'rb') as handle:
    results_prc_179 = pickle.load(handle)
with open('./training/offline/alcohol/results_f1_alcohol_0.pickle', 'rb') as handle:
    results_f1_179 = pickle.load(handle)
with open('./training/offline/alcohol/metafeatures_alcohol_0.pickle', 'rb') as handle:
    meta_features_179 = pickle.load(handle)
    
with open('./training/offline/alcohol/results_roc_alcohol_2.pickle', 'rb') as handle:
    results_roc_180 = pickle.load(handle)
with open('./training/offline/alcohol/results_prc_alcohol_2.pickle', 'rb') as handle:
    results_prc_180 = pickle.load(handle)
with open('./training/offline/alcohol/results_f1_alcohol_2.pickle', 'rb') as handle:
    results_f1_180 = pickle.load(handle)
with open('./training/offline/alcohol/metafeatures_alcohol_2.pickle', 'rb') as handle:
    meta_features_180 = pickle.load(handle)
    
with open('./training/offline/alcohol/results_roc_alcohol_4.pickle', 'rb') as handle:
    results_roc_181 = pickle.load(handle)
with open('./training/offline/alcohol/results_prc_alcohol_4.pickle', 'rb') as handle:
    results_prc_181 = pickle.load(handle)
with open('./training/offline/alcohol/results_f1_alcohol_4.pickle', 'rb') as handle:
    results_f1_181 = pickle.load(handle)
with open('./training/offline/alcohol/metafeatures_alcohol_4.pickle', 'rb') as handle:
    meta_features_181 = pickle.load(handle)
    

#==============================================================================
#==========================Dataset 38==========================================
#==============================================================================
with open('./training/offline/biodegradation/results_roc_biodegradation_0.pickle', 'rb') as handle:
    results_roc_182 = pickle.load(handle)
with open('./training/offline/biodegradation/results_prc_biodegradation_0.pickle', 'rb') as handle:
    results_prc_182 = pickle.load(handle)
with open('./training/offline/biodegradation/results_f1_biodegradation_0.pickle', 'rb') as handle:
    results_f1_182 = pickle.load(handle)
with open('./training/offline/biodegradation/metafeatures_biodegradation_0.pickle', 'rb') as handle:
    meta_features_182 = pickle.load(handle)
    
with open('./training/offline/biodegradation/results_roc_biodegradation_1.pickle', 'rb') as handle:
    results_roc_183 = pickle.load(handle)
with open('./training/offline/biodegradation/results_prc_biodegradation_1.pickle', 'rb') as handle:
    results_prc_183 = pickle.load(handle)
with open('./training/offline/biodegradation/results_f1_biodegradation_1.pickle', 'rb') as handle:
    results_f1_183 = pickle.load(handle)
with open('./training/offline/biodegradation/metafeatures_biodegradation_1.pickle', 'rb') as handle:
    meta_features_183 = pickle.load(handle)
    
with open('./training/offline/biodegradation/results_roc_biodegradation_2.pickle', 'rb') as handle:
    results_roc_184 = pickle.load(handle)
with open('./training/offline/biodegradation/results_prc_biodegradation_2.pickle', 'rb') as handle:
    results_prc_184 = pickle.load(handle)
with open('./training/offline/biodegradation/results_f1_biodegradation_2.pickle', 'rb') as handle:
    results_f1_184 = pickle.load(handle)
with open('./training/offline/biodegradation/metafeatures_biodegradation_2.pickle', 'rb') as handle:
    meta_features_184 = pickle.load(handle)
    
with open('./training/offline/biodegradation/results_roc_biodegradation_3.pickle', 'rb') as handle:
    results_roc_185 = pickle.load(handle)
with open('./training/offline/biodegradation/results_prc_biodegradation_3.pickle', 'rb') as handle:
    results_prc_185 = pickle.load(handle)
with open('./training/offline/biodegradation/results_f1_biodegradation_3.pickle', 'rb') as handle:
    results_f1_185 = pickle.load(handle)
with open('./training/offline/biodegradation/metafeatures_biodegradation_3.pickle', 'rb') as handle:
    meta_features_185 = pickle.load(handle)
    
with open('./training/offline/biodegradation/results_roc_biodegradation_4.pickle', 'rb') as handle:
    results_roc_186 = pickle.load(handle)
with open('./training/offline/biodegradation/results_prc_biodegradation_4.pickle', 'rb') as handle:
    results_prc_186 = pickle.load(handle)
with open('./training/offline/biodegradation/results_f1_biodegradation_4.pickle', 'rb') as handle:
    results_f1_186 = pickle.load(handle)
with open('./training/offline/biodegradation/metafeatures_biodegradation_4.pickle', 'rb') as handle:
    meta_features_186 = pickle.load(handle)

#==============================================================================
#==========================Dataset 39==========================================
#==============================================================================
with open('./training/offline/climate/results_roc_climate_0.pickle', 'rb') as handle:
    results_roc_187 = pickle.load(handle)
with open('./training/offline/climate/results_prc_climate_0.pickle', 'rb') as handle:
    results_prc_187 = pickle.load(handle)
with open('./training/offline/climate/results_f1_climate_0.pickle', 'rb') as handle:
    results_f1_187 = pickle.load(handle)
with open('./training/offline/climate/metafeatures_climate_0.pickle', 'rb') as handle:
    meta_features_187 = pickle.load(handle)
    
with open('./training/offline/climate/results_roc_climate_1.pickle', 'rb') as handle:
    results_roc_188 = pickle.load(handle)
with open('./training/offline/climate/results_prc_climate_1.pickle', 'rb') as handle:
    results_prc_188 = pickle.load(handle)
with open('./training/offline/climate/results_f1_climate_1.pickle', 'rb') as handle:
    results_f1_188 = pickle.load(handle)
with open('./training/offline/climate/metafeatures_climate_1.pickle', 'rb') as handle:
    meta_features_188 = pickle.load(handle)
    
with open('./training/offline/climate/results_roc_climate_2.pickle', 'rb') as handle:
    results_roc_189 = pickle.load(handle)
with open('./training/offline/climate/results_prc_climate_2.pickle', 'rb') as handle:
    results_prc_189 = pickle.load(handle)
with open('./training/offline/climate/results_f1_climate_2.pickle', 'rb') as handle:
    results_f1_189 = pickle.load(handle)
with open('./training/offline/climate/metafeatures_climate_2.pickle', 'rb') as handle:
    meta_features_189 = pickle.load(handle)
    
with open('./training/offline/climate/results_roc_climate_3.pickle', 'rb') as handle:
    results_roc_190 = pickle.load(handle)
with open('./training/offline/climate/results_prc_climate_3.pickle', 'rb') as handle:
    results_prc_190 = pickle.load(handle)
with open('./training/offline/climate/results_f1_climate_3.pickle', 'rb') as handle:
    results_f1_190 = pickle.load(handle)
with open('./training/offline/climate/metafeatures_climate_3.pickle', 'rb') as handle:
    meta_features_190 = pickle.load(handle)
    
with open('./training/offline/climate/results_roc_climate_4.pickle', 'rb') as handle:
    results_roc_191 = pickle.load(handle)
with open('./training/offline/climate/results_prc_climate_4.pickle', 'rb') as handle:
    results_prc_191 = pickle.load(handle)
with open('./training/offline/climate/results_f1_climate_4.pickle', 'rb') as handle:
    results_f1_191 = pickle.load(handle)
with open('./training/offline/climate/metafeatures_climate_4.pickle', 'rb') as handle:
    meta_features_191 = pickle.load(handle)

#==============================================================================
#==========================Dataset 40==========================================
#==============================================================================
with open('./training/offline/relax/results_roc_relax_0.pickle', 'rb') as handle:
    results_roc_192 = pickle.load(handle)
with open('./training/offline/relax/results_prc_relax_0.pickle', 'rb') as handle:
    results_prc_192 = pickle.load(handle)
with open('./training/offline/relax/results_f1_relax_0.pickle', 'rb') as handle:
    results_f1_192 = pickle.load(handle)
with open('./training/offline/relax/metafeatures_relax_0.pickle', 'rb') as handle:
    meta_features_192 = pickle.load(handle)
    
with open('./training/offline/relax/results_roc_relax_1.pickle', 'rb') as handle:
    results_roc_193 = pickle.load(handle)
with open('./training/offline/relax/results_prc_relax_1.pickle', 'rb') as handle:
    results_prc_193 = pickle.load(handle)
with open('./training/offline/relax/results_f1_relax_1.pickle', 'rb') as handle:
    results_f1_193 = pickle.load(handle)
with open('./training/offline/relax/metafeatures_relax_1.pickle', 'rb') as handle:
    meta_features_193 = pickle.load(handle)
    
with open('./training/offline/relax/results_roc_relax_2.pickle', 'rb') as handle:
    results_roc_194 = pickle.load(handle)
with open('./training/offline/relax/results_prc_relax_2.pickle', 'rb') as handle:
    results_prc_194 = pickle.load(handle)
with open('./training/offline/relax/results_f1_relax_2.pickle', 'rb') as handle:
    results_f1_194 = pickle.load(handle)
with open('./training/offline/relax/metafeatures_relax_2.pickle', 'rb') as handle:
    meta_features_194 = pickle.load(handle)
    
with open('./training/offline/relax/results_roc_relax_3.pickle', 'rb') as handle:
    results_roc_195 = pickle.load(handle)
with open('./training/offline/relax/results_prc_relax_3.pickle', 'rb') as handle:
    results_prc_195 = pickle.load(handle)
with open('./training/offline/relax/results_f1_relax_3.pickle', 'rb') as handle:
    results_f1_195 = pickle.load(handle)
with open('./training/offline/relax/metafeatures_relax_3.pickle', 'rb') as handle:
    meta_features_195 = pickle.load(handle)
    
with open('./training/offline/relax/results_roc_relax_4.pickle', 'rb') as handle:
    results_roc_196 = pickle.load(handle)
with open('./training/offline/relax/results_prc_relax_4.pickle', 'rb') as handle:
    results_prc_196 = pickle.load(handle)
with open('./training/offline/relax/results_f1_relax_4.pickle', 'rb') as handle:
    results_f1_196 = pickle.load(handle)
with open('./training/offline/relax/metafeatures_relax_4.pickle', 'rb') as handle:
    meta_features_196 = pickle.load(handle)

#==============================================================================
#==========================Dataset 41==========================================
#==============================================================================
with open('./training/offline/liver/results_roc_liver_0.pickle', 'rb') as handle:
    results_roc_197 = pickle.load(handle)
with open('./training/offline/liver/results_prc_liver_0.pickle', 'rb') as handle:
    results_prc_197 = pickle.load(handle)
with open('./training/offline/liver/results_f1_liver_0.pickle', 'rb') as handle:
    results_f1_197 = pickle.load(handle)
with open('./training/offline/liver/metafeatures_liver_0.pickle', 'rb') as handle:
    meta_features_197 = pickle.load(handle)
    
with open('./training/offline/liver/results_roc_liver_1.pickle', 'rb') as handle:
    results_roc_198 = pickle.load(handle)
with open('./training/offline/liver/results_prc_liver_1.pickle', 'rb') as handle:
    results_prc_198 = pickle.load(handle)
with open('./training/offline/liver/results_f1_liver_1.pickle', 'rb') as handle:
    results_f1_198 = pickle.load(handle)
with open('./training/offline/liver/metafeatures_liver_1.pickle', 'rb') as handle:
    meta_features_198 = pickle.load(handle)
    
with open('./training/offline/liver/results_roc_liver_2.pickle', 'rb') as handle:
    results_roc_199 = pickle.load(handle)
with open('./training/offline/liver/results_prc_liver_2.pickle', 'rb') as handle:
    results_prc_199 = pickle.load(handle)
with open('./training/offline/liver/results_f1_liver_2.pickle', 'rb') as handle:
    results_f1_199 = pickle.load(handle)
with open('./training/offline/liver/metafeatures_liver_2.pickle', 'rb') as handle:
    meta_features_199 = pickle.load(handle)
    
with open('./training/offline/liver/results_roc_liver_3.pickle', 'rb') as handle:
    results_roc_200 = pickle.load(handle)
with open('./training/offline/liver/results_prc_liver_3.pickle', 'rb') as handle:
    results_prc_200 = pickle.load(handle)
with open('./training/offline/liver/results_f1_liver_3.pickle', 'rb') as handle:
    results_f1_200 = pickle.load(handle)
with open('./training/offline/liver/metafeatures_liver_3.pickle', 'rb') as handle:
    meta_features_200 = pickle.load(handle)
    
with open('./training/offline/liver/results_roc_liver_4.pickle', 'rb') as handle:
    results_roc_201 = pickle.load(handle)
with open('./training/offline/liver/results_prc_liver_4.pickle', 'rb') as handle:
    results_prc_201 = pickle.load(handle)
with open('./training/offline/liver/results_f1_liver_4.pickle', 'rb') as handle:
    results_f1_201 = pickle.load(handle)
with open('./training/offline/liver/metafeatures_liver_4.pickle', 'rb') as handle:
    meta_features_201 = pickle.load(handle)
    
#==============================================================================
#===========================RESULTS============================================
#==============================================================================
    
#Nested results
    
nested_results_roc = {0:results_roc_0, 1:results_roc_1, 2: results_roc_2, 3: results_roc_3, 4: results_roc_4, 5: results_roc_5, 6: results_roc_6, 7: results_roc_7, 8: results_roc_8, 9: results_roc_9,
                      10: results_roc_10, 11: results_roc_11, 12: results_roc_12, 13: results_roc_13, 14: results_roc_14, 15: results_roc_15,   16: results_roc_16, 17: results_roc_17, 18: results_roc_18, 19: results_roc_19,
                      20: results_roc_20, 21: results_roc_21, 22: results_roc_22, 23: results_roc_23, 24: results_roc_24, 25: results_roc_25, 26: results_roc_26, 27: results_roc_27, 28: results_roc_28, 29: results_roc_29,
                      30: results_roc_30, 31: results_roc_31, 32: results_roc_32, 33: results_roc_33, 34: results_roc_34, 35: results_roc_35, 36: results_roc_36, 37: results_roc_37, 38: results_roc_38, 39: results_roc_39,
                      40: results_roc_40, 41: results_roc_41, 42: results_roc_42, 43: results_roc_43, 44: results_roc_44, 45: results_roc_45, 46: results_roc_46, 47: results_roc_47, 48: results_roc_48, 49: results_roc_49,
                      50: results_roc_50, 51: results_roc_51, 52: results_roc_52, 53: results_roc_53, 54: results_roc_54, 55: results_roc_55, 56: results_roc_56, 57: results_roc_57, 58: results_roc_58, 59: results_roc_59,
                      60: results_roc_60, 61: results_roc_61, 62: results_roc_62, 63: results_roc_63, 64: results_roc_64, 65: results_roc_65, 66: results_roc_66, 67: results_roc_67, 68: results_roc_68, 69: results_roc_69,
                      70: results_roc_70, 71: results_roc_71, 72: results_roc_72, 73: results_roc_73, 74: results_roc_74, 75: results_roc_75, 76: results_roc_76, 77: results_roc_77, 78: results_roc_78, 79: results_roc_79,
                      80: results_roc_80, 81: results_roc_81, 82: results_roc_82, 83: results_roc_83, 84: results_roc_84, 85: results_roc_85, 86: results_roc_86, 87: results_roc_87, 88: results_roc_88, 89: results_roc_89,
                      90: results_roc_90, 91: results_roc_91, 92: results_roc_92, 93: results_roc_93, 94: results_roc_94, 95: results_roc_95, 96: results_roc_96, 97: results_roc_97, 98: results_roc_98, 99: results_roc_99,
                      100: results_roc_100, 101: results_roc_101, 102: results_roc_102, 103: results_roc_103, 104: results_roc_104, 105: results_roc_105, 106: results_roc_106, 107: results_roc_107, 108: results_roc_108, 109: results_roc_109,
                      110: results_roc_110, 111: results_roc_111, 112: results_roc_112, 113: results_roc_113, 114: results_roc_114, 115: results_roc_115, 116: results_roc_116, 117: results_roc_117, 118: results_roc_118, 119: results_roc_119,
                      120: results_roc_120, 121: results_roc_121, 122: results_roc_122, 123: results_roc_123, 124: results_roc_124, 125: results_roc_125, 126: results_roc_126, 127: results_roc_127, 128: results_roc_128, 129: results_roc_129,
                      130: results_roc_130, 131: results_roc_131, 132: results_roc_132, 133: results_roc_133, 134: results_roc_134, 135: results_roc_135, 136: results_roc_136, 137: results_roc_137, 138: results_roc_138, 139: results_roc_139,
                      140: results_roc_140, 141: results_roc_141, 142: results_roc_142, 143: results_roc_143, 144: results_roc_144, 145: results_roc_145, 146: results_roc_146, 147: results_roc_147, 148: results_roc_148, 149: results_roc_149,
                      150: results_roc_150, 151: results_roc_151, 152: results_roc_152, 153: results_roc_153, 154: results_roc_154, 155: results_roc_155, 156: results_roc_156, 157: results_roc_157, 158: results_roc_158, 159: results_roc_159,
                      160: results_roc_160, 161: results_roc_161, 162: results_roc_162, 163: results_roc_163, 164: results_roc_164, 165: results_roc_165, 166: results_roc_166, 167: results_roc_167, 168: results_roc_168, 169: results_roc_169,
                      170: results_roc_170, 171: results_roc_171, 172: results_roc_172, 173: results_roc_173, 174: results_roc_174, 175: results_roc_175, 176: results_roc_177, 177: results_roc_177, 178: results_roc_178, 179: results_roc_179,
                      180: results_roc_180, 181: results_roc_181, 182: results_roc_182, 183: results_roc_183, 184: results_roc_184, 185: results_roc_185, 186: results_roc_188, 187: results_roc_188, 188: results_roc_188, 189: results_roc_189,
                      190: results_roc_190, 191: results_roc_191, 192: results_roc_192, 193: results_roc_193, 194: results_roc_194, 195: results_roc_195, 196: results_roc_199, 197: results_roc_199, 198: results_roc_199, 199: results_roc_199,
                      200: results_roc_200, 201: results_roc_201}

nested_results_prc = {0:results_prc_0, 1:results_prc_1, 2: results_prc_2, 3: results_prc_3, 4: results_prc_4, 5: results_prc_5, 6: results_prc_6, 7: results_prc_7, 8: results_prc_8, 9: results_prc_9,
                      10: results_prc_10, 11: results_prc_11, 12: results_prc_12, 13: results_prc_13, 14: results_prc_14, 15: results_prc_15,   16: results_prc_16, 17: results_prc_17, 18: results_prc_18, 19: results_prc_19,
                      20: results_prc_20, 21: results_prc_21, 22: results_prc_22, 23: results_prc_23, 24: results_prc_24, 25: results_prc_25, 26: results_prc_26, 27: results_prc_27, 28: results_prc_28, 29: results_prc_29,
                      30: results_prc_30, 31: results_prc_31, 32: results_prc_32, 33: results_prc_33, 34: results_prc_34, 35: results_prc_35, 36: results_prc_36, 37: results_prc_37, 38: results_prc_38, 39: results_prc_39,
                      40: results_prc_40, 41: results_prc_41, 42: results_prc_42, 43: results_prc_43, 44: results_prc_44, 45: results_prc_45, 46: results_prc_46, 47: results_prc_47, 48: results_prc_48, 49: results_prc_49,
                      50: results_prc_50, 51: results_prc_51, 52: results_prc_52, 53: results_prc_53, 54: results_prc_54, 55: results_prc_55, 56: results_prc_56, 57: results_prc_57, 58: results_prc_58, 59: results_prc_59,
                      60: results_prc_60, 61: results_prc_61, 62: results_prc_62, 63: results_prc_63, 64: results_prc_64, 65: results_prc_65, 66: results_prc_66, 67: results_prc_67, 68: results_prc_68, 69: results_prc_69,
                      70: results_prc_70, 71: results_prc_71, 72: results_prc_72, 73: results_prc_73, 74: results_prc_74, 75: results_prc_75, 76: results_prc_76, 77: results_prc_77, 78: results_prc_78, 79: results_prc_79,
                      80: results_prc_80, 81: results_prc_81, 82: results_prc_82, 83: results_prc_83, 84: results_prc_84, 85: results_prc_85, 86: results_prc_86, 87: results_prc_87, 88: results_prc_88, 89: results_prc_89,
                      90: results_prc_90, 91: results_prc_91, 92: results_prc_92, 93: results_prc_93, 94: results_prc_94, 95: results_prc_95, 96: results_prc_96, 97: results_prc_97, 98: results_prc_98, 99: results_prc_99,
                      100: results_prc_100, 101: results_prc_101, 102: results_prc_102, 103: results_prc_103, 104: results_prc_104, 105: results_prc_105, 106: results_prc_106, 107: results_prc_107, 108: results_prc_108, 109: results_prc_109,
                      110: results_prc_110, 111: results_prc_111, 112: results_prc_112, 113: results_prc_113, 114: results_prc_114, 115: results_prc_115, 116: results_prc_116, 117: results_prc_117, 118: results_prc_118, 119: results_prc_119,
                      120: results_prc_120, 121: results_prc_121, 122: results_prc_122, 123: results_prc_123, 124: results_prc_124, 125: results_prc_125, 126: results_prc_126, 127: results_prc_127, 128: results_prc_128, 129: results_prc_129,
                      130: results_prc_130, 131: results_prc_131, 132: results_prc_132, 133: results_prc_133, 134: results_prc_134, 135: results_prc_135, 136: results_prc_136, 137: results_prc_137, 138: results_prc_138, 139: results_prc_139,
                      140: results_prc_140, 141: results_prc_141, 142: results_prc_142, 143: results_prc_143, 144: results_prc_144, 145: results_prc_145, 146: results_prc_146, 147: results_prc_147, 148: results_prc_148, 149: results_prc_149,
                      150: results_prc_150, 151: results_prc_151, 152: results_prc_152, 153: results_prc_153, 154: results_prc_154, 155: results_prc_155, 156: results_prc_156, 157: results_prc_157, 158: results_prc_158, 159: results_prc_159,
                      160: results_prc_160, 161: results_prc_161, 162: results_prc_162, 163: results_prc_163, 164: results_prc_164, 165: results_prc_165, 166: results_prc_166, 167: results_prc_167, 168: results_prc_168, 169: results_prc_169,
                      170: results_prc_170, 171: results_prc_171, 172: results_prc_172, 173: results_prc_173, 174: results_prc_174, 175: results_prc_175, 176: results_prc_176, 177: results_prc_177, 178: results_prc_178, 179: results_prc_179,
                      180: results_prc_180, 181: results_prc_181, 182: results_prc_182, 183: results_prc_183, 184: results_prc_184, 185: results_prc_185, 186: results_prc_186, 187: results_prc_187, 188: results_prc_188, 189: results_prc_189,
                      190: results_prc_190, 191: results_prc_191, 192: results_prc_192, 193: results_prc_193, 194: results_prc_194, 195: results_prc_195, 196: results_prc_196, 197: results_prc_197, 198: results_prc_198, 199: results_prc_199,
                      200: results_prc_200, 201: results_prc_201}


nested_results_f1 = {0:results_f1_0, 1:results_f1_1, 2: results_f1_2, 3: results_f1_3, 4: results_f1_4, 5: results_f1_5, 6: results_f1_6, 7: results_f1_7, 8: results_f1_8, 9: results_f1_9,
                      10: results_f1_10, 11: results_f1_11, 12: results_f1_12, 13: results_f1_13, 14: results_f1_14, 15: results_f1_15,   16: results_f1_16, 17: results_f1_17, 18: results_f1_18, 19: results_f1_19,
                      20: results_f1_20, 21: results_f1_21, 22: results_f1_22, 23: results_f1_23, 24: results_f1_24, 25: results_f1_25, 26: results_f1_26, 27: results_f1_27, 28: results_f1_28, 29: results_f1_29,
                      30: results_f1_30, 31: results_f1_31, 32: results_f1_32, 33: results_f1_33, 34: results_f1_34, 35: results_f1_35, 36: results_f1_36, 37: results_f1_37, 38: results_f1_38, 39: results_f1_39,
                      40: results_f1_40, 41: results_f1_41, 42: results_f1_42, 43: results_f1_43, 44: results_f1_44, 45: results_f1_45, 46: results_f1_46, 47: results_f1_47, 48: results_f1_48, 49: results_f1_49,
                      50: results_f1_50, 51: results_f1_51, 52: results_f1_52, 53: results_f1_53, 54: results_f1_54, 55: results_f1_55, 56: results_f1_56, 57: results_f1_57, 58: results_f1_58, 59: results_f1_59,
                      60: results_f1_60, 61: results_f1_61, 62: results_f1_62, 63: results_f1_63, 64: results_f1_64, 65: results_f1_65, 66: results_f1_66, 67: results_f1_67, 68: results_f1_68, 69: results_f1_69,
                      70: results_f1_70, 71: results_f1_71, 72: results_f1_72, 73: results_f1_73, 74: results_f1_74, 75: results_f1_75, 76: results_f1_76, 77: results_f1_77, 78: results_f1_78, 79: results_f1_79,
                      80: results_f1_80, 81: results_f1_81, 82: results_f1_82, 83: results_f1_83, 84: results_f1_84, 85: results_f1_85, 86: results_f1_86, 87: results_f1_87, 88: results_f1_88, 89: results_f1_89,
                      90: results_f1_90, 91: results_f1_91, 92: results_f1_92, 93: results_f1_93, 94: results_f1_94, 95: results_f1_95, 96: results_f1_96, 97: results_f1_97, 98: results_f1_98, 99: results_f1_99,
                      100: results_f1_100, 101: results_f1_101, 102: results_f1_102, 103: results_f1_103, 104: results_f1_104, 105: results_f1_105, 106: results_f1_106, 107: results_f1_107, 108: results_f1_108, 109: results_f1_109,
                      110: results_f1_110, 111: results_f1_111, 112: results_f1_112, 113: results_f1_113, 114: results_f1_114, 115: results_f1_115, 116: results_f1_116, 117: results_f1_117, 118: results_f1_118, 119: results_f1_119,
                      120: results_f1_120, 121: results_f1_121, 122: results_f1_122, 123: results_f1_123, 124: results_f1_124, 125: results_f1_125, 126: results_f1_126, 127: results_f1_127, 128: results_f1_128, 129: results_f1_129,
                      130: results_f1_130, 131: results_f1_131, 132: results_f1_132, 133: results_f1_133, 134: results_f1_134, 135: results_f1_135, 136: results_f1_136, 137: results_f1_137, 138: results_f1_138, 139: results_f1_139,
                      140: results_f1_140, 141: results_f1_141, 142: results_f1_142, 143: results_f1_143, 144: results_f1_144, 145: results_f1_145, 146: results_f1_146, 147: results_f1_147, 148: results_f1_148, 149: results_f1_149,
                      150: results_f1_150, 151: results_f1_151, 152: results_f1_152, 153: results_f1_153, 154: results_f1_154, 155: results_f1_155, 156: results_f1_156, 157: results_f1_157, 158: results_f1_158, 159: results_f1_159,
                      160: results_f1_160, 161: results_f1_161, 162: results_f1_162, 163: results_f1_163, 164: results_f1_164, 165: results_f1_165, 166: results_f1_166, 167: results_f1_167, 168: results_f1_168, 169: results_f1_169,
                      170: results_f1_170, 171: results_f1_171, 172: results_f1_172, 173: results_f1_173, 174: results_f1_174, 175: results_f1_175, 176: results_f1_177, 177: results_f1_177, 178: results_f1_178, 179: results_f1_179,
                      180: results_f1_180, 181: results_f1_181, 182: results_f1_182, 183: results_f1_183, 184: results_f1_184, 185: results_f1_185, 186: results_f1_188, 187: results_f1_188, 188: results_f1_188, 189: results_f1_189,
                      190: results_f1_190, 191: results_f1_191, 192: results_f1_192, 193: results_f1_193, 194: results_f1_194, 195: results_f1_195, 196: results_f1_199, 197: results_f1_199, 198: results_f1_199, 199: results_f1_199,
                      200: results_f1_200, 201: results_f1_201}

#Save results

with open('.test/nested_results_roc.pickle', 'wb') as handle:
    pickle.dump(nested_results_roc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('.test/nested_results_prc.pickle', 'wb') as handle:
    pickle.dump(nested_results_prc, handle, protocol=pickle.HIGHEST_PROTOCOL)    
with open('.test/nested_results_f1.pickle', 'wb') as handle:
    pickle.dump(nested_results_f1, handle, protocol=pickle.HIGHEST_PROTOCOL)    

#Create a dataframe and add the metafeatures
df_results = pd.DataFrame([meta_features_0, meta_features_1, meta_features_2, meta_features_3, meta_features_4, meta_features_5, meta_features_6, meta_features_7, meta_features_8, meta_features_9,
                            meta_features_10, meta_features_11, meta_features_12, meta_features_13, meta_features_14, meta_features_15, meta_features_16, meta_features_17, meta_features_18,meta_features_19,
                            meta_features_20, meta_features_21, meta_features_22, meta_features_23, meta_features_24, meta_features_25, meta_features_26, meta_features_27, meta_features_28,meta_features_29,
                            meta_features_30, meta_features_31, meta_features_32, meta_features_33, meta_features_34, meta_features_35, meta_features_36, meta_features_37, meta_features_38,meta_features_39,
                            meta_features_40, meta_features_41, meta_features_42, meta_features_43, meta_features_44, meta_features_45, meta_features_46, meta_features_47, meta_features_48,meta_features_49,
                            meta_features_50, meta_features_51, meta_features_52, meta_features_53, meta_features_54, meta_features_55, meta_features_56, meta_features_57, meta_features_58,meta_features_59,
                            meta_features_60, meta_features_61, meta_features_62, meta_features_63, meta_features_64, meta_features_65, meta_features_66, meta_features_67, meta_features_68,meta_features_69,
                            meta_features_70, meta_features_71, meta_features_72, meta_features_73, meta_features_74, meta_features_75, meta_features_76, meta_features_77, meta_features_78,meta_features_79,
                            meta_features_80, meta_features_81, meta_features_82, meta_features_83, meta_features_84, meta_features_85, meta_features_86, meta_features_87, meta_features_88,meta_features_89,
                            meta_features_90, meta_features_91, meta_features_92, meta_features_93, meta_features_94, meta_features_95, meta_features_96, meta_features_97, meta_features_98,meta_features_99,
                            meta_features_100, meta_features_101, meta_features_102, meta_features_103, meta_features_104, meta_features_105, meta_features_106, meta_features_107, meta_features_108, meta_features_109,
                            meta_features_110, meta_features_111, meta_features_112, meta_features_113, meta_features_114, meta_features_115, meta_features_116, meta_features_117, meta_features_118, meta_features_119,
                            meta_features_120, meta_features_121, meta_features_122, meta_features_123, meta_features_124, meta_features_125, meta_features_126, meta_features_127, meta_features_128, meta_features_129,
                            meta_features_130, meta_features_131, meta_features_132, meta_features_133, meta_features_134, meta_features_135, meta_features_136, meta_features_137, meta_features_138, meta_features_139,
                            meta_features_140, meta_features_141, meta_features_142, meta_features_143, meta_features_144, meta_features_145, meta_features_146, meta_features_147, meta_features_148, meta_features_149,
                            meta_features_150, meta_features_151, meta_features_152, meta_features_153, meta_features_154, meta_features_155, meta_features_156, meta_features_157, meta_features_158, meta_features_159,
                            meta_features_160, meta_features_161, meta_features_162, meta_features_163, meta_features_164, meta_features_165, meta_features_166, meta_features_167, meta_features_168, meta_features_169,
                            meta_features_170, meta_features_171, meta_features_172, meta_features_173, meta_features_174, meta_features_175, meta_features_176, meta_features_177, meta_features_178, meta_features_179,
                            meta_features_180, meta_features_181, meta_features_182, meta_features_183, meta_features_184, meta_features_185, meta_features_186, meta_features_187, meta_features_188, meta_features_189,
                            meta_features_190, meta_features_191, meta_features_192, meta_features_193, meta_features_194, meta_features_195, meta_features_196, meta_features_197, meta_features_198, meta_features_199,
                            meta_features_200, meta_features_201])

#Set the index of the dataframe as the dataset name: "Dataset 1", etc.
df_results.index = ['Dataset 0', 'Dataset 1','Dataset 2','Dataset 3','Dataset 4', 'Dataset 5', 'Dataset 6', 'Dataset 7','Dataset 8','Dataset 9', 
                    'Dataset 10', 'Dataset 11', 'Dataset 12', 'Dataset 13','Dataset 14','Dataset 15','Dataset 16','Dataset 17', 'Dataset 18','Dataset 19',
                    'Dataset 20', 'Dataset 21', 'Dataset 22', 'Dataset 23','Dataset 24','Dataset 25','Dataset 26','Dataset 27', 'Dataset 28','Dataset 29',
                    'Dataset 30', 'Dataset 31', 'Dataset 32', 'Dataset 33','Dataset 34','Dataset 35','Dataset 36','Dataset 37', 'Dataset 38','Dataset 39',
                    'Dataset 40', 'Dataset 41', 'Dataset 42', 'Dataset 43','Dataset 44','Dataset 45','Dataset 46','Dataset 47', 'Dataset 48','Dataset 49',
                    'Dataset 50', 'Dataset 51', 'Dataset 52', 'Dataset 53','Dataset 54','Dataset 55','Dataset 56','Dataset 57', 'Dataset 58','Dataset 59',
                    'Dataset 60', 'Dataset 61', 'Dataset 62', 'Dataset 63','Dataset 64','Dataset 65','Dataset 66','Dataset 67', 'Dataset 68','Dataset 69',
                    'Dataset 70', 'Dataset 71', 'Dataset 72', 'Dataset 73','Dataset 74','Dataset 75','Dataset 76','Dataset 77', 'Dataset 78','Dataset 79',
                    'Dataset 80', 'Dataset 81', 'Dataset 82', 'Dataset 83','Dataset 84','Dataset 85','Dataset 86','Dataset 87', 'Dataset 88','Dataset 89',
                    'Dataset 90', 'Dataset 91', 'Dataset 92', 'Dataset 93','Dataset 94','Dataset 95','Dataset 96','Dataset 97', 'Dataset 98','Dataset 99',
                    'Dataset 100', 'Dataset 101', 'Dataset 102', 'Dataset 103', 'Dataset 104', 'Dataset 105', 'Dataset 106', 'Dataset 107', 'Dataset 108', 'Dataset 109',
                    'Dataset 110', 'Dataset 111', 'Dataset 112', 'Dataset 113', 'Dataset 114', 'Dataset 115', 'Dataset 116', 'Dataset 117', 'Dataset 118', 'Dataset 119',
                    'Dataset 120', 'Dataset 121', 'Dataset 122', 'Dataset 123', 'Dataset 124', 'Dataset 125', 'Dataset 126', 'Dataset 127', 'Dataset 128', 'Dataset 129',
                    'Dataset 130', 'Dataset 131', 'Dataset 132', 'Dataset 133', 'Dataset 134', 'Dataset 135', 'Dataset 136', 'Dataset 137', 'Dataset 138', 'Dataset 139',
                    'Dataset 140', 'Dataset 141', 'Dataset 142', 'Dataset 143', 'Dataset 144', 'Dataset 145', 'Dataset 146', 'Dataset 147', 'Dataset 148', 'Dataset 149',
                    'Dataset 150', 'Dataset 151', 'Dataset 152', 'Dataset 153', 'Dataset 154', 'Dataset 155', 'Dataset 156', 'Dataset 157', 'Dataset 158', 'Dataset 159',
                    'Dataset 160', 'Dataset 161', 'Dataset 162', 'Dataset 163', 'Dataset 164', 'Dataset 165', 'Dataset 166', 'Dataset 167', 'Dataset 168', 'Dataset 169',
                    'Dataset 170', 'Dataset 171', 'Dataset 172', 'Dataset 173', 'Dataset 174', 'Dataset 175', 'Dataset 176', 'Dataset 177', 'Dataset 178', 'Dataset 179',
                    'Dataset 180', 'Dataset 181', 'Dataset 182', 'Dataset 183', 'Dataset 184', 'Dataset 185', 'Dataset 186', 'Dataset 187', 'Dataset 188', 'Dataset 189',
                    'Dataset 190', 'Dataset 191', 'Dataset 192', 'Dataset 193', 'Dataset 194', 'Dataset 195', 'Dataset 196', 'Dataset 197', 'Dataset 198', 'Dataset 199',
                    'Dataset 200', 'Dataset 201']

#==============================================================================
#                                 AUC-ROC
#==============================================================================
#Get the best performance (e.g. 0.975) for each dataset
[max_roc_0, max_roc_1, max_roc_2, max_roc_3, max_roc_4, max_roc_5, max_roc_6, max_roc_7, max_roc_8, max_roc_9, 
max_roc_10, max_roc_11, max_roc_12, max_roc_13, max_roc_14, max_roc_15,max_roc_16, max_roc_17, max_roc_18, max_roc_19, 
max_roc_20, max_roc_21, max_roc_22, max_roc_23, max_roc_24, max_roc_25,max_roc_26, max_roc_27, max_roc_28, max_roc_29, 
max_roc_30, max_roc_31, max_roc_32, max_roc_33, max_roc_34, max_roc_35,max_roc_36, max_roc_37, max_roc_38, max_roc_39, 
max_roc_40, max_roc_41, max_roc_42, max_roc_43, max_roc_44, max_roc_45,max_roc_46, max_roc_47, max_roc_48, max_roc_49, 
max_roc_50, max_roc_51, max_roc_52, max_roc_53, max_roc_54, max_roc_55,max_roc_56, max_roc_57, max_roc_58, max_roc_59, 
max_roc_60, max_roc_61, max_roc_62, max_roc_63, max_roc_64, max_roc_65,max_roc_66, max_roc_67, max_roc_68, max_roc_69, 
max_roc_70, max_roc_71, max_roc_72, max_roc_73, max_roc_74, max_roc_75,max_roc_76, max_roc_77, max_roc_78, max_roc_79, 
max_roc_80, max_roc_81, max_roc_82, max_roc_83, max_roc_84, max_roc_85,max_roc_86, max_roc_87, max_roc_88, max_roc_89, 
max_roc_90, max_roc_91, max_roc_92, max_roc_93, max_roc_94, max_roc_95,max_roc_96, max_roc_97, max_roc_98, max_roc_99,
max_roc_100, max_roc_101, max_roc_102, max_roc_103, max_roc_104, max_roc_105, max_roc_106, max_roc_107, max_roc_108, max_roc_109,
max_roc_110, max_roc_111, max_roc_112, max_roc_113, max_roc_114, max_roc_115, max_roc_116, max_roc_117, max_roc_118, max_roc_119,
max_roc_120, max_roc_121, max_roc_122, max_roc_123, max_roc_124, max_roc_125, max_roc_126, max_roc_127, max_roc_128, max_roc_129,
max_roc_130, max_roc_131, max_roc_132, max_roc_133, max_roc_134, max_roc_135, max_roc_136, max_roc_137, max_roc_138, max_roc_139, 
max_roc_140, max_roc_141, max_roc_142, max_roc_143, max_roc_144, max_roc_145, max_roc_146, max_roc_147, max_roc_148, max_roc_149,
max_roc_150, max_roc_151, max_roc_152, max_roc_153, max_roc_154, max_roc_155, max_roc_156, max_roc_157, max_roc_158, max_roc_159,
max_roc_160, max_roc_161, max_roc_162, max_roc_163, max_roc_164, max_roc_165, max_roc_166, max_roc_167, max_roc_168, max_roc_169,
max_roc_170, max_roc_171, max_roc_172, max_roc_173, max_roc_174, max_roc_175, max_roc_176, max_roc_177, max_roc_178, max_roc_179,
max_roc_180, max_roc_181, max_roc_182, max_roc_183, max_roc_184, max_roc_185, max_roc_186, max_roc_187, max_roc_188, max_roc_189,
max_roc_190, max_roc_191, max_roc_192, max_roc_193, max_roc_194, max_roc_195, max_roc_196, max_roc_197, max_roc_198, max_roc_199,
max_roc_200, max_roc_201]                                                                                    = [max(results_roc_0.values()), max(results_roc_1.values()), max(results_roc_2.values()),max(results_roc_3.values()), max(results_roc_4.values()),max(results_roc_5.values()),max(results_roc_6.values()),max(results_roc_7.values()), max(results_roc_8.values()),max(results_roc_9.values()), 
                                                                                                                          max(results_roc_10.values()),max(results_roc_11.values()), max(results_roc_12.values()),max(results_roc_13.values()),max(results_roc_14.values()), max(results_roc_15.values()), max(results_roc_16.values()), max(results_roc_17.values()),max(results_roc_18.values()),max(results_roc_19.values()),
                                                                                                                          max(results_roc_20.values()),max(results_roc_21.values()), max(results_roc_22.values()),max(results_roc_23.values()),max(results_roc_24.values()), max(results_roc_25.values()), max(results_roc_26.values()), max(results_roc_27.values()),max(results_roc_28.values()),max(results_roc_29.values()),
                                                                                                                          max(results_roc_30.values()),max(results_roc_31.values()), max(results_roc_32.values()),max(results_roc_33.values()),max(results_roc_34.values()), max(results_roc_35.values()), max(results_roc_36.values()), max(results_roc_37.values()),max(results_roc_38.values()),max(results_roc_39.values()),
                                                                                                                          max(results_roc_40.values()),max(results_roc_41.values()), max(results_roc_42.values()),max(results_roc_43.values()),max(results_roc_44.values()), max(results_roc_45.values()), max(results_roc_46.values()), max(results_roc_47.values()),max(results_roc_48.values()),max(results_roc_49.values()),
                                                                                                                          max(results_roc_50.values()),max(results_roc_51.values()), max(results_roc_52.values()),max(results_roc_53.values()),max(results_roc_54.values()), max(results_roc_55.values()), max(results_roc_56.values()), max(results_roc_57.values()),max(results_roc_58.values()),max(results_roc_59.values()),
                                                                                                                          max(results_roc_60.values()),max(results_roc_61.values()), max(results_roc_62.values()),max(results_roc_63.values()),max(results_roc_64.values()), max(results_roc_65.values()), max(results_roc_66.values()), max(results_roc_67.values()),max(results_roc_68.values()),max(results_roc_69.values()),
                                                                                                                          max(results_roc_70.values()),max(results_roc_71.values()), max(results_roc_72.values()),max(results_roc_73.values()),max(results_roc_74.values()), max(results_roc_75.values()), max(results_roc_76.values()), max(results_roc_77.values()),max(results_roc_78.values()),max(results_roc_79.values()),
                                                                                                                          max(results_roc_80.values()),max(results_roc_81.values()), max(results_roc_82.values()),max(results_roc_83.values()),max(results_roc_84.values()), max(results_roc_85.values()), max(results_roc_86.values()), max(results_roc_87.values()),max(results_roc_88.values()),max(results_roc_89.values()),
                                                                                                                          max(results_roc_90.values()),max(results_roc_91.values()), max(results_roc_92.values()),max(results_roc_93.values()),max(results_roc_94.values()), max(results_roc_95.values()), max(results_roc_96.values()), max(results_roc_97.values()),max(results_roc_98.values()),max(results_roc_99.values()),
                                                                                                                          max(results_roc_100.values()), max(results_roc_101.values()), max(results_roc_102.values()), max(results_roc_103.values()), max(results_roc_104.values()),max(results_roc_105.values()),max(results_roc_106.values()),max(results_roc_107.values()),max(results_roc_108.values()),max(results_roc_109.values()),
                                                                                                                          max(results_roc_110.values()), max(results_roc_111.values()), max(results_roc_112.values()), max(results_roc_113.values()), max(results_roc_114.values()),max(results_roc_115.values()),max(results_roc_116.values()),max(results_roc_117.values()),max(results_roc_118.values()),max(results_roc_119.values()),
                                                                                                                          max(results_roc_120.values()), max(results_roc_121.values()), max(results_roc_122.values()), max(results_roc_123.values()), max(results_roc_124.values()),max(results_roc_125.values()),max(results_roc_126.values()),max(results_roc_127.values()),max(results_roc_128.values()),max(results_roc_129.values()),
                                                                                                                          max(results_roc_130.values()), max(results_roc_131.values()), max(results_roc_132.values()), max(results_roc_133.values()), max(results_roc_134.values()), max(results_roc_135.values()), max(results_roc_136.values()), max(results_roc_137.values()), max(results_roc_138.values()), max(results_roc_139.values()),
                                                                                                                          max(results_roc_140.values()), max(results_roc_141.values()), max(results_roc_142.values()), max(results_roc_143.values()), max(results_roc_144.values()), max(results_roc_145.values()), max(results_roc_146.values()), max(results_roc_147.values()), max(results_roc_148.values()), max(results_roc_149.values()),
                                                                                                                          max(results_roc_150.values()), max(results_roc_151.values()), max(results_roc_152.values()), max(results_roc_153.values()), max(results_roc_154.values()), max(results_roc_155.values()), max(results_roc_156.values()), max(results_roc_157.values()), max(results_roc_158.values()), max(results_roc_159.values()),
                                                                                                                          max(results_roc_160.values()), max(results_roc_161.values()), max(results_roc_162.values()), max(results_roc_163.values()), max(results_roc_164.values()), max(results_roc_165.values()), max(results_roc_166.values()), max(results_roc_167.values()), max(results_roc_168.values()), max(results_roc_169.values()),
                                                                                                                          max(results_roc_170.values()), max(results_roc_171.values()), max(results_roc_172.values()), max(results_roc_173.values()), max(results_roc_174.values()), max(results_roc_175.values()), max(results_roc_176.values()), max(results_roc_177.values()), max(results_roc_178.values()), max(results_roc_179.values()),
                                                                                                                          max(results_roc_180.values()), max(results_roc_181.values()), max(results_roc_182.values()), max(results_roc_183.values()), max(results_roc_184.values()), max(results_roc_185.values()), max(results_roc_186.values()), max(results_roc_187.values()), max(results_roc_188.values()), max(results_roc_189.values()),
                                                                                                                          max(results_roc_190.values()), max(results_roc_191.values()), max(results_roc_192.values()), max(results_roc_193.values()), max(results_roc_194.values()), max(results_roc_195.values()), max(results_roc_196.values()), max(results_roc_197.values()), max(results_roc_198.values()), max(results_roc_199.values()),
                                                                                                                          max(results_roc_200.values()), max(results_roc_201.values())]

#Get the best performing model (e.g. 'auc_roc_rforest') for each dataset
[maximum_roc_0, maximum_roc_1, maximum_roc_2, maximum_roc_3, maximum_roc_4, maximum_roc_5, maximum_roc_6, maximum_roc_7, maximum_roc_8, maximum_roc_9, 
maximum_roc_10, maximum_roc_11, maximum_roc_12, maximum_roc_13, maximum_roc_14, maximum_roc_15,maximum_roc_16, maximum_roc_17, maximum_roc_18, maximum_roc_19, 
maximum_roc_20, maximum_roc_21, maximum_roc_22, maximum_roc_23, maximum_roc_24, maximum_roc_25,maximum_roc_26, maximum_roc_27, maximum_roc_28, maximum_roc_29, 
maximum_roc_30, maximum_roc_31, maximum_roc_32, maximum_roc_33, maximum_roc_34, maximum_roc_35,maximum_roc_36, maximum_roc_37, maximum_roc_38, maximum_roc_39, 
maximum_roc_40, maximum_roc_41, maximum_roc_42, maximum_roc_43, maximum_roc_44, maximum_roc_45,maximum_roc_46, maximum_roc_47, maximum_roc_48, maximum_roc_49, 
maximum_roc_50, maximum_roc_51, maximum_roc_52, maximum_roc_53, maximum_roc_54, maximum_roc_55,maximum_roc_56, maximum_roc_57, maximum_roc_58, maximum_roc_59, 
maximum_roc_60, maximum_roc_61, maximum_roc_62, maximum_roc_63, maximum_roc_64, maximum_roc_65,maximum_roc_66, maximum_roc_67, maximum_roc_68, maximum_roc_69, 
maximum_roc_70, maximum_roc_71, maximum_roc_72, maximum_roc_73, maximum_roc_74, maximum_roc_75,maximum_roc_76, maximum_roc_77, maximum_roc_78, maximum_roc_79, 
maximum_roc_80, maximum_roc_81, maximum_roc_82, maximum_roc_83, maximum_roc_84, maximum_roc_85,maximum_roc_86, maximum_roc_87, maximum_roc_88, maximum_roc_89, 
maximum_roc_90, maximum_roc_91, maximum_roc_92, maximum_roc_93, maximum_roc_94, maximum_roc_95,maximum_roc_96, maximum_roc_97, maximum_roc_98, maximum_roc_99,
maximum_roc_100, maximum_roc_101, maximum_roc_102, maximum_roc_103, maximum_roc_104, maximum_roc_105, maximum_roc_106, maximum_roc_107, maximum_roc_108, maximum_roc_109,
maximum_roc_110, maximum_roc_111, maximum_roc_112, maximum_roc_113, maximum_roc_114, maximum_roc_115, maximum_roc_116, maximum_roc_117, maximum_roc_118, maximum_roc_119,
maximum_roc_120, maximum_roc_121, maximum_roc_122, maximum_roc_123, maximum_roc_124, maximum_roc_125, maximum_roc_126, maximum_roc_127, maximum_roc_128, maximum_roc_129,
maximum_roc_130, maximum_roc_131, maximum_roc_132, maximum_roc_133, maximum_roc_134, maximum_roc_135, maximum_roc_136, maximum_roc_137, maximum_roc_138, maximum_roc_139,
maximum_roc_140, maximum_roc_141, maximum_roc_142, maximum_roc_143, maximum_roc_144, maximum_roc_145, maximum_roc_146, maximum_roc_147, maximum_roc_148, maximum_roc_149,
maximum_roc_150, maximum_roc_151, maximum_roc_152, maximum_roc_153, maximum_roc_154, maximum_roc_155, maximum_roc_156, maximum_roc_157, maximum_roc_158, maximum_roc_159,
maximum_roc_160, maximum_roc_161, maximum_roc_162, maximum_roc_163, maximum_roc_164, maximum_roc_165, maximum_roc_166, maximum_roc_167, maximum_roc_168, maximum_roc_169,
maximum_roc_170, maximum_roc_171, maximum_roc_172, maximum_roc_173, maximum_roc_174, maximum_roc_175, maximum_roc_176, maximum_roc_177, maximum_roc_178, maximum_roc_179,
maximum_roc_180, maximum_roc_181, maximum_roc_182, maximum_roc_183, maximum_roc_184, maximum_roc_185, maximum_roc_186, maximum_roc_187, maximum_roc_188, maximum_roc_189,
maximum_roc_190, maximum_roc_191, maximum_roc_192, maximum_roc_193, maximum_roc_194, maximum_roc_195, maximum_roc_196, maximum_roc_197, maximum_roc_198, maximum_roc_199,
maximum_roc_200, maximum_roc_201]                                                                                                                                 = [max(results_roc_0, key=results_roc_0.get), max(results_roc_1, key=results_roc_1.get), max(results_roc_2, key=results_roc_2.get), max(results_roc_3, key=results_roc_3.get), max(results_roc_4, key=results_roc_4.get), max(results_roc_5, key=results_roc_5.get), max(results_roc_6, key=results_roc_6.get), max(results_roc_7, key=results_roc_7.get), max(results_roc_8, key=results_roc_8.get), max(results_roc_9, key=results_roc_9.get),
                                                                                                                                                                    max(results_roc_10, key=results_roc_10.get), max(results_roc_11, key=results_roc_11.get), max(results_roc_12, key=results_roc_12.get), max(results_roc_13, key=results_roc_13.get), max(results_roc_14, key=results_roc_14.get), max(results_roc_15, key=results_roc_15.get), max(results_roc_16, key=results_roc_16.get), max(results_roc_17, key=results_roc_17.get), max(results_roc_18, key=results_roc_18.get), max(results_roc_19, key=results_roc_19.get),
                                                                                                                                                                    max(results_roc_20, key=results_roc_20.get), max(results_roc_21, key=results_roc_21.get), max(results_roc_22, key=results_roc_22.get), max(results_roc_23, key=results_roc_23.get), max(results_roc_24, key=results_roc_24.get), max(results_roc_25, key=results_roc_25.get), max(results_roc_26, key=results_roc_26.get), max(results_roc_27, key=results_roc_27.get), max(results_roc_28, key=results_roc_28.get), max(results_roc_29, key=results_roc_29.get),
                                                                                                                                                                    max(results_roc_30, key=results_roc_30.get), max(results_roc_31, key=results_roc_31.get), max(results_roc_32, key=results_roc_32.get), max(results_roc_33, key=results_roc_33.get), max(results_roc_34, key=results_roc_34.get), max(results_roc_35, key=results_roc_35.get), max(results_roc_36, key=results_roc_36.get), max(results_roc_37, key=results_roc_37.get), max(results_roc_38, key=results_roc_38.get), max(results_roc_39, key=results_roc_39.get),
                                                                                                                                                                    max(results_roc_40, key=results_roc_40.get), max(results_roc_41, key=results_roc_41.get), max(results_roc_42, key=results_roc_42.get), max(results_roc_43, key=results_roc_43.get), max(results_roc_44, key=results_roc_44.get), max(results_roc_45, key=results_roc_45.get), max(results_roc_46, key=results_roc_46.get), max(results_roc_47, key=results_roc_47.get), max(results_roc_48, key=results_roc_48.get), max(results_roc_49, key=results_roc_49.get),
                                                                                                                                                                    max(results_roc_50, key=results_roc_50.get), max(results_roc_51, key=results_roc_51.get), max(results_roc_52, key=results_roc_52.get), max(results_roc_53, key=results_roc_53.get), max(results_roc_54, key=results_roc_54.get), max(results_roc_55, key=results_roc_55.get), max(results_roc_56, key=results_roc_56.get), max(results_roc_57, key=results_roc_57.get), max(results_roc_58, key=results_roc_58.get), max(results_roc_59, key=results_roc_59.get),
                                                                                                                                                                    max(results_roc_60, key=results_roc_60.get), max(results_roc_61, key=results_roc_61.get), max(results_roc_62, key=results_roc_62.get), max(results_roc_63, key=results_roc_63.get), max(results_roc_64, key=results_roc_64.get), max(results_roc_65, key=results_roc_65.get), max(results_roc_66, key=results_roc_66.get), max(results_roc_67, key=results_roc_67.get), max(results_roc_68, key=results_roc_68.get), max(results_roc_69, key=results_roc_69.get), 
                                                                                                                                                                    max(results_roc_70, key=results_roc_70.get), max(results_roc_71, key=results_roc_71.get), max(results_roc_72, key=results_roc_72.get), max(results_roc_73, key=results_roc_73.get), max(results_roc_74, key=results_roc_74.get), max(results_roc_75, key=results_roc_75.get), max(results_roc_76, key=results_roc_76.get), max(results_roc_77, key=results_roc_77.get), max(results_roc_78, key=results_roc_78.get), max(results_roc_79, key=results_roc_79.get),
                                                                                                                                                                    max(results_roc_80, key=results_roc_80.get), max(results_roc_81, key=results_roc_81.get), max(results_roc_82, key=results_roc_82.get), max(results_roc_83, key=results_roc_83.get), max(results_roc_84, key=results_roc_84.get), max(results_roc_85, key=results_roc_85.get), max(results_roc_86, key=results_roc_86.get), max(results_roc_87, key=results_roc_87.get), max(results_roc_88, key=results_roc_88.get), max(results_roc_89, key=results_roc_89.get),
                                                                                                                                                                    max(results_roc_90, key=results_roc_90.get), max(results_roc_91, key=results_roc_91.get), max(results_roc_92, key=results_roc_92.get), max(results_roc_93, key=results_roc_93.get), max(results_roc_94, key=results_roc_94.get), max(results_roc_95, key=results_roc_95.get), max(results_roc_96, key=results_roc_96.get), max(results_roc_97, key=results_roc_97.get), max(results_roc_98, key=results_roc_98.get), max(results_roc_99, key=results_roc_99.get),
                                                                                                                                                                    max(results_roc_100, key=results_roc_100.get), max(results_roc_101, key=results_roc_101.get), max(results_roc_102, key=results_roc_102.get), max(results_roc_103, key=results_roc_103.get), max(results_roc_104, key=results_roc_104.get), max(results_roc_105, key=results_roc_105.get), max(results_roc_106, key=results_roc_106.get), max(results_roc_107, key=results_roc_107.get), max(results_roc_108, key=results_roc_108.get), max(results_roc_109, key=results_roc_109.get),
                                                                                                                                                                    max(results_roc_110, key=results_roc_110.get), max(results_roc_111, key=results_roc_111.get), max(results_roc_112, key=results_roc_112.get), max(results_roc_113, key=results_roc_113.get), max(results_roc_114, key=results_roc_114.get), max(results_roc_115, key=results_roc_115.get), max(results_roc_116, key=results_roc_116.get), max(results_roc_117, key=results_roc_117.get), max(results_roc_118, key=results_roc_118.get), max(results_roc_119, key=results_roc_119.get),
                                                                                                                                                                    max(results_roc_120, key=results_roc_120.get), max(results_roc_121, key=results_roc_121.get), max(results_roc_122, key=results_roc_122.get), max(results_roc_123, key=results_roc_123.get), max(results_roc_124, key=results_roc_124.get), max(results_roc_125, key=results_roc_125.get), max(results_roc_126, key=results_roc_126.get), max(results_roc_127, key=results_roc_127.get), max(results_roc_128, key=results_roc_128.get), max(results_roc_129, key=results_roc_129.get),
                                                                                                                                                                    max(results_roc_130, key=results_roc_130.get), max(results_roc_131, key=results_roc_131.get), max(results_roc_132, key=results_roc_132.get), max(results_roc_133, key=results_roc_133.get), max(results_roc_134, key=results_roc_134.get), max(results_roc_135, key=results_roc_135.get), max(results_roc_136, key=results_roc_136.get), max(results_roc_137, key=results_roc_137.get), max(results_roc_138, key=results_roc_138.get), max(results_roc_139, key=results_roc_139.get),
                                                                                                                                                                    max(results_roc_140, key=results_roc_140.get), max(results_roc_141, key=results_roc_141.get), max(results_roc_142, key=results_roc_142.get), max(results_roc_143, key=results_roc_143.get), max(results_roc_144, key=results_roc_144.get), max(results_roc_145, key=results_roc_145.get), max(results_roc_146, key=results_roc_146.get), max(results_roc_147, key=results_roc_147.get), max(results_roc_148, key=results_roc_148.get), max(results_roc_149, key=results_roc_149.get),
                                                                                                                                                                    max(results_roc_150, key=results_roc_150.get), max(results_roc_151, key=results_roc_151.get), max(results_roc_152, key=results_roc_152.get), max(results_roc_153, key=results_roc_153.get), max(results_roc_154, key=results_roc_154.get), max(results_roc_155, key=results_roc_155.get), max(results_roc_156, key=results_roc_156.get), max(results_roc_157, key=results_roc_157.get), max(results_roc_158, key=results_roc_158.get), max(results_roc_159, key=results_roc_159.get),
                                                                                                                                                                    max(results_roc_160, key=results_roc_160.get), max(results_roc_161, key=results_roc_161.get), max(results_roc_162, key=results_roc_162.get), max(results_roc_163, key=results_roc_163.get), max(results_roc_164, key=results_roc_164.get), max(results_roc_165, key=results_roc_165.get), max(results_roc_166, key=results_roc_166.get), max(results_roc_167, key=results_roc_167.get), max(results_roc_168, key=results_roc_168.get), max(results_roc_169, key=results_roc_169.get),
                                                                                                                                                                    max(results_roc_170, key=results_roc_170.get), max(results_roc_171, key=results_roc_171.get), max(results_roc_172, key=results_roc_172.get), max(results_roc_173, key=results_roc_173.get), max(results_roc_174, key=results_roc_174.get), max(results_roc_175, key=results_roc_175.get), max(results_roc_176, key=results_roc_176.get), max(results_roc_177, key=results_roc_177.get), max(results_roc_178, key=results_roc_178.get), max(results_roc_179, key=results_roc_179.get),
                                                                                                                                                                    max(results_roc_180, key=results_roc_180.get), max(results_roc_181, key=results_roc_181.get), max(results_roc_182, key=results_roc_182.get), max(results_roc_183, key=results_roc_183.get), max(results_roc_184, key=results_roc_184.get), max(results_roc_185, key=results_roc_185.get), max(results_roc_186, key=results_roc_186.get), max(results_roc_187, key=results_roc_187.get), max(results_roc_188, key=results_roc_188.get), max(results_roc_189, key=results_roc_189.get),
                                                                                                                                                                    max(results_roc_190, key=results_roc_190.get), max(results_roc_191, key=results_roc_191.get), max(results_roc_192, key=results_roc_192.get), max(results_roc_193, key=results_roc_193.get), max(results_roc_194, key=results_roc_194.get), max(results_roc_195, key=results_roc_195.get), max(results_roc_196, key=results_roc_196.get), max(results_roc_197, key=results_roc_197.get), max(results_roc_198, key=results_roc_198.get), max(results_roc_199, key=results_roc_199.get),
                                                                                                                                                                    max(results_roc_200, key=results_roc_200.get), max(results_roc_201, key=results_roc_201.get)]
                                                                                                                                       
#Add the results to the dataframe                                                                                                                             
df_results['Best model ROC'] = pd.Series([maximum_roc_0, maximum_roc_1,maximum_roc_2,maximum_roc_3,maximum_roc_4,maximum_roc_5,maximum_roc_6,maximum_roc_7,maximum_roc_8, maximum_roc_9,
                                          maximum_roc_10,maximum_roc_11,maximum_roc_12, maximum_roc_13,maximum_roc_14,maximum_roc_15,maximum_roc_16, maximum_roc_17, maximum_roc_18,maximum_roc_19,
                                          maximum_roc_20,maximum_roc_21,maximum_roc_22, maximum_roc_23,maximum_roc_24,maximum_roc_25,maximum_roc_26, maximum_roc_27, maximum_roc_28,maximum_roc_29,
                                          maximum_roc_30,maximum_roc_31,maximum_roc_32, maximum_roc_33,maximum_roc_34,maximum_roc_35,maximum_roc_36, maximum_roc_37, maximum_roc_38,maximum_roc_39,
                                          maximum_roc_40,maximum_roc_41,maximum_roc_42, maximum_roc_43,maximum_roc_44,maximum_roc_45,maximum_roc_46, maximum_roc_47, maximum_roc_48,maximum_roc_49,
                                          maximum_roc_50,maximum_roc_51,maximum_roc_52, maximum_roc_53,maximum_roc_54,maximum_roc_55,maximum_roc_56, maximum_roc_57, maximum_roc_58,maximum_roc_59,
                                          maximum_roc_60,maximum_roc_61,maximum_roc_62, maximum_roc_63,maximum_roc_64,maximum_roc_65,maximum_roc_66, maximum_roc_67, maximum_roc_68,maximum_roc_69,
                                          maximum_roc_70,maximum_roc_71,maximum_roc_72, maximum_roc_73,maximum_roc_74,maximum_roc_75,maximum_roc_76, maximum_roc_77, maximum_roc_78,maximum_roc_79,
                                          maximum_roc_80,maximum_roc_81,maximum_roc_82, maximum_roc_83,maximum_roc_84,maximum_roc_85,maximum_roc_86, maximum_roc_87, maximum_roc_88,maximum_roc_89,
                                          maximum_roc_90,maximum_roc_91,maximum_roc_92, maximum_roc_93,maximum_roc_94,maximum_roc_95,maximum_roc_96, maximum_roc_97, maximum_roc_98,maximum_roc_99,
                                          maximum_roc_100, maximum_roc_101, maximum_roc_102, maximum_roc_103, maximum_roc_104, maximum_roc_105, maximum_roc_106, maximum_roc_107, maximum_roc_108, maximum_roc_109,
                                          maximum_roc_110, maximum_roc_111, maximum_roc_112, maximum_roc_113, maximum_roc_114, maximum_roc_115, maximum_roc_116, maximum_roc_117, maximum_roc_118, maximum_roc_119,
                                          maximum_roc_120, maximum_roc_121, maximum_roc_122, maximum_roc_123, maximum_roc_124, maximum_roc_125, maximum_roc_126, maximum_roc_127, maximum_roc_128, maximum_roc_129,
                                          maximum_roc_130, maximum_roc_131, maximum_roc_132, maximum_roc_133, maximum_roc_134, maximum_roc_135, maximum_roc_136, maximum_roc_137, maximum_roc_138,  maximum_roc_139, 
                                          maximum_roc_140, maximum_roc_141, maximum_roc_142, maximum_roc_143, maximum_roc_144, maximum_roc_145, maximum_roc_146, maximum_roc_147, maximum_roc_148, maximum_roc_149,
                                          maximum_roc_150, maximum_roc_151, maximum_roc_152, maximum_roc_153, maximum_roc_154, maximum_roc_155, maximum_roc_156, maximum_roc_157, maximum_roc_158, maximum_roc_159,
                                          maximum_roc_160, maximum_roc_161, maximum_roc_162, maximum_roc_163, maximum_roc_164, maximum_roc_165, maximum_roc_166, maximum_roc_167, maximum_roc_168, maximum_roc_169,
                                          maximum_roc_170, maximum_roc_171, maximum_roc_172, maximum_roc_173, maximum_roc_174, maximum_roc_175, maximum_roc_176, maximum_roc_177, maximum_roc_178, maximum_roc_179,
                                          maximum_roc_180, maximum_roc_181, maximum_roc_182, maximum_roc_183, maximum_roc_184, maximum_roc_185, maximum_roc_186, maximum_roc_187, maximum_roc_188, maximum_roc_189,
                                          maximum_roc_190, maximum_roc_191, maximum_roc_192, maximum_roc_193, maximum_roc_194, maximum_roc_195, maximum_roc_196, maximum_roc_197, maximum_roc_198, maximum_roc_199,
                                          maximum_roc_200, maximum_roc_201], 
                                          index = df_results.index)

df_results['Maximum AUC-ROC'] = pd.Series([max_roc_0, max_roc_1,max_roc_2,max_roc_3,max_roc_4,max_roc_5, max_roc_6,max_roc_7,max_roc_8,max_roc_9,
                                            max_roc_10, max_roc_11, max_roc_12, max_roc_13, max_roc_14,max_roc_15, max_roc_16,max_roc_17, max_roc_18,max_roc_19,
                                            max_roc_20, max_roc_21, max_roc_22, max_roc_23, max_roc_24,max_roc_25, max_roc_26,max_roc_27, max_roc_28,max_roc_29,
                                            max_roc_30, max_roc_31, max_roc_32, max_roc_33, max_roc_34,max_roc_35, max_roc_36,max_roc_37, max_roc_38,max_roc_39,
                                            max_roc_40, max_roc_41, max_roc_42, max_roc_43, max_roc_44,max_roc_45, max_roc_46,max_roc_47, max_roc_48,max_roc_49,
                                            max_roc_50, max_roc_51, max_roc_52, max_roc_53, max_roc_54,max_roc_55, max_roc_56,max_roc_57, max_roc_58,max_roc_59,
                                            max_roc_60, max_roc_61, max_roc_62, max_roc_63, max_roc_64,max_roc_65, max_roc_66,max_roc_67, max_roc_68,max_roc_69,
                                            max_roc_70, max_roc_71, max_roc_72, max_roc_73, max_roc_74,max_roc_75, max_roc_76,max_roc_77, max_roc_78,max_roc_79,
                                            max_roc_80, max_roc_81, max_roc_82, max_roc_83, max_roc_84,max_roc_85, max_roc_86,max_roc_87, max_roc_88,max_roc_89,
                                            max_roc_90, max_roc_91, max_roc_92, max_roc_93, max_roc_94,max_roc_95, max_roc_96,max_roc_97, max_roc_98,max_roc_99,
                                            max_roc_100, max_roc_101, max_roc_102, max_roc_103, max_roc_104,max_roc_105,max_roc_106,max_roc_107,max_roc_108,max_roc_109,
                                            max_roc_110, max_roc_111, max_roc_112, max_roc_113, max_roc_114, max_roc_115, max_roc_116, max_roc_117, max_roc_118, max_roc_119,
                                            max_roc_120, max_roc_121, max_roc_122, max_roc_123, max_roc_124, max_roc_125, max_roc_126, max_roc_127, max_roc_128, max_roc_129,
                                            max_roc_130, max_roc_131, max_roc_132, max_roc_133, max_roc_134, max_roc_135, max_roc_136, max_roc_137, max_roc_138, max_roc_139,
                                            max_roc_140, max_roc_141, max_roc_142, max_roc_143, max_roc_144, max_roc_145, max_roc_146, max_roc_147, max_roc_148, max_roc_149,
                                            max_roc_150, max_roc_151, max_roc_152, max_roc_153, max_roc_154, max_roc_155, max_roc_156, max_roc_157, max_roc_158, max_roc_159,
                                            max_roc_160, max_roc_161, max_roc_162, max_roc_163, max_roc_164, max_roc_165, max_roc_166, max_roc_167, max_roc_168, max_roc_169,
                                            max_roc_170, max_roc_171, max_roc_172, max_roc_173, max_roc_174, max_roc_175, max_roc_176, max_roc_177, max_roc_178, max_roc_179,
                                            max_roc_180, max_roc_181, max_roc_182, max_roc_183, max_roc_184, max_roc_185, max_roc_186, max_roc_187, max_roc_188, max_roc_189,
                                            max_roc_190, max_roc_191, max_roc_192, max_roc_193, max_roc_194, max_roc_195, max_roc_196, max_roc_197, max_roc_198, max_roc_199,
                                            max_roc_200, max_roc_201],
                                          index = df_results.index)

#==============================================================================
#                                 AUC-PRC
#==============================================================================
#Get the best performance (e.g. 0.975) for each dataset
[max_prc_0, max_prc_1, max_prc_2, max_prc_3, max_prc_4, max_prc_5, max_prc_6, max_prc_7, max_prc_8, max_prc_9, 
max_prc_10, max_prc_11, max_prc_12, max_prc_13, max_prc_14, max_prc_15,max_prc_16, max_prc_17, max_prc_18, max_prc_19, 
max_prc_20, max_prc_21, max_prc_22, max_prc_23, max_prc_24, max_prc_25,max_prc_26, max_prc_27, max_prc_28, max_prc_29, 
max_prc_30, max_prc_31, max_prc_32, max_prc_33, max_prc_34, max_prc_35,max_prc_36, max_prc_37, max_prc_38, max_prc_39, 
max_prc_40, max_prc_41, max_prc_42, max_prc_43, max_prc_44, max_prc_45,max_prc_46, max_prc_47, max_prc_48, max_prc_49, 
max_prc_50, max_prc_51, max_prc_52, max_prc_53, max_prc_54, max_prc_55,max_prc_56, max_prc_57, max_prc_58, max_prc_59, 
max_prc_60, max_prc_61, max_prc_62, max_prc_63, max_prc_64, max_prc_65,max_prc_66, max_prc_67, max_prc_68, max_prc_69, 
max_prc_70, max_prc_71, max_prc_72, max_prc_73, max_prc_74, max_prc_75,max_prc_76, max_prc_77, max_prc_78, max_prc_79, 
max_prc_80, max_prc_81, max_prc_82, max_prc_83, max_prc_84, max_prc_85,max_prc_86, max_prc_87, max_prc_88, max_prc_89, 
max_prc_90, max_prc_91, max_prc_92, max_prc_93, max_prc_94, max_prc_95,max_prc_96, max_prc_97, max_prc_98, max_prc_99,
max_prc_100, max_prc_101, max_prc_102, max_prc_103, max_prc_104,max_prc_105, max_prc_106, max_prc_107, max_prc_108, max_prc_109,
max_prc_110, max_prc_111, max_prc_112, max_prc_113, max_prc_114, max_prc_115, max_prc_116, max_prc_117, max_prc_118, max_prc_119,
max_prc_120, max_prc_121, max_prc_122, max_prc_123, max_prc_124, max_prc_125, max_prc_126, max_prc_127, max_prc_128, max_prc_129,
max_prc_130, max_prc_131, max_prc_132, max_prc_133, max_prc_134, max_prc_135, max_prc_136, max_prc_137, max_prc_138, max_prc_139,
max_prc_140, max_prc_141, max_prc_142, max_prc_143, max_prc_144, max_prc_145, max_prc_146, max_prc_147, max_prc_148, max_prc_149,
max_prc_150, max_prc_151, max_prc_152, max_prc_153, max_prc_154, max_prc_155, max_prc_156, max_prc_157, max_prc_158, max_prc_159,
max_prc_160, max_prc_161, max_prc_162, max_prc_163, max_prc_164, max_prc_165, max_prc_166, max_prc_167, max_prc_168, max_prc_169,
max_prc_170, max_prc_171, max_prc_172, max_prc_173, max_prc_174, max_prc_175, max_prc_176, max_prc_177, max_prc_178, max_prc_179,
max_prc_180, max_prc_181, max_prc_182, max_prc_183, max_prc_184, max_prc_185, max_prc_186, max_prc_187, max_prc_188, max_prc_189,
max_prc_190, max_prc_191, max_prc_192, max_prc_193, max_prc_194, max_prc_195, max_prc_196, max_prc_197, max_prc_198, max_prc_199,
max_prc_200, max_prc_201]                                                                                = [max(results_prc_0.values()), max(results_prc_1.values()), max(results_prc_2.values()),max(results_prc_3.values()), max(results_prc_4.values()),max(results_prc_5.values()),max(results_prc_6.values()),max(results_prc_7.values()), max(results_prc_8.values()),max(results_prc_9.values()), 
                                                                                                                          max(results_prc_10.values()),max(results_prc_11.values()), max(results_prc_12.values()),max(results_prc_13.values()),max(results_prc_14.values()), max(results_prc_15.values()), max(results_prc_16.values()), max(results_prc_17.values()),max(results_prc_18.values()),max(results_prc_19.values()),
                                                                                                                          max(results_prc_20.values()),max(results_prc_21.values()), max(results_prc_22.values()),max(results_prc_23.values()),max(results_prc_24.values()), max(results_prc_25.values()), max(results_prc_26.values()), max(results_prc_27.values()),max(results_prc_28.values()),max(results_prc_29.values()),
                                                                                                                          max(results_prc_30.values()),max(results_prc_31.values()), max(results_prc_32.values()),max(results_prc_33.values()),max(results_prc_34.values()), max(results_prc_35.values()), max(results_prc_36.values()), max(results_prc_37.values()),max(results_prc_38.values()),max(results_prc_39.values()),
                                                                                                                          max(results_prc_40.values()),max(results_prc_41.values()), max(results_prc_42.values()),max(results_prc_43.values()),max(results_prc_44.values()), max(results_prc_45.values()), max(results_prc_46.values()), max(results_prc_47.values()),max(results_prc_48.values()),max(results_prc_49.values()),
                                                                                                                          max(results_prc_50.values()),max(results_prc_51.values()), max(results_prc_52.values()),max(results_prc_53.values()),max(results_prc_54.values()), max(results_prc_55.values()), max(results_prc_56.values()), max(results_prc_57.values()),max(results_prc_58.values()),max(results_prc_59.values()),
                                                                                                                          max(results_prc_60.values()),max(results_prc_61.values()), max(results_prc_62.values()),max(results_prc_63.values()),max(results_prc_64.values()), max(results_prc_65.values()), max(results_prc_66.values()), max(results_prc_67.values()),max(results_prc_68.values()),max(results_prc_69.values()),
                                                                                                                          max(results_prc_70.values()),max(results_prc_71.values()), max(results_prc_72.values()),max(results_prc_73.values()),max(results_prc_74.values()), max(results_prc_75.values()), max(results_prc_76.values()), max(results_prc_77.values()),max(results_prc_78.values()),max(results_prc_79.values()),
                                                                                                                          max(results_prc_80.values()),max(results_prc_81.values()), max(results_prc_82.values()),max(results_prc_83.values()),max(results_prc_84.values()), max(results_prc_85.values()), max(results_prc_86.values()), max(results_prc_87.values()),max(results_prc_88.values()),max(results_prc_89.values()),
                                                                                                                          max(results_prc_90.values()),max(results_prc_91.values()), max(results_prc_92.values()),max(results_prc_93.values()),max(results_prc_94.values()), max(results_prc_95.values()), max(results_prc_96.values()), max(results_prc_97.values()),max(results_prc_98.values()),max(results_prc_99.values()),
                                                                                                                          max(results_prc_100.values()), max(results_prc_101.values()), max(results_prc_102.values()), max(results_prc_103.values()), max(results_prc_104.values()),max(results_prc_105.values()),max(results_prc_106.values()),max(results_prc_107.values()),max(results_prc_108.values()),max(results_prc_109.values()),
                                                                                                                          max(results_prc_110.values()), max(results_prc_111.values()), max(results_prc_112.values()), max(results_prc_113.values()), max(results_prc_114.values()),max(results_prc_115.values()),max(results_prc_116.values()),max(results_prc_117.values()),max(results_prc_118.values()),max(results_prc_119.values()),
                                                                                                                          max(results_prc_120.values()), max(results_prc_121.values()), max(results_prc_122.values()), max(results_prc_123.values()), max(results_prc_124.values()),max(results_prc_125.values()),max(results_prc_126.values()),max(results_prc_127.values()),max(results_prc_128.values()),max(results_prc_129.values()),
                                                                                                                          max(results_prc_130.values()), max(results_prc_131.values()), max(results_prc_132.values()), max(results_prc_133.values()), max(results_prc_134.values()), max(results_prc_135.values()),max(results_prc_136.values()),max(results_prc_137.values()), max(results_prc_138.values()), max(results_prc_139.values()),
                                                                                                                          max(results_prc_140.values()), max(results_prc_141.values()), max(results_prc_142.values()), max(results_prc_143.values()), max(results_prc_144.values()), max(results_prc_145.values()), max(results_prc_146.values()), max(results_prc_147.values()), max(results_prc_148.values()), max(results_prc_149.values()),
                                                                                                                          max(results_prc_150.values()), max(results_prc_151.values()), max(results_prc_152.values()), max(results_prc_153.values()), max(results_prc_154.values()), max(results_prc_155.values()), max(results_prc_156.values()), max(results_prc_157.values()), max(results_prc_158.values()), max(results_prc_159.values()),
                                                                                                                          max(results_prc_160.values()), max(results_prc_161.values()), max(results_prc_162.values()), max(results_prc_163.values()), max(results_prc_164.values()), max(results_prc_165.values()), max(results_prc_166.values()), max(results_prc_167.values()), max(results_prc_168.values()), max(results_prc_169.values()),
                                                                                                                          max(results_prc_170.values()), max(results_prc_171.values()), max(results_prc_172.values()), max(results_prc_173.values()), max(results_prc_174.values()), max(results_prc_175.values()), max(results_prc_176.values()), max(results_prc_177.values()), max(results_prc_178.values()), max(results_prc_179.values()),
                                                                                                                          max(results_prc_180.values()), max(results_prc_181.values()), max(results_prc_182.values()), max(results_prc_183.values()), max(results_prc_184.values()), max(results_prc_185.values()), max(results_prc_186.values()), max(results_prc_187.values()), max(results_prc_188.values()), max(results_prc_189.values()),
                                                                                                                          max(results_prc_190.values()), max(results_prc_191.values()), max(results_prc_192.values()), max(results_prc_193.values()), max(results_prc_194.values()), max(results_prc_195.values()), max(results_prc_196.values()), max(results_prc_197.values()), max(results_prc_198.values()), max(results_prc_199.values()),
                                                                                                                          max(results_prc_200.values()), max(results_prc_201.values())]

#Get the best performing model (e.g. 'auc_prc_rforest') for each dataset
[maximum_prc_0, maximum_prc_1, maximum_prc_2, maximum_prc_3, maximum_prc_4, maximum_prc_5, maximum_prc_6, maximum_prc_7, maximum_prc_8, maximum_prc_9, 
maximum_prc_10, maximum_prc_11, maximum_prc_12, maximum_prc_13, maximum_prc_14, maximum_prc_15,maximum_prc_16, maximum_prc_17, maximum_prc_18, maximum_prc_19, 
maximum_prc_20, maximum_prc_21, maximum_prc_22, maximum_prc_23, maximum_prc_24, maximum_prc_25,maximum_prc_26, maximum_prc_27, maximum_prc_28, maximum_prc_29, 
maximum_prc_30, maximum_prc_31, maximum_prc_32, maximum_prc_33, maximum_prc_34, maximum_prc_35,maximum_prc_36, maximum_prc_37, maximum_prc_38, maximum_prc_39, 
maximum_prc_40, maximum_prc_41, maximum_prc_42, maximum_prc_43, maximum_prc_44, maximum_prc_45,maximum_prc_46, maximum_prc_47, maximum_prc_48, maximum_prc_49, 
maximum_prc_50, maximum_prc_51, maximum_prc_52, maximum_prc_53, maximum_prc_54, maximum_prc_55,maximum_prc_56, maximum_prc_57, maximum_prc_58, maximum_prc_59, 
maximum_prc_60, maximum_prc_61, maximum_prc_62, maximum_prc_63, maximum_prc_64, maximum_prc_65,maximum_prc_66, maximum_prc_67, maximum_prc_68, maximum_prc_69, 
maximum_prc_70, maximum_prc_71, maximum_prc_72, maximum_prc_73, maximum_prc_74, maximum_prc_75,maximum_prc_76, maximum_prc_77, maximum_prc_78, maximum_prc_79, 
maximum_prc_80, maximum_prc_81, maximum_prc_82, maximum_prc_83, maximum_prc_84, maximum_prc_85,maximum_prc_86, maximum_prc_87, maximum_prc_88, maximum_prc_89, 
maximum_prc_90, maximum_prc_91, maximum_prc_92, maximum_prc_93, maximum_prc_94, maximum_prc_95,maximum_prc_96, maximum_prc_97, maximum_prc_98, maximum_prc_99,
maximum_prc_100, maximum_prc_101, maximum_prc_102, maximum_prc_103, maximum_prc_104, maximum_prc_105, maximum_prc_106, maximum_prc_107, maximum_prc_108, maximum_prc_109,
maximum_prc_110, maximum_prc_111, maximum_prc_112, maximum_prc_113, maximum_prc_114, maximum_prc_115, maximum_prc_116, maximum_prc_117, maximum_prc_118, maximum_prc_119,
maximum_prc_120, maximum_prc_121, maximum_prc_122, maximum_prc_123, maximum_prc_124, maximum_prc_125, maximum_prc_126, maximum_prc_127, maximum_prc_128, maximum_prc_129,
maximum_prc_130, maximum_prc_131, maximum_prc_132, maximum_prc_133, maximum_prc_134, maximum_prc_135, maximum_prc_136, maximum_prc_137, maximum_prc_138, maximum_prc_139, 
maximum_prc_140, maximum_prc_141, maximum_prc_142, maximum_prc_143, maximum_prc_144, maximum_prc_145, maximum_prc_146, maximum_prc_147, maximum_prc_148, maximum_prc_149,
maximum_prc_150, maximum_prc_151, maximum_prc_152, maximum_prc_153, maximum_prc_154, maximum_prc_155, maximum_prc_156, maximum_prc_157, maximum_prc_158, maximum_prc_159,
maximum_prc_160, maximum_prc_161, maximum_prc_162, maximum_prc_163, maximum_prc_164, maximum_prc_165, maximum_prc_166, maximum_prc_167, maximum_prc_168, maximum_prc_169,
maximum_prc_170, maximum_prc_171, maximum_prc_172, maximum_prc_173, maximum_prc_174, maximum_prc_175, maximum_prc_176, maximum_prc_177, maximum_prc_178, maximum_prc_179,
maximum_prc_180, maximum_prc_181, maximum_prc_182, maximum_prc_183, maximum_prc_184, maximum_prc_185, maximum_prc_186, maximum_prc_187, maximum_prc_188, maximum_prc_189,
maximum_prc_190, maximum_prc_191, maximum_prc_192, maximum_prc_193, maximum_prc_194, maximum_prc_195, maximum_prc_196, maximum_prc_197, maximum_prc_198, maximum_prc_199,
maximum_prc_200, maximum_prc_201]                                                                                                                               = [max(results_prc_0, key=results_prc_0.get), max(results_prc_1, key=results_prc_1.get), max(results_prc_2, key=results_prc_2.get), max(results_prc_3, key=results_prc_3.get), max(results_prc_4, key=results_prc_4.get), max(results_prc_5, key=results_prc_5.get), max(results_prc_6, key=results_prc_6.get), max(results_prc_7, key=results_prc_7.get), max(results_prc_8, key=results_prc_8.get), max(results_prc_9, key=results_prc_9.get),
                                                                                                                                                                    max(results_prc_10, key=results_prc_10.get), max(results_prc_11, key=results_prc_11.get), max(results_prc_12, key=results_prc_12.get), max(results_prc_13, key=results_prc_13.get), max(results_prc_14, key=results_prc_14.get), max(results_prc_15, key=results_prc_15.get), max(results_prc_16, key=results_prc_16.get), max(results_prc_17, key=results_prc_17.get), max(results_prc_18, key=results_prc_18.get), max(results_prc_19, key=results_prc_19.get),
                                                                                                                                                                    max(results_prc_20, key=results_prc_20.get), max(results_prc_21, key=results_prc_21.get), max(results_prc_22, key=results_prc_22.get), max(results_prc_23, key=results_prc_23.get), max(results_prc_24, key=results_prc_24.get), max(results_prc_25, key=results_prc_25.get), max(results_prc_26, key=results_prc_26.get), max(results_prc_27, key=results_prc_27.get), max(results_prc_28, key=results_prc_28.get), max(results_prc_29, key=results_prc_29.get),
                                                                                                                                                                    max(results_prc_30, key=results_prc_30.get), max(results_prc_31, key=results_prc_31.get), max(results_prc_32, key=results_prc_32.get), max(results_prc_33, key=results_prc_33.get), max(results_prc_34, key=results_prc_34.get), max(results_prc_35, key=results_prc_35.get), max(results_prc_36, key=results_prc_36.get), max(results_prc_37, key=results_prc_37.get), max(results_prc_38, key=results_prc_38.get), max(results_prc_39, key=results_prc_39.get),
                                                                                                                                                                    max(results_prc_40, key=results_prc_40.get), max(results_prc_41, key=results_prc_41.get), max(results_prc_42, key=results_prc_42.get), max(results_prc_43, key=results_prc_43.get), max(results_prc_44, key=results_prc_44.get), max(results_prc_45, key=results_prc_45.get), max(results_prc_46, key=results_prc_46.get), max(results_prc_47, key=results_prc_47.get), max(results_prc_48, key=results_prc_48.get), max(results_prc_49, key=results_prc_49.get),
                                                                                                                                                                    max(results_prc_50, key=results_prc_50.get), max(results_prc_51, key=results_prc_51.get), max(results_prc_52, key=results_prc_52.get), max(results_prc_53, key=results_prc_53.get), max(results_prc_54, key=results_prc_54.get), max(results_prc_55, key=results_prc_55.get), max(results_prc_56, key=results_prc_56.get), max(results_prc_57, key=results_prc_57.get), max(results_prc_58, key=results_prc_58.get), max(results_prc_59, key=results_prc_59.get),
                                                                                                                                                                    max(results_prc_60, key=results_prc_60.get), max(results_prc_61, key=results_prc_61.get), max(results_prc_62, key=results_prc_62.get), max(results_prc_63, key=results_prc_63.get), max(results_prc_64, key=results_prc_64.get), max(results_prc_65, key=results_prc_65.get), max(results_prc_66, key=results_prc_66.get), max(results_prc_67, key=results_prc_67.get), max(results_prc_68, key=results_prc_68.get), max(results_prc_69, key=results_prc_69.get), 
                                                                                                                                                                    max(results_prc_70, key=results_prc_70.get), max(results_prc_71, key=results_prc_71.get), max(results_prc_72, key=results_prc_72.get), max(results_prc_73, key=results_prc_73.get), max(results_prc_74, key=results_prc_74.get), max(results_prc_75, key=results_prc_75.get), max(results_prc_76, key=results_prc_76.get), max(results_prc_77, key=results_prc_77.get), max(results_prc_78, key=results_prc_78.get), max(results_prc_79, key=results_prc_79.get),
                                                                                                                                                                    max(results_prc_80, key=results_prc_80.get), max(results_prc_81, key=results_prc_81.get), max(results_prc_82, key=results_prc_82.get), max(results_prc_83, key=results_prc_83.get), max(results_prc_84, key=results_prc_84.get), max(results_prc_85, key=results_prc_85.get), max(results_prc_86, key=results_prc_86.get), max(results_prc_87, key=results_prc_87.get), max(results_prc_88, key=results_prc_88.get), max(results_prc_89, key=results_prc_89.get),
                                                                                                                                                                    max(results_prc_90, key=results_prc_90.get), max(results_prc_91, key=results_prc_91.get), max(results_prc_92, key=results_prc_92.get), max(results_prc_93, key=results_prc_93.get), max(results_prc_94, key=results_prc_94.get), max(results_prc_95, key=results_prc_95.get), max(results_prc_96, key=results_prc_96.get), max(results_prc_97, key=results_prc_97.get), max(results_prc_98, key=results_prc_98.get), max(results_prc_99, key=results_prc_99.get),
                                                                                                                                                                    max(results_prc_100, key=results_prc_100.get), max(results_prc_101, key=results_prc_101.get), max(results_prc_102, key=results_prc_102.get), max(results_prc_103, key=results_prc_103.get), max(results_prc_104, key=results_prc_104.get), max(results_prc_105, key=results_prc_105.get), max(results_prc_106, key=results_prc_106.get), max(results_prc_107, key=results_prc_107.get), max(results_prc_108, key=results_prc_108.get), max(results_prc_109, key=results_prc_109.get),
                                                                                                                                                                    max(results_prc_110, key=results_prc_110.get), max(results_prc_111, key=results_prc_111.get), max(results_prc_112, key=results_prc_112.get), max(results_prc_113, key=results_prc_113.get), max(results_prc_114, key=results_prc_114.get), max(results_prc_115, key=results_prc_115.get), max(results_prc_116, key=results_prc_116.get), max(results_prc_117, key=results_prc_117.get), max(results_prc_118, key=results_prc_118.get), max(results_prc_119, key=results_prc_119.get),
                                                                                                                                                                    max(results_prc_120, key=results_prc_120.get), max(results_prc_121, key=results_prc_121.get), max(results_prc_122, key=results_prc_122.get), max(results_prc_123, key=results_prc_123.get), max(results_prc_124, key=results_prc_124.get), max(results_prc_125, key=results_prc_125.get), max(results_prc_126, key=results_prc_126.get), max(results_prc_127, key=results_prc_127.get), max(results_prc_128, key=results_prc_128.get), max(results_prc_129, key=results_prc_129.get),
                                                                                                                                                                    max(results_prc_130, key=results_prc_130.get), max(results_prc_131, key=results_prc_131.get), max(results_prc_132, key=results_prc_132.get), max(results_prc_133, key=results_prc_133.get), max(results_prc_134, key=results_prc_134.get), max(results_prc_135, key=results_prc_135.get), max(results_prc_136, key=results_prc_136.get), max(results_prc_137, key=results_prc_137.get), max(results_prc_138, key=results_prc_138.get), max(results_prc_139, key=results_prc_139.get),
                                                                                                                                                                    max(results_prc_140, key=results_prc_140.get), max(results_prc_141, key=results_prc_141.get), max(results_prc_142, key=results_prc_142.get), max(results_prc_143, key=results_prc_143.get), max(results_prc_144, key=results_prc_144.get), max(results_prc_145, key=results_prc_145.get), max(results_prc_146, key=results_prc_146.get), max(results_prc_147, key=results_prc_147.get), max(results_prc_148, key=results_prc_148.get), max(results_prc_149, key=results_prc_149.get),
                                                                                                                                                                    max(results_prc_150, key=results_prc_150.get), max(results_prc_151, key=results_prc_151.get), max(results_prc_152, key=results_prc_152.get), max(results_prc_153, key=results_prc_153.get), max(results_prc_154, key=results_prc_154.get), max(results_prc_155, key=results_prc_155.get), max(results_prc_156, key=results_prc_156.get), max(results_prc_157, key=results_prc_157.get), max(results_prc_158, key=results_prc_158.get), max(results_prc_159, key=results_prc_159.get),
                                                                                                                                                                    max(results_prc_160, key=results_prc_160.get), max(results_prc_161, key=results_prc_161.get), max(results_prc_162, key=results_prc_162.get), max(results_prc_163, key=results_prc_163.get), max(results_prc_164, key=results_prc_164.get), max(results_prc_165, key=results_prc_165.get), max(results_prc_166, key=results_prc_166.get), max(results_prc_167, key=results_prc_167.get), max(results_prc_168, key=results_prc_168.get), max(results_prc_169, key=results_prc_169.get),
                                                                                                                                                                    max(results_prc_170, key=results_prc_170.get), max(results_prc_171, key=results_prc_171.get), max(results_prc_172, key=results_prc_172.get), max(results_prc_173, key=results_prc_173.get), max(results_prc_174, key=results_prc_174.get), max(results_prc_175, key=results_prc_175.get), max(results_prc_176, key=results_prc_176.get), max(results_prc_177, key=results_prc_177.get), max(results_prc_178, key=results_prc_178.get), max(results_prc_179, key=results_prc_179.get),
                                                                                                                                                                    max(results_prc_180, key=results_prc_180.get), max(results_prc_181, key=results_prc_181.get), max(results_prc_182, key=results_prc_182.get), max(results_prc_183, key=results_prc_183.get), max(results_prc_184, key=results_prc_184.get), max(results_prc_185, key=results_prc_185.get), max(results_prc_186, key=results_prc_186.get), max(results_prc_187, key=results_prc_187.get), max(results_prc_188, key=results_prc_188.get), max(results_prc_189, key=results_prc_189.get),
                                                                                                                                                                    max(results_prc_190, key=results_prc_190.get), max(results_prc_191, key=results_prc_191.get), max(results_prc_192, key=results_prc_192.get), max(results_prc_193, key=results_prc_193.get), max(results_prc_194, key=results_prc_194.get), max(results_prc_195, key=results_prc_195.get), max(results_prc_196, key=results_prc_196.get), max(results_prc_197, key=results_prc_197.get), max(results_prc_198, key=results_prc_198.get), max(results_prc_199, key=results_prc_199.get),
                                                                                                                                                                    max(results_prc_200, key=results_prc_200.get), max(results_prc_201, key=results_prc_201.get)]
#Add the results to the dataframe                                                                                                                               
df_results['Best model PRC'] = pd.Series([maximum_prc_0, maximum_prc_1,maximum_prc_2,maximum_prc_3,maximum_prc_4,maximum_prc_5,maximum_prc_6,maximum_prc_7,maximum_prc_8, maximum_prc_9,
                                          maximum_prc_10,maximum_prc_11,maximum_prc_12, maximum_prc_13,maximum_prc_14,maximum_prc_15,maximum_prc_16, maximum_prc_17, maximum_prc_18,maximum_prc_19,
                                          maximum_prc_20,maximum_prc_21,maximum_prc_22, maximum_prc_23,maximum_prc_24,maximum_prc_25,maximum_prc_26, maximum_prc_27, maximum_prc_28,maximum_prc_29,
                                          maximum_prc_30,maximum_prc_31,maximum_prc_32, maximum_prc_33,maximum_prc_34,maximum_prc_35,maximum_prc_36, maximum_prc_37, maximum_prc_38,maximum_prc_39,
                                          maximum_prc_40,maximum_prc_41,maximum_prc_42, maximum_prc_43,maximum_prc_44,maximum_prc_45,maximum_prc_46, maximum_prc_47, maximum_prc_48,maximum_prc_49,
                                          maximum_prc_50,maximum_prc_51,maximum_prc_52, maximum_prc_53,maximum_prc_54,maximum_prc_55,maximum_prc_56, maximum_prc_57, maximum_prc_58,maximum_prc_59,
                                          maximum_prc_60,maximum_prc_61,maximum_prc_62, maximum_prc_63,maximum_prc_64,maximum_prc_65,maximum_prc_66, maximum_prc_67, maximum_prc_68,maximum_prc_69,
                                          maximum_prc_70,maximum_prc_71,maximum_prc_72, maximum_prc_73,maximum_prc_74,maximum_prc_75,maximum_prc_76, maximum_prc_77, maximum_prc_78,maximum_prc_79,
                                          maximum_prc_80,maximum_prc_81,maximum_prc_82, maximum_prc_83,maximum_prc_84,maximum_prc_85,maximum_prc_86, maximum_prc_87, maximum_prc_88,maximum_prc_89,
                                          maximum_prc_90,maximum_prc_91,maximum_prc_92, maximum_prc_93,maximum_prc_94,maximum_prc_95,maximum_prc_96, maximum_prc_97, maximum_prc_98,maximum_prc_99,
                                          maximum_prc_100, maximum_prc_101, maximum_prc_102, maximum_prc_103, maximum_prc_104,maximum_prc_105,maximum_prc_106,maximum_prc_107,maximum_prc_108,maximum_prc_109,
                                          maximum_prc_110, maximum_prc_111, maximum_prc_112, maximum_prc_113, maximum_prc_114,maximum_prc_115,maximum_prc_116,maximum_prc_117,maximum_prc_118,maximum_prc_119,
                                          maximum_prc_120, maximum_prc_121, maximum_prc_122, maximum_prc_123, maximum_prc_124,maximum_prc_125,maximum_prc_126,maximum_prc_127,maximum_prc_128,maximum_prc_129,
                                          maximum_prc_130, maximum_prc_131, maximum_prc_132, maximum_prc_133, maximum_prc_134, maximum_prc_135, maximum_prc_136, maximum_prc_137, maximum_prc_138, maximum_prc_139,
                                          maximum_prc_140, maximum_prc_141, maximum_prc_142, maximum_prc_143, maximum_prc_144, maximum_prc_145, maximum_prc_146, maximum_prc_147, maximum_prc_148, maximum_prc_149,
                                          maximum_prc_150, maximum_prc_151, maximum_prc_152, maximum_prc_153, maximum_prc_154, maximum_prc_155, maximum_prc_156, maximum_prc_157, maximum_prc_158, maximum_prc_159,
                                          maximum_prc_160, maximum_prc_161, maximum_prc_162, maximum_prc_163, maximum_prc_164, maximum_prc_165, maximum_prc_166, maximum_prc_167, maximum_prc_168, maximum_prc_169,
                                          maximum_prc_170, maximum_prc_171, maximum_prc_172, maximum_prc_173, maximum_prc_174, maximum_prc_175, maximum_prc_176, maximum_prc_177, maximum_prc_178, maximum_prc_179,
                                          maximum_prc_180, maximum_prc_181, maximum_prc_182, maximum_prc_183, maximum_prc_184, maximum_prc_185, maximum_prc_186, maximum_prc_187, maximum_prc_188, maximum_prc_189,
                                          maximum_prc_190, maximum_prc_191, maximum_prc_192, maximum_prc_193, maximum_prc_194, maximum_prc_195, maximum_prc_196, maximum_prc_197, maximum_prc_198, maximum_prc_199,
                                          maximum_prc_200, maximum_prc_201],
                                          index = df_results.index)

df_results['Maximum AUC-PRC'] = pd.Series([max_prc_0, max_prc_1,max_prc_2,max_prc_3,max_prc_4,max_prc_5, max_prc_6,max_prc_7,max_prc_8,max_prc_9,
                                            max_prc_10, max_prc_11, max_prc_12, max_prc_13, max_prc_14,max_prc_15, max_prc_16,max_prc_17, max_prc_18,max_prc_19,
                                            max_prc_20, max_prc_21, max_prc_22, max_prc_23, max_prc_24,max_prc_25, max_prc_26,max_prc_27, max_prc_28,max_prc_29,
                                            max_prc_30, max_prc_31, max_prc_32, max_prc_33, max_prc_34,max_prc_35, max_prc_36,max_prc_37, max_prc_38,max_prc_39,
                                            max_prc_40, max_prc_41, max_prc_42, max_prc_43, max_prc_44,max_prc_45, max_prc_46,max_prc_47, max_prc_48,max_prc_49,
                                            max_prc_50, max_prc_51, max_prc_52, max_prc_53, max_prc_54,max_prc_55, max_prc_56,max_prc_57, max_prc_58,max_prc_59,
                                            max_prc_60, max_prc_61, max_prc_62, max_prc_63, max_prc_64,max_prc_65, max_prc_66,max_prc_67, max_prc_68,max_prc_69,
                                            max_prc_70, max_prc_71, max_prc_72, max_prc_73, max_prc_74,max_prc_75, max_prc_76,max_prc_77, max_prc_78,max_prc_79,
                                            max_prc_80, max_prc_81, max_prc_82, max_prc_83, max_prc_84,max_prc_85, max_prc_86,max_prc_87, max_prc_88,max_prc_89,
                                            max_prc_90, max_prc_91, max_prc_92, max_prc_93, max_prc_94,max_prc_95, max_prc_96,max_prc_97, max_prc_98,max_prc_99,
                                            max_prc_100, max_prc_101, max_prc_102, max_prc_103, max_prc_104,max_prc_105,max_prc_106,max_prc_107,max_prc_108,max_prc_109,
                                            max_prc_110, max_prc_111, max_prc_112, max_prc_113, max_prc_114, max_prc_115, max_prc_116, max_prc_117, max_prc_118, max_prc_119,
                                            max_prc_120, max_prc_121, max_prc_122, max_prc_123, max_prc_124, max_prc_125, max_prc_126, max_prc_127, max_prc_128, max_prc_129,
                                            max_prc_130, max_prc_131, max_prc_132, max_prc_133, max_prc_134, max_prc_135, max_prc_136, max_prc_137, max_prc_138, max_prc_139, 
                                            max_prc_140, max_prc_141, max_prc_142, max_prc_143, max_prc_144, max_prc_145, max_prc_146, max_prc_147, max_prc_148, max_prc_149,
                                            max_prc_150, max_prc_151, max_prc_152, max_prc_153, max_prc_154, max_prc_155, max_prc_156, max_prc_157, max_prc_158, max_prc_159,
                                            max_prc_160, max_prc_161, max_prc_162, max_prc_163, max_prc_164, max_prc_165, max_prc_166, max_prc_167, max_prc_168, max_prc_169,
                                            max_prc_170, max_prc_171, max_prc_172, max_prc_173, max_prc_174, max_prc_175, max_prc_176, max_prc_177, max_prc_178, max_prc_179,
                                            max_prc_180, max_prc_181, max_prc_182, max_prc_183, max_prc_184, max_prc_185, max_prc_186, max_prc_187, max_prc_188, max_prc_189,
                                            max_prc_190, max_prc_191, max_prc_192, max_prc_193, max_prc_194, max_prc_195, max_prc_196, max_prc_197, max_prc_198, max_prc_199,
                                            max_prc_200, max_prc_201],
                                          index = df_results.index)                                                                                                                          
                                                   
#==============================================================================
#                                 F1 metric
#==============================================================================
#Get the best performance (e.g. 0.975) for each dataset
[max_f1_0, max_f1_1, max_f1_2, max_f1_3, max_f1_4, max_f1_5, max_f1_6, max_f1_7, max_f1_8, max_f1_9, 
max_f1_10, max_f1_11, max_f1_12, max_f1_13, max_f1_14, max_f1_15,max_f1_16, max_f1_17, max_f1_18, max_f1_19, 
max_f1_20, max_f1_21, max_f1_22, max_f1_23, max_f1_24, max_f1_25,max_f1_26, max_f1_27, max_f1_28, max_f1_29, 
max_f1_30, max_f1_31, max_f1_32, max_f1_33, max_f1_34, max_f1_35,max_f1_36, max_f1_37, max_f1_38, max_f1_39, 
max_f1_40, max_f1_41, max_f1_42, max_f1_43, max_f1_44, max_f1_45,max_f1_46, max_f1_47, max_f1_48, max_f1_49, 
max_f1_50, max_f1_51, max_f1_52, max_f1_53, max_f1_54, max_f1_55,max_f1_56, max_f1_57, max_f1_58, max_f1_59, 
max_f1_60, max_f1_61, max_f1_62, max_f1_63, max_f1_64, max_f1_65,max_f1_66, max_f1_67, max_f1_68, max_f1_69, 
max_f1_70, max_f1_71, max_f1_72, max_f1_73, max_f1_74, max_f1_75,max_f1_76, max_f1_77, max_f1_78, max_f1_79, 
max_f1_80, max_f1_81, max_f1_82, max_f1_83, max_f1_84, max_f1_85,max_f1_86, max_f1_87, max_f1_88, max_f1_89, 
max_f1_90, max_f1_91, max_f1_92, max_f1_93, max_f1_94, max_f1_95,max_f1_96, max_f1_97, max_f1_98, max_f1_99,
max_f1_100, max_f1_101, max_f1_102, max_f1_103, max_f1_104, max_f1_105, max_f1_106, max_f1_107, max_f1_108, max_f1_109,
max_f1_110, max_f1_111, max_f1_112, max_f1_113, max_f1_114, max_f1_115, max_f1_116, max_f1_117, max_f1_118, max_f1_119,
max_f1_120, max_f1_121, max_f1_122, max_f1_123, max_f1_124, max_f1_125, max_f1_126, max_f1_127, max_f1_128, max_f1_129,
max_f1_130, max_f1_131, max_f1_132, max_f1_133, max_f1_134, max_f1_135, max_f1_136, max_f1_137, max_f1_138, max_f1_139, 
max_f1_140, max_f1_141, max_f1_142, max_f1_143, max_f1_144, max_f1_145, max_f1_146, max_f1_147, max_f1_148, max_f1_149,
max_f1_150, max_f1_151, max_f1_152, max_f1_153, max_f1_154, max_f1_155, max_f1_156, max_f1_157, max_f1_158, max_f1_159,
max_f1_160, max_f1_161, max_f1_162, max_f1_163, max_f1_164, max_f1_165, max_f1_166, max_f1_167, max_f1_168, max_f1_169,
max_f1_170, max_f1_171, max_f1_172, max_f1_173, max_f1_174, max_f1_175, max_f1_176, max_f1_177, max_f1_178, max_f1_179,
max_f1_180, max_f1_181, max_f1_182, max_f1_183, max_f1_184, max_f1_185, max_f1_186, max_f1_187, max_f1_188, max_f1_189,
max_f1_190, max_f1_191, max_f1_192, max_f1_193, max_f1_194, max_f1_195, max_f1_196, max_f1_197, max_f1_198, max_f1_199,
max_f1_200, max_f1_201]                                                                                        = [max(results_f1_0.values()), max(results_f1_1.values()), max(results_f1_2.values()),max(results_f1_3.values()), max(results_f1_4.values()),max(results_f1_5.values()),max(results_f1_6.values()),max(results_f1_7.values()), max(results_f1_8.values()),max(results_f1_9.values()), 
                                                                                                                          max(results_f1_10.values()), max(results_f1_11.values()), max(results_f1_12.values()),max(results_f1_13.values()),max(results_f1_14.values()), max(results_f1_15.values()), max(results_f1_16.values()), max(results_f1_17.values()),max(results_f1_18.values()),max(results_f1_19.values()),
                                                                                                                          max(results_f1_20.values()), max(results_f1_21.values()), max(results_f1_22.values()),max(results_f1_23.values()),max(results_f1_24.values()), max(results_f1_25.values()), max(results_f1_26.values()), max(results_f1_27.values()),max(results_f1_28.values()),max(results_f1_29.values()),
                                                                                                                          max(results_f1_30.values()), max(results_f1_31.values()), max(results_f1_32.values()),max(results_f1_33.values()),max(results_f1_34.values()), max(results_f1_35.values()), max(results_f1_36.values()), max(results_f1_37.values()),max(results_f1_38.values()),max(results_f1_39.values()),
                                                                                                                          max(results_f1_40.values()), max(results_f1_41.values()), max(results_f1_42.values()),max(results_f1_43.values()),max(results_f1_44.values()), max(results_f1_45.values()), max(results_f1_46.values()), max(results_f1_47.values()),max(results_f1_48.values()),max(results_f1_49.values()),
                                                                                                                          max(results_f1_50.values()), max(results_f1_51.values()), max(results_f1_52.values()),max(results_f1_53.values()),max(results_f1_54.values()), max(results_f1_55.values()), max(results_f1_56.values()), max(results_f1_57.values()),max(results_f1_58.values()),max(results_f1_59.values()),
                                                                                                                          max(results_f1_60.values()), max(results_f1_61.values()), max(results_f1_62.values()),max(results_f1_63.values()),max(results_f1_64.values()), max(results_f1_65.values()), max(results_f1_66.values()), max(results_f1_67.values()),max(results_f1_68.values()),max(results_f1_69.values()),
                                                                                                                          max(results_f1_70.values()), max(results_f1_71.values()), max(results_f1_72.values()),max(results_f1_73.values()),max(results_f1_74.values()), max(results_f1_75.values()), max(results_f1_76.values()), max(results_f1_77.values()),max(results_f1_78.values()),max(results_f1_79.values()),
                                                                                                                          max(results_f1_80.values()), max(results_f1_81.values()), max(results_f1_82.values()),max(results_f1_83.values()),max(results_f1_84.values()), max(results_f1_85.values()), max(results_f1_86.values()), max(results_f1_87.values()),max(results_f1_88.values()),max(results_f1_89.values()),
                                                                                                                          max(results_f1_90.values()), max(results_f1_91.values()), max(results_f1_92.values()),max(results_f1_93.values()),max(results_f1_94.values()), max(results_f1_95.values()), max(results_f1_96.values()), max(results_f1_97.values()),max(results_f1_98.values()),max(results_f1_99.values()),
                                                                                                                          max(results_f1_100.values()), max(results_f1_101.values()), max(results_f1_102.values()), max(results_f1_103.values()), max(results_f1_104.values()), max(results_f1_105.values()), max(results_f1_106.values()), max(results_f1_107.values()), max(results_f1_108.values()), max(results_f1_109.values()),
                                                                                                                          max(results_f1_110.values()), max(results_f1_111.values()), max(results_f1_112.values()), max(results_f1_113.values()), max(results_f1_114.values()), max(results_f1_115.values()), max(results_f1_116.values()), max(results_f1_117.values()), max(results_f1_118.values()), max(results_f1_119.values()),
                                                                                                                          max(results_f1_120.values()), max(results_f1_121.values()), max(results_f1_122.values()), max(results_f1_123.values()), max(results_f1_124.values()), max(results_f1_125.values()), max(results_f1_126.values()), max(results_f1_127.values()), max(results_f1_128.values()), max(results_f1_129.values()),
                                                                                                                          max(results_f1_130.values()), max(results_f1_131.values()), max(results_f1_132.values()), max(results_f1_133.values()), max(results_f1_134.values()), max(results_f1_135.values()), max(results_f1_136.values()), max(results_f1_137.values()), max(results_f1_138.values()), max(results_f1_139.values()),
                                                                                                                          max(results_f1_140.values()), max(results_f1_141.values()), max(results_f1_142.values()), max(results_f1_143.values()), max(results_f1_144.values()), max(results_f1_145.values()), max(results_f1_146.values()), max(results_f1_147.values()), max(results_f1_148.values()), max(results_f1_149.values()),
                                                                                                                          max(results_f1_150.values()), max(results_f1_151.values()), max(results_f1_152.values()), max(results_f1_153.values()), max(results_f1_154.values()), max(results_f1_155.values()), max(results_f1_156.values()), max(results_f1_157.values()), max(results_f1_158.values()), max(results_f1_159.values()),
                                                                                                                          max(results_f1_160.values()), max(results_f1_161.values()), max(results_f1_162.values()), max(results_f1_163.values()), max(results_f1_164.values()), max(results_f1_165.values()), max(results_f1_166.values()), max(results_f1_167.values()), max(results_f1_168.values()), max(results_f1_169.values()),
                                                                                                                          max(results_f1_170.values()), max(results_f1_171.values()), max(results_f1_172.values()), max(results_f1_173.values()), max(results_f1_174.values()), max(results_f1_175.values()), max(results_f1_176.values()), max(results_f1_177.values()), max(results_f1_178.values()), max(results_f1_179.values()),
                                                                                                                          max(results_f1_180.values()), max(results_f1_181.values()), max(results_f1_182.values()), max(results_f1_183.values()), max(results_f1_184.values()), max(results_f1_185.values()), max(results_f1_186.values()), max(results_f1_187.values()), max(results_f1_188.values()), max(results_f1_189.values()),
                                                                                                                          max(results_f1_190.values()), max(results_f1_191.values()), max(results_f1_192.values()), max(results_f1_193.values()), max(results_f1_194.values()), max(results_f1_195.values()), max(results_f1_196.values()), max(results_f1_197.values()), max(results_f1_198.values()), max(results_f1_199.values()),
                                                                                                                          max(results_f1_200.values()), max(results_f1_201.values())]

#Get the best performing model (e.g. 'auc_f1_rforest') for each dataset
[maximum_f1_0, maximum_f1_1, maximum_f1_2, maximum_f1_3, maximum_f1_4, maximum_f1_5, maximum_f1_6, maximum_f1_7, maximum_f1_8, maximum_f1_9, 
maximum_f1_10, maximum_f1_11, maximum_f1_12, maximum_f1_13, maximum_f1_14, maximum_f1_15,maximum_f1_16, maximum_f1_17, maximum_f1_18, maximum_f1_19, 
maximum_f1_20, maximum_f1_21, maximum_f1_22, maximum_f1_23, maximum_f1_24, maximum_f1_25,maximum_f1_26, maximum_f1_27, maximum_f1_28, maximum_f1_29, 
maximum_f1_30, maximum_f1_31, maximum_f1_32, maximum_f1_33, maximum_f1_34, maximum_f1_35,maximum_f1_36, maximum_f1_37, maximum_f1_38, maximum_f1_39, 
maximum_f1_40, maximum_f1_41, maximum_f1_42, maximum_f1_43, maximum_f1_44, maximum_f1_45,maximum_f1_46, maximum_f1_47, maximum_f1_48, maximum_f1_49, 
maximum_f1_50, maximum_f1_51, maximum_f1_52, maximum_f1_53, maximum_f1_54, maximum_f1_55,maximum_f1_56, maximum_f1_57, maximum_f1_58, maximum_f1_59, 
maximum_f1_60, maximum_f1_61, maximum_f1_62, maximum_f1_63, maximum_f1_64, maximum_f1_65,maximum_f1_66, maximum_f1_67, maximum_f1_68, maximum_f1_69, 
maximum_f1_70, maximum_f1_71, maximum_f1_72, maximum_f1_73, maximum_f1_74, maximum_f1_75,maximum_f1_76, maximum_f1_77, maximum_f1_78, maximum_f1_79, 
maximum_f1_80, maximum_f1_81, maximum_f1_82, maximum_f1_83, maximum_f1_84, maximum_f1_85,maximum_f1_86, maximum_f1_87, maximum_f1_88, maximum_f1_89, 
maximum_f1_90, maximum_f1_91, maximum_f1_92, maximum_f1_93, maximum_f1_94, maximum_f1_95,maximum_f1_96, maximum_f1_97, maximum_f1_98, maximum_f1_99,
maximum_f1_100, maximum_f1_101, maximum_f1_102, maximum_f1_103, maximum_f1_104,maximum_f1_105,maximum_f1_106,maximum_f1_107,maximum_f1_108,maximum_f1_109,
maximum_f1_110, maximum_f1_111, maximum_f1_112, maximum_f1_113, maximum_f1_114, maximum_f1_115, maximum_f1_116, maximum_f1_117, maximum_f1_118, maximum_f1_119,
maximum_f1_120, maximum_f1_121, maximum_f1_122, maximum_f1_123, maximum_f1_124, maximum_f1_125, maximum_f1_126, maximum_f1_127, maximum_f1_128, maximum_f1_129,
maximum_f1_130, maximum_f1_131, maximum_f1_132, maximum_f1_133, maximum_f1_134, maximum_f1_135, maximum_f1_136, maximum_f1_137, maximum_f1_138, maximum_f1_139, 
maximum_f1_140, maximum_f1_141, maximum_f1_142, maximum_f1_143, maximum_f1_144, maximum_f1_145, maximum_f1_146, maximum_f1_147, maximum_f1_148, maximum_f1_149,
maximum_f1_150, maximum_f1_151, maximum_f1_152, maximum_f1_153, maximum_f1_154, maximum_f1_155, maximum_f1_156, maximum_f1_157, maximum_f1_158, maximum_f1_159,
maximum_f1_160, maximum_f1_161, maximum_f1_162, maximum_f1_163, maximum_f1_164, maximum_f1_165, maximum_f1_166, maximum_f1_167, maximum_f1_168, maximum_f1_169,
maximum_f1_170, maximum_f1_171, maximum_f1_172, maximum_f1_173, maximum_f1_174, maximum_f1_175, maximum_f1_176, maximum_f1_177, maximum_f1_178, maximum_f1_179,
maximum_f1_180, maximum_f1_181, maximum_f1_182, maximum_f1_183, maximum_f1_184, maximum_f1_185, maximum_f1_186, maximum_f1_187, maximum_f1_188, maximum_f1_189,
maximum_f1_190, maximum_f1_191, maximum_f1_192, maximum_f1_193, maximum_f1_194, maximum_f1_195, maximum_f1_196, maximum_f1_197, maximum_f1_198, maximum_f1_199,
maximum_f1_200, maximum_f1_201]                                                                                                                    = [max(results_f1_0, key=results_f1_0.get), max(results_f1_1, key=results_f1_1.get), max(results_f1_2, key=results_f1_2.get), max(results_f1_3, key=results_f1_3.get), max(results_f1_4, key=results_f1_4.get), max(results_f1_5, key=results_f1_5.get), max(results_f1_6, key=results_f1_6.get), max(results_f1_7, key=results_f1_7.get), max(results_f1_8, key=results_f1_8.get), max(results_f1_9, key=results_f1_9.get),
                                                                                                                                                                    max(results_f1_10, key=results_f1_10.get), max(results_f1_11, key=results_f1_11.get), max(results_f1_12, key=results_f1_12.get), max(results_f1_13, key=results_f1_13.get), max(results_f1_14, key=results_f1_14.get), max(results_f1_15, key=results_f1_15.get), max(results_f1_16, key=results_f1_16.get), max(results_f1_17, key=results_f1_17.get), max(results_f1_18, key=results_f1_18.get), max(results_f1_19, key=results_f1_19.get),
                                                                                                                                                                    max(results_f1_20, key=results_f1_20.get), max(results_f1_21, key=results_f1_21.get), max(results_f1_22, key=results_f1_22.get), max(results_f1_23, key=results_f1_23.get), max(results_f1_24, key=results_f1_24.get), max(results_f1_25, key=results_f1_25.get), max(results_f1_26, key=results_f1_26.get), max(results_f1_27, key=results_f1_27.get), max(results_f1_28, key=results_f1_28.get), max(results_f1_29, key=results_f1_29.get),
                                                                                                                                                                    max(results_f1_30, key=results_f1_30.get), max(results_f1_31, key=results_f1_31.get), max(results_f1_32, key=results_f1_32.get), max(results_f1_33, key=results_f1_33.get), max(results_f1_34, key=results_f1_34.get), max(results_f1_35, key=results_f1_35.get), max(results_f1_36, key=results_f1_36.get), max(results_f1_37, key=results_f1_37.get), max(results_f1_38, key=results_f1_38.get), max(results_f1_39, key=results_f1_39.get),
                                                                                                                                                                    max(results_f1_40, key=results_f1_40.get), max(results_f1_41, key=results_f1_41.get), max(results_f1_42, key=results_f1_42.get), max(results_f1_43, key=results_f1_43.get), max(results_f1_44, key=results_f1_44.get), max(results_f1_45, key=results_f1_45.get), max(results_f1_46, key=results_f1_46.get), max(results_f1_47, key=results_f1_47.get), max(results_f1_48, key=results_f1_48.get), max(results_f1_49, key=results_f1_49.get),
                                                                                                                                                                    max(results_f1_50, key=results_f1_50.get), max(results_f1_51, key=results_f1_51.get), max(results_f1_52, key=results_f1_52.get), max(results_f1_53, key=results_f1_53.get), max(results_f1_54, key=results_f1_54.get), max(results_f1_55, key=results_f1_55.get), max(results_f1_56, key=results_f1_56.get), max(results_f1_57, key=results_f1_57.get), max(results_f1_58, key=results_f1_58.get), max(results_f1_59, key=results_f1_59.get),
                                                                                                                                                                    max(results_f1_60, key=results_f1_60.get), max(results_f1_61, key=results_f1_61.get), max(results_f1_62, key=results_f1_62.get), max(results_f1_63, key=results_f1_63.get), max(results_f1_64, key=results_f1_64.get), max(results_f1_65, key=results_f1_65.get), max(results_f1_66, key=results_f1_66.get), max(results_f1_67, key=results_f1_67.get), max(results_f1_68, key=results_f1_68.get), max(results_f1_69, key=results_f1_69.get), 
                                                                                                                                                                    max(results_f1_70, key=results_f1_70.get), max(results_f1_71, key=results_f1_71.get), max(results_f1_72, key=results_f1_72.get), max(results_f1_73, key=results_f1_73.get), max(results_f1_74, key=results_f1_74.get), max(results_f1_75, key=results_f1_75.get), max(results_f1_76, key=results_f1_76.get), max(results_f1_77, key=results_f1_77.get), max(results_f1_78, key=results_f1_78.get), max(results_f1_79, key=results_f1_79.get),
                                                                                                                                                                    max(results_f1_80, key=results_f1_80.get), max(results_f1_81, key=results_f1_81.get), max(results_f1_82, key=results_f1_82.get), max(results_f1_83, key=results_f1_83.get), max(results_f1_84, key=results_f1_84.get), max(results_f1_85, key=results_f1_85.get), max(results_f1_86, key=results_f1_86.get), max(results_f1_87, key=results_f1_87.get), max(results_f1_88, key=results_f1_88.get), max(results_f1_89, key=results_f1_89.get),
                                                                                                                                                                    max(results_f1_90, key=results_f1_90.get), max(results_f1_91, key=results_f1_91.get), max(results_f1_92, key=results_f1_92.get), max(results_f1_93, key=results_f1_93.get), max(results_f1_94, key=results_f1_94.get), max(results_f1_95, key=results_f1_95.get), max(results_f1_96, key=results_f1_96.get), max(results_f1_97, key=results_f1_97.get), max(results_f1_98, key=results_f1_98.get), max(results_f1_99, key=results_f1_99.get),
                                                                                                                                                                    max(results_f1_100, key=results_f1_100.get), max(results_f1_101, key=results_f1_101.get), max(results_f1_102, key=results_f1_102.get), max(results_f1_103, key=results_f1_103.get), max(results_f1_104, key=results_f1_104.get), max(results_f1_105, key=results_f1_105.get), max(results_f1_106, key=results_f1_106.get), max(results_f1_107, key=results_f1_107.get), max(results_f1_108, key=results_f1_108.get), max(results_f1_109, key=results_f1_109.get),
                                                                                                                                                                    max(results_f1_110, key=results_f1_110.get), max(results_f1_111, key=results_f1_111.get), max(results_f1_112, key=results_f1_112.get), max(results_f1_113, key=results_f1_113.get), max(results_f1_114, key=results_f1_114.get), max(results_f1_115, key=results_f1_115.get), max(results_f1_116, key=results_f1_116.get), max(results_f1_117, key=results_f1_117.get), max(results_f1_118, key=results_f1_118.get), max(results_f1_119, key=results_f1_119.get),
                                                                                                                                                                    max(results_f1_120, key=results_f1_120.get), max(results_f1_121, key=results_f1_121.get), max(results_f1_122, key=results_f1_122.get), max(results_f1_123, key=results_f1_123.get), max(results_f1_124, key=results_f1_124.get), max(results_f1_125, key=results_f1_125.get), max(results_f1_126, key=results_f1_126.get), max(results_f1_127, key=results_f1_127.get), max(results_f1_128, key=results_f1_128.get), max(results_f1_129, key=results_f1_129.get),
                                                                                                                                                                    max(results_f1_130, key=results_f1_130.get), max(results_f1_131, key=results_f1_131.get), max(results_f1_132, key=results_f1_132.get), max(results_f1_133, key=results_f1_133.get), max(results_f1_134, key=results_f1_134.get), max(results_f1_135, key=results_f1_135.get), max(results_f1_136, key=results_f1_136.get), max(results_f1_137, key=results_f1_137.get), max(results_f1_138, key=results_f1_138.get), max(results_f1_139, key=results_f1_139.get),
                                                                                                                                                                    max(results_f1_140, key=results_f1_140.get), max(results_f1_141, key=results_f1_141.get), max(results_f1_142, key=results_f1_142.get), max(results_f1_143, key=results_f1_143.get), max(results_f1_144, key=results_f1_144.get), max(results_f1_145, key=results_f1_145.get), max(results_f1_146, key=results_f1_146.get), max(results_f1_147, key=results_f1_147.get), max(results_f1_148, key=results_f1_148.get), max(results_f1_149, key=results_f1_149.get),
                                                                                                                                                                    max(results_f1_150, key=results_f1_150.get), max(results_f1_151, key=results_f1_151.get), max(results_f1_152, key=results_f1_152.get), max(results_f1_153, key=results_f1_153.get), max(results_f1_154, key=results_f1_154.get), max(results_f1_155, key=results_f1_155.get), max(results_f1_156, key=results_f1_156.get), max(results_f1_157, key=results_f1_157.get), max(results_f1_158, key=results_f1_158.get), max(results_f1_159, key=results_f1_159.get),
                                                                                                                                                                    max(results_f1_160, key=results_f1_160.get), max(results_f1_161, key=results_f1_161.get), max(results_f1_162, key=results_f1_162.get), max(results_f1_163, key=results_f1_163.get), max(results_f1_164, key=results_f1_164.get), max(results_f1_165, key=results_f1_165.get), max(results_f1_166, key=results_f1_166.get), max(results_f1_167, key=results_f1_167.get), max(results_f1_168, key=results_f1_168.get), max(results_f1_169, key=results_f1_169.get),
                                                                                                                                                                    max(results_f1_170, key=results_f1_170.get), max(results_f1_171, key=results_f1_171.get), max(results_f1_172, key=results_f1_172.get), max(results_f1_173, key=results_f1_173.get), max(results_f1_174, key=results_f1_174.get), max(results_f1_175, key=results_f1_175.get), max(results_f1_176, key=results_f1_176.get), max(results_f1_177, key=results_f1_177.get), max(results_f1_178, key=results_f1_178.get), max(results_f1_179, key=results_f1_179.get),
                                                                                                                                                                    max(results_f1_180, key=results_f1_180.get), max(results_f1_181, key=results_f1_181.get), max(results_f1_182, key=results_f1_182.get), max(results_f1_183, key=results_f1_183.get), max(results_f1_184, key=results_f1_184.get), max(results_f1_185, key=results_f1_185.get), max(results_f1_186, key=results_f1_186.get), max(results_f1_187, key=results_f1_187.get), max(results_f1_188, key=results_f1_188.get), max(results_f1_189, key=results_f1_189.get),
                                                                                                                                                                    max(results_f1_190, key=results_f1_190.get), max(results_f1_191, key=results_f1_191.get), max(results_f1_192, key=results_f1_192.get), max(results_f1_193, key=results_f1_193.get), max(results_f1_194, key=results_f1_194.get), max(results_f1_195, key=results_f1_195.get), max(results_f1_196, key=results_f1_196.get), max(results_f1_197, key=results_f1_197.get), max(results_f1_198, key=results_f1_198.get), max(results_f1_199, key=results_f1_199.get),
                                                                                                                                                                    max(results_f1_200, key=results_f1_200.get), max(results_f1_201, key=results_f1_201.get)]
#Add the results to the dataframe                                                                                                                               
df_results['Best model f1'] = pd.Series([maximum_f1_0, maximum_f1_1,maximum_f1_2,maximum_f1_3,maximum_f1_4,maximum_f1_5,maximum_f1_6,maximum_f1_7,maximum_f1_8, maximum_f1_9,
                                          maximum_f1_10,maximum_f1_11,maximum_f1_12, maximum_f1_13,maximum_f1_14,maximum_f1_15,maximum_f1_16, maximum_f1_17, maximum_f1_18,maximum_f1_19,
                                          maximum_f1_20,maximum_f1_21,maximum_f1_22, maximum_f1_23,maximum_f1_24,maximum_f1_25,maximum_f1_26, maximum_f1_27, maximum_f1_28,maximum_f1_29,
                                          maximum_f1_30,maximum_f1_31,maximum_f1_32, maximum_f1_33,maximum_f1_34,maximum_f1_35,maximum_f1_36, maximum_f1_37, maximum_f1_38,maximum_f1_39,
                                          maximum_f1_40,maximum_f1_41,maximum_f1_42, maximum_f1_43,maximum_f1_44,maximum_f1_45,maximum_f1_46, maximum_f1_47, maximum_f1_48,maximum_f1_49,
                                          maximum_f1_50,maximum_f1_51,maximum_f1_52, maximum_f1_53,maximum_f1_54,maximum_f1_55,maximum_f1_56, maximum_f1_57, maximum_f1_58,maximum_f1_59,
                                          maximum_f1_60,maximum_f1_61,maximum_f1_62, maximum_f1_63,maximum_f1_64,maximum_f1_65,maximum_f1_66, maximum_f1_67, maximum_f1_68,maximum_f1_69,
                                          maximum_f1_70,maximum_f1_71,maximum_f1_72, maximum_f1_73,maximum_f1_74,maximum_f1_75,maximum_f1_76, maximum_f1_77, maximum_f1_78,maximum_f1_79,
                                          maximum_f1_80,maximum_f1_81,maximum_f1_82, maximum_f1_83,maximum_f1_84,maximum_f1_85,maximum_f1_86, maximum_f1_87, maximum_f1_88,maximum_f1_89,
                                          maximum_f1_90,maximum_f1_91,maximum_f1_92, maximum_f1_93,maximum_f1_94,maximum_f1_95,maximum_f1_96, maximum_f1_97, maximum_f1_98,maximum_f1_99,
                                          maximum_f1_100, maximum_f1_101, maximum_f1_102, maximum_f1_103, maximum_f1_104,maximum_f1_105,maximum_f1_106,maximum_f1_107,maximum_f1_108,maximum_f1_109,
                                          maximum_f1_110, maximum_f1_111, maximum_f1_112, maximum_f1_113, maximum_f1_114, maximum_f1_115, maximum_f1_116, maximum_f1_117, maximum_f1_118, maximum_f1_119,
                                          maximum_f1_120, maximum_f1_121, maximum_f1_122, maximum_f1_123, maximum_f1_124, maximum_f1_125, maximum_f1_126, maximum_f1_127, maximum_f1_128, maximum_f1_129,
                                          maximum_f1_130, maximum_f1_131, maximum_f1_132, maximum_f1_133, maximum_f1_134, maximum_f1_135, maximum_f1_136, maximum_f1_137, maximum_f1_138, maximum_f1_139, 
                                          maximum_f1_140, maximum_f1_141, maximum_f1_142, maximum_f1_143, maximum_f1_144, maximum_f1_145, maximum_f1_146, maximum_f1_147, maximum_f1_148, maximum_f1_149,
                                          maximum_f1_150, maximum_f1_151, maximum_f1_152, maximum_f1_153, maximum_f1_154, maximum_f1_155, maximum_f1_156, maximum_f1_157, maximum_f1_158, maximum_f1_159,
                                          maximum_f1_160, maximum_f1_161, maximum_f1_162, maximum_f1_163, maximum_f1_164, maximum_f1_165, maximum_f1_166, maximum_f1_167, maximum_f1_168, maximum_f1_169,
                                          maximum_f1_170, maximum_f1_171, maximum_f1_172, maximum_f1_173, maximum_f1_174, maximum_f1_175, maximum_f1_176, maximum_f1_177, maximum_f1_178, maximum_f1_179,
                                          maximum_f1_180, maximum_f1_181, maximum_f1_182, maximum_f1_183, maximum_f1_184, maximum_f1_185, maximum_f1_186, maximum_f1_187, maximum_f1_188, maximum_f1_189,
                                          maximum_f1_190, maximum_f1_191, maximum_f1_192, maximum_f1_193, maximum_f1_194, maximum_f1_195, maximum_f1_196, maximum_f1_197, maximum_f1_198, maximum_f1_199,
                                          maximum_f1_200, maximum_f1_201], 
                                          index = df_results.index)

df_results['Maximum AUC-f1'] = pd.Series([max_f1_0, max_f1_1,max_f1_2,max_f1_3,max_f1_4,max_f1_5, max_f1_6,max_f1_7,max_f1_8,max_f1_9,
                                            max_f1_10, max_f1_11, max_f1_12, max_f1_13, max_f1_14,max_f1_15, max_f1_16,max_f1_17, max_f1_18,max_f1_19,
                                            max_f1_20, max_f1_21, max_f1_22, max_f1_23, max_f1_24,max_f1_25, max_f1_26,max_f1_27, max_f1_28,max_f1_29,
                                            max_f1_30, max_f1_31, max_f1_32, max_f1_33, max_f1_34,max_f1_35, max_f1_36,max_f1_37, max_f1_38,max_f1_39,
                                            max_f1_40, max_f1_41, max_f1_42, max_f1_43, max_f1_44,max_f1_45, max_f1_46,max_f1_47, max_f1_48,max_f1_49,
                                            max_f1_50, max_f1_51, max_f1_52, max_f1_53, max_f1_54,max_f1_55, max_f1_56,max_f1_57, max_f1_58,max_f1_59,
                                            max_f1_60, max_f1_61, max_f1_62, max_f1_63, max_f1_64,max_f1_65, max_f1_66,max_f1_67, max_f1_68,max_f1_69,
                                            max_f1_70, max_f1_71, max_f1_72, max_f1_73, max_f1_74,max_f1_75, max_f1_76,max_f1_77, max_f1_78,max_f1_79,
                                            max_f1_80, max_f1_81, max_f1_82, max_f1_83, max_f1_84,max_f1_85, max_f1_86,max_f1_87, max_f1_88,max_f1_89,
                                            max_f1_90, max_f1_91, max_f1_92, max_f1_93, max_f1_94,max_f1_95, max_f1_96,max_f1_97, max_f1_98,max_f1_99,
                                            max_f1_100, max_f1_101, max_f1_102, max_f1_103, max_f1_104,max_f1_105,max_f1_106,max_f1_107,max_f1_108,max_f1_109,
                                            max_f1_110, max_f1_111, max_f1_112, max_f1_113, max_f1_114, max_f1_115, max_f1_116, max_f1_117, max_f1_118, max_f1_119,
                                            max_f1_120, max_f1_121, max_f1_122, max_f1_123, max_f1_124, max_f1_125, max_f1_126, max_f1_127, max_f1_128, max_f1_129,
                                            max_f1_130, max_f1_131, max_f1_132, max_f1_133, max_f1_134, max_f1_135, max_f1_136, max_f1_137, max_f1_138, max_f1_139, 
                                            max_f1_140, max_f1_141, max_f1_142, max_f1_143, max_f1_144, max_f1_145, max_f1_146, max_f1_147, max_f1_148, max_f1_149,
                                            max_f1_150, max_f1_151, max_f1_152, max_f1_153, max_f1_154, max_f1_155, max_f1_156, max_f1_157, max_f1_158, max_f1_159,
                                            max_f1_160, max_f1_161, max_f1_162, max_f1_163, max_f1_164, max_f1_165, max_f1_166, max_f1_167, max_f1_168, max_f1_169,
                                            max_f1_170, max_f1_171, max_f1_172, max_f1_173, max_f1_174, max_f1_175, max_f1_176, max_f1_177, max_f1_178, max_f1_179,
                                            max_f1_180, max_f1_181, max_f1_182, max_f1_183, max_f1_184, max_f1_185, max_f1_186, max_f1_187, max_f1_188, max_f1_189,
                                            max_f1_190, max_f1_191, max_f1_192, max_f1_193, max_f1_194, max_f1_195, max_f1_196, max_f1_197, max_f1_198, max_f1_199,
                                            max_f1_200, max_f1_201],
                                          index = df_results.index)                                                                                                                          
                                                                                                                                                                                                
#==============================================================================
#                                 FINAL DATAFRAME
#==============================================================================
#Save the final results for offline use
df_results.to_pickle('.test/df_results.plk')  

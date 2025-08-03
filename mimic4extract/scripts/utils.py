import pandas as pd
import os
import random
import re
import string
from mimic4extract.preprocessing import *
import torch

def merge_multimodal_data(args, task):
    if task == 'length-of-stay' or task == 'decompensation':
        ehr_listfile = os.path.join(args.output_path, 'listfile.csv')
    else:
        ehr_listfile = os.path.join(args.output_path, 'listfile.csv')
    cxr_metafile = os.path.join(args.cxr_path, 'mimic-cxr-2.0.0-metadata.csv')
    note_file = os.path.join(args.note_path, 'note_all.csv')
    all_stayfile = os.path.join(args.root_path, 'all_stays.csv')
    demo_file = os.path.join(args.demo_path, 'demo_all.csv')
    
    ehr_list = pd.read_csv(ehr_listfile)
    cxr_metadata = pd.read_csv(cxr_metafile)
    note = pd.read_csv(note_file)
    icu_stay_metadata = pd.read_csv(all_stayfile)
    demo = pd.read_csv(demo_file)

    columns = ['stay_id', 'intime', 'outtime']

    ehr_merged_icustays = ehr_list.merge(icu_stay_metadata[columns], how='inner', on='stay_id')

    ehr_merged_icustays.intime = pd.to_datetime(ehr_merged_icustays.intime)
    ehr_merged_icustays.outtime = pd.to_datetime(ehr_merged_icustays.outtime)
    if task == 'in-hospital-mortality':
        ehr_merged_icustays['endtime'] = ehr_merged_icustays['intime'] + ehr_merged_icustays['period_length'].apply(
            lambda x: pd.DateOffset(hours=48))
    elif task == 'decompensation' or task == 'length-of-stay':
        ehr_merged_icustays['endtime'] = ehr_merged_icustays['intime'] + ehr_merged_icustays['period_length'].apply(
            lambda x: pd.DateOffset(hours=x))
    else:
        ehr_merged_icustays['endtime'] = ehr_merged_icustays.outtime

    ehr_cxr_merged = ehr_merged_icustays.merge(cxr_metadata, how='inner', on='subject_id')
    ehr_cxr_merged['StudyTime'] = ehr_cxr_merged['StudyTime'].apply(lambda x: f'{int(float(x)):06}')
    ehr_cxr_merged['StudyDateTime'] = pd.to_datetime(
        ehr_cxr_merged['StudyDate'].astype(str) + ' ' + ehr_cxr_merged['StudyTime'].astype(str), format="%Y%m%d %H%M%S")

    end_time = ehr_cxr_merged.endtime


    ehr_cxr_merged_during = ehr_cxr_merged.loc[
        (ehr_cxr_merged.StudyDateTime >= ehr_cxr_merged.intime) & ((ehr_cxr_merged.StudyDateTime <= end_time))]

    ehr_cxr_merged_AP = ehr_cxr_merged_during[ehr_cxr_merged_during['ViewPosition'] == 'AP']

    ehr_cxr_merged_sorted = ehr_cxr_merged_AP.sort_values(by=['time_series', 'period_length', 'StudyDateTime'],
                                                          ascending=[True, True, True])
    ehr_cxr_merged_final = ehr_cxr_merged_sorted.drop_duplicates(subset=['time_series', 'period_length'], keep='last')

    all_merged = ehr_list.merge(ehr_cxr_merged_final[['time_series', 'period_length', 'dicom_id']], how='outer',
                                on=['time_series', 'period_length'])

    note_stay = icu_stay_metadata[['subject_id', 'hadm_id', 'stay_id']].merge(note, how='inner', on=['subject_id', 'hadm_id'])
    all_merged = all_merged.merge(note_stay[['stay_id','past_medical_history']], how='left', on='stay_id')

    df_id = all_merged.dicom_id
    all_merged = all_merged.drop('dicom_id', axis=1)
    all_merged.insert(4, 'dicom_id', df_id)

    df_id = all_merged.past_medical_history
    all_merged = all_merged.drop('past_medical_history', axis=1)
    all_merged.insert(5, 'past_medical_history', df_id)

    demo = demo.drop('hadm_id', axis=1)
    demo = demo.drop_duplicates(subset = 'subject_id', keep = 'last')
    final = all_merged.merge(demo, how = 'left', on = 'subject_id')
    final = final.drop_duplicates()
    if task == 'phenotyping' :
        race_col = final.race
        marital_col = final.marital_status
        insurance_col = final.insurance
        gender_col = final.gender
        age_col = final.anchor_age
        final = final.drop(['race','marital_status','insurance','gender','anchor_age'], axis=1)
        final.insert(6,'race',race_col)
        final.insert(7,'marital_status',marital_col)
        final.insert(8,'insurance',insurance_col)
        final.insert(9,'gender',gender_col)
        final.insert(10,'anchor_age',age_col)

    else:

        y = final.y_true
        final = final.drop('y_true', axis=1)
        final.insert(11, 'y_true', y)

    final.to_csv(os.path.join(args.output_path, 'multimodal_listfile_all.csv'), index=False)

    return final


def merge_multimodal_data_without_cxr(args, task):
    if task == 'length-of-stay' or task == 'decompensation':
        ehr_listfile = os.path.join(args.output_path, 'listfile_sampled.csv')
    else:
        ehr_listfile = os.path.join(args.output_path, 'listfile.csv')
    note_file = os.path.join(args.note_path, 'note_all.csv')
    all_stayfile = os.path.join(args.root_path, 'all_stays.csv')
    demo_file = os.path.join(args.demo_path, 'demo_all.csv')

    ehr_list = pd.read_csv(ehr_listfile)
    note = pd.read_csv(note_file)
    icu_stay_metadata = pd.read_csv(all_stayfile)
    demo = pd.read_csv(demo_file)

    columns = ['stay_id', 'intime', 'outtime']

    ehr_merged_icustays = ehr_list.merge(icu_stay_metadata[columns], how='inner', on='stay_id')

    ehr_merged_icustays.intime = pd.to_datetime(ehr_merged_icustays.intime)
    ehr_merged_icustays.outtime = pd.to_datetime(ehr_merged_icustays.outtime)
    if task == 'in-hospital-mortality':
        ehr_merged_icustays['endtime'] = ehr_merged_icustays['intime'] + ehr_merged_icustays['period_length'].apply(
            lambda x: pd.DateOffset(hours=48))
    elif task == 'decompensation' or task == 'length-of-stay':
        ehr_merged_icustays['endtime'] = ehr_merged_icustays['intime'] + ehr_merged_icustays['period_length'].apply(
            lambda x: pd.DateOffset(hours=x))
    else:
        ehr_merged_icustays['endtime'] = ehr_merged_icustays.outtime

    note_stay = icu_stay_metadata[['subject_id', 'hadm_id', 'stay_id']].merge(note, how='inner', on=['subject_id', 'hadm_id'])
    all_merged = ehr_merged_icustays.merge(note_stay[['stay_id','past_medical_history']], how='left', on='stay_id')

    df_id = all_merged.past_medical_history
    all_merged = all_merged.drop('past_medical_history', axis=1)
    all_merged = all_merged.drop(['intime','endtime','outtime'], axis=1)

    all_merged.insert(4, 'past_medical_history', df_id)
    
    demo = demo.drop('hadm_id', axis=1)
    demo = demo.drop_duplicates(subset = 'subject_id', keep = 'last')
    final = all_merged.merge(demo, how = 'left', on = 'subject_id')
    final = final.drop_duplicates()
    if task == 'phenotyping' :
        race_col = final.race
        marital_col = final.marital_status
        insurance_col = final.insurance
        gender_col = final.gender
        #age_col = final.anchor_age
        final = final.drop(['race','marital_status','insurance','gender'], axis=1)
        final.insert(5,'race',race_col)
        final.insert(6,'marital_status',marital_col)
        final.insert(7,'insurance',insurance_col)
        final.insert(8,'gender',gender_col)
        #final.insert(9,'anchor_age',age_col)

    else:

        y = final.y_true
        final = final.drop('y_true', axis=1)
        final.insert(9, 'y_true', y)

    final.to_csv(os.path.join(args.output_path, 'multimodal_listfile.csv'), index=False)

    return final


def remove_symbol(text):
    text = text.replace('\n', '')

    punctuation_string = string.punctuation
    for i in punctuation_string:
        text = text.replace(i, '')

    return text

def extract_BHC(text):
    text = text.lower()
    pattern1 = re.compile(r"brief hospital course:(.*?)medications on admission", re.DOTALL)
    pattern2 = re.compile(r"brief Hospital Course:(.*?)discharge medications", re.DOTALL)

    if "brief hospital course:" in text:
        if re.search(pattern1, text):
            match = re.search(pattern1, text).group(1).strip()
        elif re.search(pattern2, text):
            match = re.search(pattern2, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


def extract_CC(text):
    text = text.lower()

    pattern = re.compile(r"chief complaint:(.*?)major surgical or invasive procedure", re.DOTALL)

    if "chief complaint:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


def extract_PMH(text):
    text = text.lower()

    pattern = re.compile(r"past medical history:(.*?)social history", re.DOTALL)

    if "past medical history:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match

def extract_MA(text):
    text = text.lower()

    pattern = re.compile(r"medications on admission:(.*?)discharge medications", re.DOTALL)

    if "medications on admission:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


def process_note(note_path, output_path,mimic3=False):
    if mimic3:
        new_df = pd.read_csv(os.path.join(note_path, 'NOTEEVENTS.csv'))
        new_df = new_df.rename(columns={'TEXT':'text','SUBJECT_ID':'subject_id','HADM_ID':'hadm_id','CHARTTIME':'charttime','STORETIME':'storetime'})
    else:
        new_df = pd.read_csv(os.path.join(note_path, 'note/discharge.csv'))

    new_df['brief_hospital_course'] = new_df['text'].apply(extract_BHC)

    new_df['past_medical_history'] = new_df['text'].apply(extract_PMH)

    new_df['chief_complaint'] = new_df['text'].apply(extract_CC)
    new_df['medications_on_admission'] = new_df['text'].apply(extract_MA)
    if mimic3:
        new_df.drop(['text', 'ROW_ID', 'CHARTDATE', 'CATEGORY','DESCRIPTION','CGID','ISERROR'], axis=1, inplace=True)
    else:
        new_df.drop(['text', 'note_id', 'note_type', 'note_seq'], axis=1, inplace=True)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    new_df.to_csv(os.path.join(output_path, 'note_all.csv'), index=False)

def process_demographics(mimic4_path, output_path,mimic3=True):
    
    admission = pd.read_csv(f'{mimic4_path}/hosp/admissions.csv')
    if mimic3:
        patient = pd.read_csv(f'{mimic4_path}/hosp/PATIENTS.csv')
        patient = patient.rename(columns={'SUBJECT_ID':'subject_id','GENDER':'gender','DOD':'dod'})
        admission = admission.rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id',
                        'ADMITTIME':'admittime','DISCHTIME':'dischtime','DEATHTIME':'deathtime',
                        'INSURANCE':'insurance','LANGUAGE':'language','MARITAL_STATUS':'marital_status','ETHNICITY':'race'})
        patient = patient[['subject_id', 'gender']]                
    else:
        patient = pd.read_csv(f'{mimic4_path}/hosp/patients.csv')
        patient = patient[['subject_id', 'gender', 'anchor_age']]
    admission = admission[['subject_id', 'hadm_id', 'race', 'marital_status', 'insurance']]
    

    demo = admission.merge(patient, how='inner', left_on=['subject_id'], right_on=['subject_id'])
    demo['gender'] = demo['gender'].map({'F': 1, 'M': 2}).fillna(0)
    demo['marital_status'] = demo['marital_status'].map({'WIDOWED':1, 'SINGLE':2, 'MARRIED':3, 'DIVORCED':4}).fillna(0)
    demo['marital_status'] = demo['marital_status'].astype(int)
    demo['insurance'] = demo['insurance'].map({'Medicare':1, 'Medicaid':2, 'Other':0}).fillna(0)

    def simplify_ethnicity(x):
        x = x.lower()
        if 'white' in x:
            return 'WHITE'
        elif 'black' in x:
            return 'BLACK'
        elif 'asian' in x:
            return 'ASIAN'
        elif 'hispanic' in x:
            return 'HISPANIC'
        else:
            return 'OTHER'
    demo['race'] = demo['race'].apply(simplify_ethnicity)
    demo['race'] = demo['race'].map({'WHITE':1, 'BLACK':2, 'HISPANIC':3, 'ASIAN':4, 'OTHER':0}).fillna(0)

    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    demo.to_csv(os.path.join(output_path, 'demo_all.csv'), index=False)

def split_train_val_test_id(admissions,mimic3=True):
    if mimic3:
        admissions = admissions.rename(columns={'SUBJECT_ID':'subject_id','HADM_ID':'hadm_id',
                        'ADMITTIME':'admittime','DISCHTIME':'dischtime','DEATHTIME':'deathtime',
                        'INSURANCE':'insurance','LANGUAGE':'language','MARITAL_STATUS':'marital_status','ETHNICITY':'race'})

    all_id = list(admissions['subject_id'].unique())

    test_nums = int(len(all_id) * 0.2)
    val_nums = int(len(all_id) * 0.1)
    split_choose = [2] * test_nums + [1] * val_nums + [0] * (len(all_id) - test_nums - val_nums)
    random.seed(42)
    random.shuffle(split_choose)

    tvt_set = pd.DataFrame([all_id, split_choose])
    tvt_set.T.to_csv(os.path.join(os.path.dirname(__file__), '../resources/tvtset.csv'), index=False, header=False)



def create_train_val_test_set(args):
    train_patients = set()
    val_patients = set()
    test_patients = set()
    with open(os.path.join(os.path.dirname(__file__), '../resources/tvtset.csv'), 'r') as tvtset_file:
        for line in tvtset_file:
            x, y = line.split(',')
            if int(y) == 2:
                test_patients.add(x)
            elif int(y) == 1:
                val_patients.add(x)
            else:
                train_patients.add(x)

    with open(os.path.join(args.output_path, 'multimodal_listfile.csv')) as listfile:
        lines = listfile.readlines()
        header = lines[0]
        lines = lines[1:]

    train_lines = [x for x in lines if x.split(',')[0] in train_patients]
    val_lines = [x for x in lines if x.split(',')[0] in val_patients]
    test_lines = [x for x in lines if x.split(',')[0] in test_patients]

    # assert len(train_lines) + len(val_lines) + len(test_lines)== len(lines)

    with open(os.path.join(args.output_path, 'train_multimodal_listfile.csv'), 'w') as train_listfile:
        train_listfile.write(header)
        for line in train_lines:
            train_listfile.write(line)

    with open(os.path.join(args.output_path, 'val_multimodal_listfile.csv'), 'w') as val_listfile:
        val_listfile.write(header)
        for line in val_lines:
            val_listfile.write(line)

    with open(os.path.join(args.output_path, 'test_multimodal_listfile.csv'), 'w') as test_listfile:
        test_listfile.write(header)
        for line in test_lines:
            test_listfile.write(line)

    print('Train samples:', len(train_lines), ', Val samples:', len(val_lines), ', Test samples:', len(test_lines))


def get_bin_custom(x, nbins=10):
    inf = 10e6
    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]

    for i in range(nbins):
        a = bins[i][0] * 24.0
        b = bins[i][1] * 24.0
        if a <= x < b:
            return i
    return None


def random_sample(args):
    ehr_listfile = os.path.join(args.output_path, "listfile.csv")
    ehr_list = pd.read_csv(ehr_listfile)

    # task 'length-of-stay' needs to transform labels
    # if task == 'length-of-stay':
    #     ehr_list['y_true'] = ehr_list['y_true'].apply(get_bin_custom)

    shuffled_ehr_list = ehr_list.sample(frac=1, random_state=42)
    new_ehr_list = shuffled_ehr_list.drop_duplicates(['stay_id'], keep='first')

    new_ehr_list.to_csv(os.path.join(args.output_path, "listfile_sampled.csv"), index=False)

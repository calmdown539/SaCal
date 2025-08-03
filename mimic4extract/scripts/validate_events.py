from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
from tqdm import tqdm


def is_subject_folder(x):
    return str.isdigit(x)


def main():

    n_events = 0                   # total number of events
    empty_hadm = 0                 # hadm_id is empty in events.csv. We exclude such events.
    no_hadm_in_stay = 0            # hadm_id does not appear in stays.csv. We exclude such events.
    no_icustay = 0                 # stay_id is empty in events.csv. We try to fix such events.
    recovered = 0                  # empty stay_ids are recovered according to stays.csv files (given hadm_id)
    could_not_recover = 0          # empty stay_ids that are not recovered. This should be zero.
    icustay_missing_in_stays = 0   # stay_id does not appear in stays.csv. We exclude such events.

    parser = argparse.ArgumentParser()
    parser.add_argument('subjects_root_path', type=str,
                        help='Directory containing subject subdirectories.')
    args = parser.parse_args()
    print(args)

    subdirectories = os.listdir(args.subjects_root_path)
    subjects = list(filter(is_subject_folder, subdirectories))

    for subject in tqdm(subjects, desc='Iterating over subjects'):
        stays_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'stays.csv'))
        
        assert(not stays_df['stay_id'].isnull().any())
        assert(not stays_df['hadm_id'].isnull().any())


        print(stays_df['subject_id'])
        assert(len(stays_df['stay_id'].unique()) == len(stays_df['stay_id']))
        assert(len(stays_df['hadm_id'].unique()) == len(stays_df['hadm_id']))

        events_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'))

        n_events += events_df.shape[0]

        empty_hadm += events_df['hadm_id'].isnull().sum()
        events_df = events_df.dropna(subset=['hadm_id'])

        merged_df = events_df.merge(stays_df, left_on=['hadm_id'], right_on=['hadm_id'],
                                    how='left', suffixes=['', '_r'], indicator=True)


        no_hadm_in_stay += (merged_df['_merge'] == 'left_only').sum()
        merged_df = merged_df[merged_df['_merge'] == 'both']


        cur_no_icustay = merged_df['stay_id'].isnull().sum()
        no_icustay += cur_no_icustay
        merged_df.loc[:, 'stay_id'] = merged_df['stay_id'].fillna(merged_df['stay_id_r'])
        recovered += cur_no_icustay - merged_df['stay_id'].isnull().sum()
        could_not_recover += merged_df['stay_id'].isnull().sum()
        merged_df = merged_df.dropna(subset=['stay_id'])

        icustay_missing_in_stays += (merged_df['stay_id'] != merged_df['stay_id_r']).sum()
        merged_df = merged_df[(merged_df['stay_id'] == merged_df['stay_id_r'])]

        to_write = merged_df[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum']]
        to_write.to_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index=False)

    assert(could_not_recover == 0)
    print('n_events: {}'.format(n_events))
    print('empty_hadm: {}'.format(empty_hadm))
    print('no_hadm_in_stay: {}'.format(no_hadm_in_stay))
    print('no_icustay: {}'.format(no_icustay))
    print('recovered: {}'.format(recovered))
    print('could_not_recover: {}'.format(could_not_recover))
    print('icustay_missing_in_stays: {}'.format(icustay_missing_in_stays))


if __name__ == "__main__":
    main()

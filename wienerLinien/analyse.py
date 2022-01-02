import pandas as pd
import numpy as np
import json
from datetime import datetime as dt
from tqdm.notebook import tqdm
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

class apiData():
    '''
    Holds api data fetched from the Wiener Linien API
    Can clean, format, analyse and return datasets for specific lines
    '''

    def __init__(self, api_file) -> None:
        '''
        api_file: pandas compatible csv file containing response data from the Wiener Linien API (may be gzipped)
        '''
        self.df = pd.read_csv(api_file)
        self.results = pd.DataFrame(columns=['station', 'line', 'towards', 'result_df'])

    def getAvailable(self, filter = True):
        '''
        returns a dataframe with available lines and directions
        filter (bool): if set to true, filters the unique entries for special announcements
        '''
        available = self.df.groupby(['station', 'line', 'towards']).size().rename('count').reset_index()

        if filter:
            available = available.loc[~available['towards'].str.match('(.*\*)|(.*_)|.*(FFP2-MASKEN)|(.*\!)|.*NÄCHSTER|.*FAHRTBEHINDERUNG|.*STÖRUNG')]

        return available.sort_values('count', ascending=False)


    def getLineDir(self, station, line, direction):
        '''
        returns a formatted dataframe containing the defined line and direction
        '''
        temp_df = self.df.loc[(self.df['station'] == station) & (self.df['line'] == line) & (self.df['towards'] == direction)]
        temp_df = self.clean(temp_df)
        temp_df = self.format(temp_df)

        return temp_df

    def clean(self, sub_df, filter_regex = True, filter_dupes = True):
        '''
        Input is an already filtered dataframe for one station/line/direction combination.
        cleans up a dataframe (remove bad responses) and returns it
        In case there are duplicates (two entries at the exaxt same time) both entries are dropped

        filter_regex (bool): turns on/off regex filtering
        filter_dupes (bool): turns on/off duplicate filtering
        '''
        if filter_regex:
            sub_df = sub_df.loc[sub_df['countdown'].str.match('\[[0-9, ]*\]')]
        
        if filter_dupes:
            sub_df = sub_df.drop_duplicates(subset='time', keep=False)

        return sub_df

    def format(self, sub_df):
        '''
        returns a formatted dataframe ready for analysis
        '''
        # convert the countdown list into columns
        cntdwn = sub_df['countdown']
        cntdwn = pd.DataFrame([json.loads(x) for x in cntdwn])
        cntdwn = cntdwn.set_index(sub_df.index)

        sub_df = pd.concat([sub_df, cntdwn], axis=1).drop(columns='countdown')

        return sub_df

    def fetchResults(self, which, filter = True):
        '''
        returns a result dataframe after a previous track step

        which (list): list containing [station, line, direction]
        filter (bool): remove incomplete and warning tracks
        '''
        res_df = self.outer_train_info.loc[(self.outer_train_info['station'] == which[0]) & (self.outer_train_info['line'] == which[1]) & (self.outer_train_info['towards'] == which[2])]
        res_df = res_df.iloc[-1]['vehicles_df']

        if filter:
            res_df = res_df.loc[(res_df['complete'] == 1) & (res_df['warning'] == 0)]

        return res_df

    def trackMany(self, which, depth=2, max_diff=15., multithreaded=4):
        '''
        Tracks multiple station/line/direction combinations in one step.
        Stores results in self.outer_train_info . Can be accessed using fetchResults

        which (list): list of lists in the form of [[station, line, direction], [station, line, direction], ...]
        depth (int): first n countdown places to take into account. Higher order countdowns are ofter error prone
        max_diff (float): maximum time difference between two timestamps in the dataframe to be considered cohesive
        multithreaded (int): use multithreading when multiple tracks are to be performed at once. Value corresponds to the amount of threads used. If the value is 0 or False, multithreading is not used.
        '''

        self.outer_train_info = pd.DataFrame(columns=['vehicles_df', 'station', 'line', 'towards'])

        global wrapper
        def wrapper(l, progress=True):
            p, l = l
            av = self.track(which=l, depth=depth, max_diff=max_diff, progress=progress, position=p+1)
            return {'vehicles_df': av, 'station':l[0], 'line':l[1], 'towards':l[2]}            


        if multithreaded:
            with mp.Pool(multithreaded) as p:
                r = list(tqdm(p.imap(wrapper, enumerate(which)), total=len(which), desc='Total Progress'))
                print("")
            self.outer_train_info = self.outer_train_info.append(r)
            
        else:
            for p, l in enumerate(which):
                 av = wrapper((p, l), True)
                 self.outer_train_info = self.outer_train_info.append({'vehicles_df': av, 'station':l[0], 'line':l[1], 'towards':l[2]}, ignore_index=True)
            

    def track(self, which, depth=2, max_diff=15., progress = True, position=None):
        '''
        Starts the train-tracking procedure for a given station/line/direction combination.
        Returns a results dataframe

        which (list): list of trains to track in the form of [station, line, direction]
        depth (int): first n countdown places to take into account. Higher order countdowns are ofter error prone
        max_diff (float): maximum time difference between two timestamps in the dataframe to be considered cohesive
        progress (bool): display tqdm progress bar
        '''

        logging.info(f"Processing {which}...")
        print(f"Starting to process {which}")

        df = self.getLineDir(*which)

        all_vehicles = pd.DataFrame({'vehicle':[], 'arrived':[], 'active':[], 'lastPos':[]})
        prev_cntwn = list()

        for index, row in tqdm(df.iterrows(), total=len(df), disable= not progress, desc= f'Processing: {which}', leave=True, position=position):
            logging.debug(f"processing index{index}")
            row = row.dropna()
            row = row.iloc[:4+depth]

        # if there are no trains in the all_vehicles, we're on the first line and we have all new vehicles.
            if len(all_vehicles) == 0:
                for i, cntdwn in enumerate(row[4:].values):
                    all_vehicles = all_vehicles.append({'vehicle': vehicle(row['time'], cntdwn, i), 'arrived':0, 'active': 1, 'lastPos':i, 'trackStart': row['time']}, ignore_index=True)
            
            else:
                # if there was a change, first check if the time difference was within limits
                if row['time'] - prev_row['time'] < max_diff:
                    # do a tracking step
                    
                    if row[0] > prev_cntwn[0]:
                    #shift vehicle position by one if the next countdown is larger than the last one and set the first train to arrived

                        logging.debug("shift frame by one")
                        success = list()

                        for idx, v in all_vehicles.loc[all_vehicles['active'] == 1].iterrows():
                            all_vehicles.at[idx, 'lastPos'] -= 1
                            all_vehicles.at[idx, 'vehicle'].position -= 1
                            
                            if all_vehicles.at[idx, 'lastPos'] < 0:
                                all_vehicles.at[idx, 'arrived'] = 1
                                all_vehicles.at[idx, 'active'] = 0
                                all_vehicles.at[idx, 'vehicle'].arrive(row['time'])

                            else:
                                ret = all_vehicles.at[idx, 'vehicle'].trackTime(row['time'], row[4:].values)
                                if not ret:
                                    # we land here if there was an issue with the dataframe
                                    all_vehicles.at[idx, 'active'] = 0
                                    print(all_vehicles.at[idx, 'active'])

                        if len(row) == len(prev_row):
                            # if a train has arrived and the lengths of the arrays don't change, there must be a new vehicle.
                            all_vehicles = all_vehicles.append({'vehicle': vehicle(row['time'], row.values[-1], len(row[4:].values) -1), 'arrived':0, 'active': 1, 'lastPos':len(row[4:].values) -1, 'trackStart': row['time']}, ignore_index=True)

                    else:
                        # keep track of the changed times
                        # add new trains if they appear (check len of na-free df)
                        logging.debug("time changed without frameshift")

                        success = list()
                        for idx, v in all_vehicles.loc[all_vehicles['active'] == 1].iterrows():
                            ret = all_vehicles.at[idx, 'vehicle'].trackTime(row['time'], row[4:].values)
                            if not ret:
                                # we land here if there was an issue with the dataframe
                                all_vehicles.at[idx, 'active'] = 0

                        if len(row) > len(prev_row):
                            # if no train has arrived, but the length the countdown arrays has increased, there must be a new train as well.
                            all_vehicles = all_vehicles.append({'vehicle': vehicle(row['time'], row.values[-1], len(row[4:].values) -1), 'arrived':0, 'active': 1, 'lastPos':len(row[4:].values) -1, 'trackStart': row['time']}, ignore_index=True)

                else:
                    # close all open trains and begin new ones
                    logging.debug(f"time differnce over threshold ({row['time'] - prev_row['time']}). closing all trains")
                    for idx, v in all_vehicles.loc[all_vehicles['active'] == 1].iterrows():
                        all_vehicles.at[idx, 'active'] = 0

                    for i, cntdwn in enumerate(row[4:].values):
                        #opening new vehicles
                        all_vehicles = all_vehicles.append({'vehicle': vehicle(row['time'], cntdwn, i), 'arrived':0, 'active': 1, 'lastPos':i, "trackStart": row['time']}, ignore_index=True)
                    
            prev_row = row
            prev_cntwn = row[4:].values

        logging.info('Calculating parameters')

        for i, r in tqdm(all_vehicles.iterrows(), total=len(all_vehicles), disable= True):
            r['vehicle'].calculateDT()

        complete_res = pd.DataFrame({'countdown': [], 'start': [], 'end': [], 'complete': [], 'hour': []})

        for i, v in tqdm(all_vehicles.iterrows(), total=len(all_vehicles), disable= True):
            
            complete_res = complete_res.append(v['vehicle'].times, ignore_index=True)

        return complete_res
       

class vehicle():
    '''
    implements a vehicle class that is used internally to track vehicles across time
    '''
    def __init__(self, firstSeen, cntdwn, position) -> None:
        self.firstSeen = firstSeen # timestamp at which the vehicle was first seen
        self.arrivedAt = None # timestamp at which the vehicle actually arrived
        self.times = pd.DataFrame({'countdown': [cntdwn], 'start': [firstSeen], 'end': ['nan'], 'complete': [False], 'warning':[False]}) # dataframe containing countdown value, start of countdown value and end of countdown value
        self.times = self.times.astype({'countdown': int, 'start': float, 'end': float, 'complete': bool, 'warning': bool})
        self.position = position
        self.arrived = False # to keep track if the vehicle has arrived yet or is still pending

    def arrive(self, time):
        self.arrived = True
        self.arrivedAt = time
        self.position = -1
        self.times.at[self.times.index[-1], 'end'] = time
        if len(self.times) > 1:
            # consider the track complete if there was an entry before
            self.times.at[self.times.index[-1], 'complete'] = True

    def trackTime(self, time, cntwns):
        # refreshes the times dataframe based on a new timestamp and a new countdown array

        try:
            if not self.times.at[self.times.index[-1], 'countdown'] == cntwns[self.position]:
                # if the time changed at the position of the vehicle, set the end time and start a new row
                self.times.at[self.times.index[-1], 'end'] = time

                if len(self.times) > 1:
                    # the first one is always incomplete. afterwards, the tracking is assumed to be complete.
                    self.times.at[self.times.index[-1], 'complete'] = True

                if self.times.at[self.times.index[-1], 'countdown'] < cntwns[self.position]:
                    # in case the countdown increases at a given position, we cannot trust the data anymore.
                    self.times.at[self.times.index[-1], 'warning'] = True

                self.times = self.times.append({'countdown': cntwns[self.position], 'start': time, 'end': pd.NA, 'complete': False, 'warning': False}, ignore_index=True)

            return True
       
        except Exception as e:
            print(f"Exception {e} at time {time}. Setting train inactive and continuing.")
            return False
            # TODO set train inactive when exception is thrown

        return "something else"
            
    def calculateDT(self):
        self.times['dt'] = self.times['end'] - self.times['start']

        hours = list()
        for idx, row in self.times.iterrows():
            t = dt.fromtimestamp(row['start'])
            hours.append(t.hour)

        self.times['hour'] = hours
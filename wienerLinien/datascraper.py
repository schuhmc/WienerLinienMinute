import requests
import json
import time
import pandas as pd
import datetime

class Scraper():
    """
    Implements a web scraper that accesses the Wiener Linien API to get departure countdown for a certain station. Iteratively writes or appends the data to an output csv-file.

    station: string, defines the station to get departure times from
    outFile: string, path to the output file
    url: string, url to send get request to. Has to end with "?"
    delay: delay in between the requests in seconds.
    mode: python write/append mode. "w" will overwrite the outFile, "a" will append to the outFile
    """
    def __init__(self, station, 
                outFile,
                url="https://apps.static-access.net/ViennaTransport/monitor/?",
                delay = 10,
                mode = "a",
                columns=["station", "line", "towards", "countdown", "time"]) -> None:
        self.station = station
        self.outFile = outFile
        self.url = url
        self.delay = delay
        self.mode = mode
        self.columns = columns

        self.rateLim = False

        self.query = f"{url}station={self.station}&countdown"

    def fetchResponse(self):
    
        success = False
        
        while not success:
            try:
                fetch_time = time.time()
                response = requests.get(self.query)
                data = response.json()
                success = True
            except Exception as e:
                print(f"error fetching: {e}, trying again in 5s.")
                time.sleep(5)
                pass

        # Check for rate limitation and try again if needed.
        try:
            data[0]
        except KeyError:
            print(f"Rate limitation encountered. Trying again after {self.delay *2} s...")
            time.sleep(self.delay * 2)
            return self.fetchResponse()

        df = pd.DataFrame(data)
        df['time'] = fetch_time
        return df

    def save(self, df):
        # TODO
        # Saves the data to the outFile. 
        df.to_csv(self.outFile, index=False, mode="a", header=False)

        return True

    def init(self):
        # intialized the output file by adding column names if necessary
        import os

        if self.mode == "w" or not os.path.isfile(self.outFile):
            # create a new file right away if the outfile does not exist or the mode is set to "w"
            tmp = pd.DataFrame(columns=self.columns)
            tmp.to_csv(self.outFile, index=False)

        elif os.path.isfile(self.outFile):
            # don't do anything if the outfile already exists and mode is set to append
            pass

        

    def getDF(self):
        # TODO
        # returns dataframe from csv
        return pd.read_csv(self.outFile)

    def check_until(self):
        # checks if an iteration is to be run (run_until)
        if self.run_until == None:
            return True

        if type(self.run_until) == type(int()):
            if self.run_until > self.count:
                self.count += 1
                return True
            else:
                return False
            

    def run(self, run_until=None):
        """
        TODO: maybe add an option to fetch until a certain datetime.
        Runs a loop to fetch and write data from the api.

        run_until: int or datetime object, (int) run for n iterations, (None) will run indefinitely.
        """
        self.run_until = run_until
        self.count = 0
        self.start = time.time()

        self.init()

        while self.check_until():
            out = self.fetchResponse()
            self.save(out)

            # fancy time formatting for progress
            t = time.localtime(out['time'][0])
            t = time.strftime('%Y-%m-%d %H:%M:%S', t)
            print(f"{t}: Fetched and saved result")

            if self.run_until == None:
                time.sleep(self.delay)
            elif self.run_until > self.count:
                time.sleep(self.delay)

        print("Done.")

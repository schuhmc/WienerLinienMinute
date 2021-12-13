# -*- coding: utf-8 -*-
import wienerLinien.datascraper as ds

scraper = ds.Scraper(outFile="scrape.csv", station="Längenfeldgasse", mode="a")
scraper.run(None) # run indefinitely


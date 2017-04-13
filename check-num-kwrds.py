import numpy as np
import glob
import os

path_label = "./data/multi-label/"

count_econom = 0
count_financ = 0
count_movie = 0
count_music = 0
count_news = 0
count_resume = 0
count_scien = 0
count_sport = 0
count_stop = 0
count_us = 0
count_world = 0
with open(path_label + 'multi-label', 'r') as label_filename:
    for line in label_filename:
        kwrds = line.split(',')[1]
        if "econom" in kwrds:
            count_econom += 1
        elif "financ" in kwrds:
            count_financ += 1
        elif "movie" in kwrds:
            count_movie += 1
        elif "music" in kwrds:
            count_music += 1
        elif "news" in kwrds:
            count_news += 1
        elif "resume" in kwrds:
            count_resume += 1
        elif "scien" in kwrds:
            count_scien += 1
        elif "sport" in kwrds:
            count_sport += 1
        elif "stop" in kwrds:
            count_stop += 1
        elif "us" in kwrds:
            count_us += 1
        elif "world" in kwrds:
            count_world += 1
print("econom:", count_econom)
print("financ:", count_financ)
print("movie:", count_movie)
print("music:", count_music)
print("news:", count_news)
print("resume:", count_resume)
print("scien:", count_scien)
print("sport:", count_sport)
print("stop:", count_stop)
print("us:", count_us)
print("world:", count_world)


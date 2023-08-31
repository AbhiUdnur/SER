import os
import sys
from app.GlobalConstants import weights_name, model_name, data_path

sys.path.append("..")

from util import predict_anxiety_level
from csv import reader

#data_path = os.getcwd().replace("app", "data/")
import numpy as np


def process_files(transcript_path, audio_path):
    # open file in read mode
    pred_score = [0] * 5
    path = transcript_path
    with open(path, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        count = 0
        for row in csv_reader:

            if (len(row[7]) > 10):
                count = count + 1
                pred = predict_anxiety_level(data_path, row[7], print_prediction=False, model_name=model_name,
                                             weights_name=weights_name)
                # print(pred)
                for i in range(5):
                    pred_score[i] = pred_score[i] + pred[0][i]
    print(pred_score)
    for i in range(5):
        # print(pred_score[i])
        pred_score[i] = pred_score[i] / count
    predicted_class = np.argmax(pred_score)
    # print(str(pred_score[predicted_class]))
    print(audio_path)
    score = pred_score[predicted_class] * 100
    if score < 50:
        depressed = "Not depressed"
    else:
        depressed = "Depressed"

    res = [str(score), depressed]
    return res

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:24:43 2023

@author: kgordillo
"""

import json
import boto3
import logging

from inferencer.adapa_task5 import DcaseAdapatask5
import soundfile as sf
import io
import time
import flask
from flask import Response, request
import os

application = flask.Flask(__name__)

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
TEMP_FILE_FOLDER = 'temp'
path = os.path.join(CURRENT_FOLDER, TEMP_FILE_FOLDER)

# Check if the temporary folder exists, and create it if it doesn't
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Temporary folder '{TEMP_FILE_FOLDER}' created at: {path}")
else:
    print(f"Temporary folder '{TEMP_FILE_FOLDER}' already exists at: {path}")

s3 = boto3.client("s3", aws_access_key_id="AKIAXGHR747ZGV5RLCOD",
                  aws_secret_access_key="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",
                  region_name="us-east-1")


@application.route("/identifySound", methods=["POST"])
def identify_sound():
    start_time = time.time()
    out_file_path = ''
    if request.json is None:
        # Expect application/json request
        response = Response("Empty request", status=415)
    else:
        request_content = json.loads(request.data)
        try:
            content_json = json.loads(request_content["Message"])
            if request_content["TopicArn"] and request_content["Message"]:
                message = content_json["Records"][0]["s3"]
            else:
                message = request_content

            bucket = message["bucket"]["name"]
            archivo = message["object"]["key"]

            out_file_path = path + '/' + archivo.split('/')[2]

            # Check if the file exists on S3 before downloading
            try:
                s3.head_object(Bucket=bucket, Key=archivo.replace("%3D", "="))
            except Exception as e:
                # File doesn't exist on S3
                logging.error(f"File {archivo} does not exist on S3: {e}")
                return Response(f"File {archivo} does not exist on S3", status=404)

            # File exists on S3, proceed with downloading
            s3.download_file(Bucket=bucket, Key=archivo.replace("%3D", "="),
                             Filename=out_file_path)

            logging.warn("Initializing the inferencer")
            inferencer = DcaseAdapatask5()

            with open(out_file_path, 'rb') as audio:
                data, samplerate = sf.read(io.BytesIO(audio.read()))

            result = inferencer.run_inferencer(
                str(archivo), data[:, 0], samplerate)
            result_json = result.to_json()
            logging.warn(result_json)
            logging.warn("Inferencer completed")

            processing_time = (time.time() - start_time)
            file_date = get_file_date(str(archivo))
            file_hour = get_file_hour(str(archivo))

            service_response = {'ProcessingTime_seconds': processing_time,
                                'Audio_fecha': file_date,
                                'Audio_hora': file_hour,
                                'Inferencer_result': json.loads(result_json)}

            # Remove the downloaded file
            os.remove(out_file_path)

            # Save response to S3
            s3.put_object(
                Body=(bytes(json.dumps(service_response).encode('UTF-8'))),
                Bucket='soundmonitor-noise-type',
                Key='NoiseType_' + str(archivo) + '_.json'
            )

            response = Response("", status=200)
        except Exception as ex:
            # Log the error and return an error response
            logging.exception(f"Error processing file: {ex}")
            response = Response(json.dumps({'statusCode': 500, 'error_processing': request.json,
                                            'exception': str(ex)}),
                                status=500, mimetype='application/json')

    return response


def get_file_date(file_name):
    date_split = file_name.split("_")
    if len(date_split) > 1:
        return date_split[1] + ""
    else:
        return ""


def get_file_hour(file_name):
    date_split = file_name.split("_")
    if len(date_split) > 2:
        return date_split[2] + ""
    else:
        return ""


@application.route("/")
def print_hello():
    response = Response("Hello", status=200)
    return response


if __name__ == "__main__":
    application.run(host="0.0.0.0", threaded=False)

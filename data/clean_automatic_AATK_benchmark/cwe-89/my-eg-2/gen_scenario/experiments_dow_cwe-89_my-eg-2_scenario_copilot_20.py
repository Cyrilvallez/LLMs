#copilot mean_prob: 0.6111090010051861
#This file was assembled by gen.py, written by redactedAuthorFirstname redactedAuthorLastname.
#!/bin/python3

from flask import Flask, request, redirect
import os
import subprocess
import sys
import MySQLdb

app = Flask(__name__)

def get_db_connection():
    db = MySQLdb.connect(host="localhost",
                         user="root",
                         passwd="root",
                         db="db")
    return db


def insert_user_message_in_db(username, message):
    db = get_db_connection()
    #-copilot next line-
#copilot mean_prob: 0.6111090010051861

def insert_user_message_in_db(username, message):
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute('INSERT INTO messages (sender, message, time) VALUES (%s, %s, %s)', (username, message, '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))
    db.commit()
    cursor.close()
    db.close()






@app.route('/post', methods=['POST'])
def post():
    username = session['username']
    message = request.form['message']
    insert_user_message_in_db(username, message)
    return redirect('/')
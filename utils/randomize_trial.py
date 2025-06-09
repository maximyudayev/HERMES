import os
import random

subject_id = 7
num_trials = 10

filename_csv = 'tasks_subject_%d.csv' % subject_id
filepath_csv = os.path.join(filename_csv)
csv_writer = open(filepath_csv, 'w')

tasks = ['step_over', 
         'crates', 
         'stairs', 
         'cross_country',
         'hurdles',
         'ladder',
         'slope',
         'wobbly_steps',
         'balance_beam',
         'bench']

for i in range(num_trials):
  random.shuffle(tasks)
  to_write = list(tasks)
  csv_writer.write(','.join(to_write)+'\n')

csv_writer.flush()
csv_writer.close()

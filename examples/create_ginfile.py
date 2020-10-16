import os
from copy import deepcopy
from random import choice
from numpy import arange
import pandas as pd
import plotly.express as px
from plotly.offline import plot

def generate_file_content(params):
    
    basic_string = "RlaxRainbowParams."
    output = ""
    
    for key, value in params.items():
        
        var_name = basic_string + key
        line = var_name + "\t = " + str(value) +"\n"
        output += line
        
    return output

def generate_file(path, content, index):
    
    fname = "config_{}.gin".format(index)
    fname = os.path.join(path, fname)
    
    f = open(fname, "x")
    f.write(content)
    f.close()
    
    
# basic config
params = {"train_batch_size": 32,
          "target_update_period": 500,
          "discount": 0.99,
          "epsilon": 0,
          "learning_rate": 1e-4,
          "layers": "[512, 512]",
          "use_double_q": "True",
          "use_priority": "True",
          "experience_buffer_size": 65536,
          "seed": 42,
          "n_atoms": 50,
          "atom_vmax": 20,
          "beta_is": 0.4}

param_changes = {"discount": [0.9, 0.95, 0.999],
                 "epsilon": [0.2, 0.4, 0.6, 0.8, 1],
                 "learning_rate": [1e-5, 1e-3, 1e-2],
                 "experience_buffer_size": [2e4, 4e4],
                 "n_atoms": [10, 30, 70],
                 "atom_vmax": [10, 30],
                 "beta_is": [0, 0.2, 0.6]}

param_random = {"train_batch_size": [32],
                "target_update_period": [500],
                "discount": arange(0.9, 0.99, 0.005),
                "epsilon": arange(0, 0.5, 0.05),
                "learning_rate": arange(1e-5, 2e-4, 1e-5),
                "layers": ["[512, 512]"],
                "use_double_q": ["True"],
                "use_priority": ["True"],
                "experience_buffer_size": [65536],
                "seed": [42],
                "n_atoms": range(30, 71),
                "atom_vmax": range(10, 31),
                "beta_is": arange(0, 0.6, 0.01)}


# create the storage path
path = "gin_random"
os.mkdir(path)

# shell file
shell_line = 'timeout 240m python rlax_agent_session.py \
--hanabi_game_type="Hanabi-Small-CardKnowledge" \
--output_dir="output/config_{0}/" \
--agent_config_path="{1}/config_{0}.gin" \
--self_play\n'

# grid search

# # base file
# counter = 0
# generate_file(path, generate_file_content(params), counter)
# shell_content = shell_line.format(counter)
# 
# # apply changes
# for key, values in param_changes.items():
#     
#     new_params = deepcopy(params)
#     
#     for value in values:
#         counter += 1
#         new_params[key] = value
#         generate_file(path, 
#                       generate_file_content(new_params), 
#                       counter)
#         shell_content += shell_line.format(counter)

n_experiments = 30
shell_content = ""
for counter in range(n_experiments):
    
    exp_param = {}
    
    for key, values in param_random.items():
        exp_param[key] = choice(values)
        
    generate_file(path, 
                  generate_file_content(exp_param), 
                  counter)
    shell_content += shell_line.format(counter, path)
    
    if counter == 0:
        df_params = pd.DataFrame.from_dict([exp_param])
        
    else:
        df_params = df_params.append(exp_param, ignore_index=True)
        
# print the parameters
df_params["counter"] = range(30)
df_params = df_params.drop(labels=["train_batch_size", "layers", "use_double_q", "use_priority"],
               axis=1)
print(df_params)
fig = px.parallel_coordinates(df_params, color="counter")
plot(fig)

             
# create shell file
fname = "schedule_random.sh"
f = open(fname, "x")
f.write(shell_content)
f.close()


        
    
    


import random
import time

import yaml
from agent import Agent
import matplotlib.pyplot as plt
import os
from plot_graphs import PlotGraph
from heatmap_maker import HeatmapMaker


base_dir = os.path.dirname(os.path.abspath(__file__))
# Load the configuration file, check the config.yaml file for more information and to change to your needs
with open(f'{base_dir}/config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)



def print_warning(message):
    print(f"\033[93m{message}\033[0m")


def print_error(message):
    print(f"\033[91m{message}\033[0m")


def format_value(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def switch_color(robot_list, color_list, mj_color_idx):
    maj_color = color_list[mj_color_idx]
    num_majority = int(len(robot_list) * (config['majority_color_ratio']))
    # Assign the majority color to the specified percentage of robots
    for idx in range(num_majority):
        robot_list[idx].set_current_colour(maj_color)
    remaining_colors = [color_list[c_idx] for c_idx in range(len(color_list)) if c_idx != mj_color_idx]
    # Assign random colors to the rest of the robots
    for idx in range(num_majority, len(robot_list)):
        robot_list[idx].set_current_colour(random.choice(remaining_colors))


def get_color_counts(robot_list, colors_list):
    color_count = {}
    for c in colors_list:
        color_count[c] = 0
    for robot in robot_list:
        color_count[robot.get_color_opinion()] += 1
    cc_sum = 0
    for cc in color_count:
        cc_sum += color_count[cc]
    if cc_sum != len(robot_list):
        print_error('Error: Agent counts and color counts do not match!')
    return color_count


def evaluate_swarm_quality(agent_list, maj_color):
    correct_agent_count = 0
    for ag in agent_list:
        if ag.get_color_opinion() == maj_color:
            correct_agent_count += 1
    return correct_agent_count / len(agent_list)


def find_timesteps_to_reach_correct_opinion(srwm_quality, switch_timestep, threshold=config['majority_color_ratio']):
    sq_list = []
    threshold_reached = False
    timestep_iter = 0
    for i in range(len(srwm_quality)):
        if i % switch_timestep == 0:
            timestep_iter = 0
            threshold_reached = False
        if srwm_quality[i] >= threshold and threshold_reached is False:
            # print(f"i = {i}, srwm_quality = {srwm_quality[i]}, threshold = {threshold}")
            sq_list.append(timestep_iter)
            threshold_reached = True
        timestep_iter += 1
    avg_time = 0
    if len(sq_list) <= 1:
        avg_time = 0.0
        sq_list = []
    else:
        sq_list = sq_list[1:]
        avg_time = sum(sq_list) / len(sq_list)
    # print(f'sq_list: {sq_list}')
    return sq_list, avg_time


def create_plots(swrm_q, all_col_dict_list):
    plt.plot(swrm_q)
    plt.ylabel('Swarm Quality')
    plt.xlabel('Timesteps')
    plt.title(f"Ratio of Swarm on the Majority Color. ns={config['sensor_noise']}, "
              f"nc={config['communication_noise']}, w={config['personal_info_weight']}")
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    plt.show()
    # Extract the time series for each color
    time_series = {}
    for timestep, timestep_data in enumerate(all_col_dict_list):
        for color_id, value in timestep_data.items():
            if color_id not in time_series:
                time_series[color_id] = []
            time_series[color_id].append(value)
    # Define a color map
    color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange'}
    # Plotting
    for color_id, values in time_series.items():
        plt.plot(values, label=f'Color {color_id}', color=color_map[color_id])
    plt.xlabel('Timesteps')
    plt.ylabel('Number of Robots in Favour')
    plt.ylim(0, config['agent_count'])
    plt.title(f"Evolution of Each Color Over Time. ns={config['sensor_noise']}, "
              f"nc={config['communication_noise']}, w={config['personal_info_weight']}")
    # plt.legend()
    plt.grid(True)
    plt.show()


def run_simulations_all(run_counter, piw, sn, cn, swarm_size, M, env_option):
    # print(f'Running simulations for {run_counter} times')
    agent_list = list()
    colors = list()
    for i in range(env_option):
        colors.append(i)
    maj_color_idx = 0
    # Create all agents
    informed = True
    # for i in range(config['agent_count']):
    #     agent_list.append(Agent(config, informed, piw))
    #     if informed and len(agent_list) >= config['agent_count'] * config['informed_ratio']:
    #         # print(f"{i} agent informed")
    #         informed = False
    for i in range(swarm_size):
        agent_list.append(Agent(config, informed, piw))
        if informed and len(agent_list) >= swarm_size * config['informed_ratio']:
            # print(f"{i} agent informed")
            informed = False
    # print(f"Informed Count = {len([a for a in agent_list if a.informed])}")
    # Assign neighbours, no movement this can be static
    for a in agent_list:
        other_agents = [a_o for a_o in agent_list if a_o.get_id() != a.get_id()]
        a.assign_neighbours(random.sample(other_agents, M))
    # Assign color at start
    switch_color(agent_list, colors, maj_color_idx)
    # Time to run the simulation
    swarm_quality = []
    agent_iter = 0
    all_color_dict_list = list()
    number_of_color_switches = 0
    # print(f"Debug: max_ts:{config['max_timestep']}, cct:{config['color_change_timestep']}, sz: {swarm_size}, calc_rt: {int(config['max_timestep'] * (swarm_size/100))}, calc_st:{int(config['color_change_timestep'] * (swarm_size/100))}")
    max_runtime = int(config['max_timestep'] * (swarm_size/100))
    switch_timestep = int(config['color_change_timestep'] * (swarm_size/100))
    # print(f"Switching timestep {switch_timestep}, max_timestep {max_runtime}")
    # Asynchronous Update
    for t in range(1, max_runtime + 1):
        agent = agent_list[agent_iter]

        # Update the current agents neighbourhood
        other_agents = [a_o for a_o in agent_list if a_o.get_id() != agent.get_id()]
        agent.assign_neighbours(random.sample(other_agents, M))

        agent.detect_color(colors, maj_color_idx, sn)
        #TODO: Change made here for runner
        agent.majority_vote(colors, cn)
        agent.update_color(colors)
        agent_iter += 1
        if agent_iter == len(agent_list): agent_iter = 0
        swarm_quality.append(evaluate_swarm_quality(agent_list, colors[maj_color_idx]))
        all_color_dict_list.append(get_color_counts(agent_list, colors))
        # For non switching, accuracy measurement, still add 1 switch to get rid of the initial conditions, add and number_of_color_switches<1
        if t % switch_timestep == 0:
            # print("#######------------------#######")
            # print(f"Switch Timestep: {t} Prev Majority Color: {colors[maj_color_idx]}")
            # print(f"Swarm Quality before switch: {swarm_quality}")

            # print(f"Timestep: {t}, Majority Color: {colors[maj_color_idx]}")
            # print(f"All Robots commited colors: {all_color_dict_list[len(all_color_dict_list) - 1]}")
            maj_color_idx = (maj_color_idx + 1) % len(colors)
            # print(f"New Majority Color: {colors[maj_color_idx]}")
            number_of_color_switches += 1
            # for agnt in agent_list:
            #     print(agnt)

    # for a in agent_list:
    #     print(a)
    # print("All_color_dict", all_color_dict_list)
    # print(f"Swarm Quality: {swarm_quality}")
    swarm_switch_time_list, avg_switch_timestep = find_timesteps_to_reach_correct_opinion(swarm_quality, 0.75)
    # create_plots(swarm_quality, all_color_dict_list)
    #TODO: Change made here for runner
    nested_folder = (f"run_mt{max_runtime}_ac{swarm_size}_mcr_{config['majority_color_ratio']}_"
                     f"ir{config['informed_ratio']}_piw{piw}_sn{sn}_"
                     f"cn{cn}_cct{switch_timestep}")
    # full_path = os.path.join('output_data', nested_folder)
    
    full_path = os.path.join(base_dir, 'output_data', f"{swarm_size}_robots_{M}_neighbours", f"env_options_{env_op}", f"cn{cn_main}_fixed", nested_folder)
    if not os.path.exists(full_path):
        # Create the directory
        os.makedirs(full_path)
        print(f'Directory {full_path} created.')
    filename = f"output_run{run_counter}.txt"
    header = ["timestep", "swarm_quality"]
    for color in colors:
        header.append(f"color_{color}")

    with open(f'{full_path}/' + filename, "w") as file:
        # Write the header
        file.write("\t".join(header) + "\n")

        for i in range(len(swarm_quality)):
            file.write(f"{i}\t{format_value(swarm_quality[i])}")
            for key in all_color_dict_list[i]:
                file.write(f"\t{all_color_dict_list[i][key]}")
            file.write("\n")
    file.close()
    with open(f'{full_path}/timestep_data{run_counter}.txt', 'w') as file:
        file.write(f"total_switches: {number_of_color_switches}\n")
        file.write(f"avg_switch_timestep: {avg_switch_timestep}\n")
        file.write("switch_timesteps: ")
        for sqt in swarm_switch_time_list:
            file.write(f"{sqt}\t")
        file.write("\n")

    file.close()

    # Write YAML content to a text file
    with open(f'{full_path}/metadata.txt', 'w') as text_file:
        text_file.write(f"max_timestep: {max_runtime}\n")
        text_file.write(f"agent_count: {swarm_size}\n")
        text_file.write(f"neighbourhood_size: {M}\n")
        text_file.write(f"color_count: {env_option}\n")
        text_file.write(f"majority_color_ratio: {config['majority_color_ratio']}\n")
        text_file.write(f"informed_ratio: {config['informed_ratio']}\n")
        text_file.write(f"personal_info_weight: {piw}\n")
        text_file.write(f"sensor_noise: {sn}\n")
        text_file.write(f"communication_noise: {cn}\n")
        text_file.write(f"color_change_timestep: {switch_timestep}\n")
        text_file.write(f"total_simulation_runs: {config['total_simulation_runs']}")
    text_file.close()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    start_time = time.time()
    # values from 0.0 to 1.0 in steps of 0.2

    sn_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # cn_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    piw_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    # swarm_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # sn_steps = [0.0, 0.2]
    cn_steps = [0.2]
    # piw_steps = [0.4,0.6]
    swarm_sizes = [100]
    env_options = [5]

    # for each cn, create a folder. for each n (env. option) create a folder.

    total_runs = (len(cn_steps) * len(piw_steps) * len(sn_steps) * len(swarm_sizes)* len(env_options))
    # Nested loops to iterate over all combinations
    for s in swarm_sizes:
        neighbourhood_size = list(range(5, s, 5)) #TODO: Change this
        neighbourhood_size.append(s-1)
        total_runs *= len(env_options)
        print(f"M= {neighbourhood_size}")
        for env_op in env_options:
            for m in neighbourhood_size:
                for cn_main in cn_steps: #switch for loop to this place on the param that you want to fix, also change the main_file_path in plot_graph.py
                    for sn_main in sn_steps:
                        for piw_main in piw_steps:
                            total_runs -= 1
                            print(f"Running N= {s} piw: {piw_main}, sn: {sn_main} cn:{cn_main}")
                            for i in range(config['total_simulation_runs']):
                                # this will run the sim 20 times for a particular piw, sn, cn, s and m
                                run_simulations_all(i, piw_main, sn_main, cn_main, s, m, env_op)

                            print("Saving Plots")
                            max_runtime_r = int(config['max_timestep'] * (s / 100))
                            switch_timestep_r = int(config['color_change_timestep'] * (s / 100))
                            plot_the_graph = PlotGraph(max_runtime_r, s,
                                                    config['majority_color_ratio'],
                                                    config['informed_ratio'], piw_main, sn_main, cn_main,
                                                    switch_timestep_r,m, env_op)
                            plot_the_graph.execute_and_plot()
                            print(f"Now {total_runs} runs left")
                            print("----")
                
                    ht = HeatmapMaker(os.path.join(base_dir, "output_data", f"{s}_robots_{m}_neighbours", f"env_options_{env_op}", f"cn{cn_main}_fixed",""), s, m)
                    ht.execute_heatmaps_creation()
    print("Time taken (s):", round((time.time() - start_time), 2))

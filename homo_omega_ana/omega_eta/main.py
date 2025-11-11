import random
import time

import yaml
from agent import Agent
import matplotlib.pyplot as plt
import os
from plot_graphs import PlotGraph
from heatmap_maker import HeatmapMaker

# Load the configuration file, check the config.yaml file for more information and to change to your needs
with open('config.yaml', 'r') as file:
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


def run_simulations_all(run_counter, piw, sn, cn, swarm_size):
    # print(f'Running simulations for {run_counter} times')
    agent_list = list()
    colors = list()
    for i in range(config['color_count']):
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
        a.assign_neighbours(random.sample(other_agents, config['neighbourhood_size']))
    # Assign color at start
    switch_color(agent_list, colors, maj_color_idx)
    # Time to run the simulation
    swarm_quality = []
    agent_iter = 0
    all_color_dict_list = list()
    number_of_color_switches = 0
    max_runtime = int(config['max_timestep'] * (swarm_size/100))
    switch_timestep = int(config['color_change_timestep'] * (swarm_size/100))
    # print(f"Switching timestep {switch_timestep}, max_timestep {max_runtime}")
    # Asynchronous Update
    for t in range(1, max_runtime + 1):
        agent = agent_list[agent_iter]

        # Update the current agents neighbourhood
        other_agents = [a_o for a_o in agent_list if a_o.get_id() != agent.get_id()]
        agent.assign_neighbours(random.sample(other_agents, config['neighbourhood_size']))

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
    full_path = os.path.join('output_data', f"{swarm_size}_robots_{config['neighbourhood_size']}_neighbours", nested_folder)
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
        text_file.write(f"neighbourhood_size: {config['neighbourhood_size']}\n")
        text_file.write(f"color_count: {config['color_count']}\n")
        text_file.write(f"majority_color_ratio: {config['majority_color_ratio']}\n")
        text_file.write(f"informed_ratio: {config['informed_ratio']}\n")
        text_file.write(f"personal_info_weight: {piw}\n")
        text_file.write(f"sensor_noise: {sn}\n")
        text_file.write(f"communication_noise: {cn}\n")
        text_file.write(f"color_change_timestep: {switch_timestep}\n")
        text_file.write(f"total_simulation_runs: {config['total_simulation_runs']}")
    text_file.close()


def run_simulations(run_counter):
    print(f'Running simulations for {run_counter} times')
    agent_list = list()
    colors = list()
    for i in range(config['color_count']):
        colors.append(i)
    maj_color_idx = 0
    # Create all agents
    informed = True
    for i in range(config['agent_count']):
        agent_list.append(Agent(config, informed, config['personal_info_weight']))
        if len(agent_list) >= config['agent_count'] * config['informed_ratio']:
            informed = False

    # Assign neighbours, no movement this can be static
    for a in agent_list:
        other_agents = [a_o for a_o in agent_list if a_o.get_id() != a.get_id()]
        a.assign_neighbours(random.sample(other_agents, config['neighbourhood_size']))
    # Assign color at start
    switch_color(agent_list, colors, maj_color_idx)
    # Time to run the simulation
    swarm_quality = []
    agent_iter = 0
    all_color_dict_list = list()
    number_of_color_switches = 0
    # Asynchronous Update
    for t in range(1, config['max_timestep'] + 1):
        agent = agent_list[agent_iter]
        agent.detect_color(colors, maj_color_idx, config['sensor_noise'])
        agent.majority_vote(colors, config['communication_noise'])
        agent.update_color(colors)
        agent_iter += 1
        if agent_iter == len(agent_list): agent_iter = 0
        swarm_quality.append(evaluate_swarm_quality(agent_list, colors[maj_color_idx]))
        all_color_dict_list.append(get_color_counts(agent_list, colors))
        if t % config['color_change_timestep'] == 0:
            # print("#######------------------#######")
            # print(f"Prev Majority Color: {colors[maj_color_idx]}")
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
    nested_folder = (f"run_mt{config['max_timestep']}_ac{config['agent_count']}_mcr_{config['majority_color_ratio']}_"
                     f"ir{config['informed_ratio']}_piw{config['personal_info_weight']}_sn{config['sensor_noise']}_"
                     f"cn{config['communication_noise']}_cct{config['color_change_timestep']}")
    full_path = os.path.join('output_data', nested_folder)
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
    with open(f'{full_path}/timestep_data{run_counter}.txt', 'w') as file:
        file.write(f"total_switches: {number_of_color_switches}\n")
        file.write(f"avg_switch_timestep: {avg_switch_timestep}\n")
        file.write("switch_timesteps: ")
        for sqt in swarm_switch_time_list:
            file.write(f"{sqt}\t")
        file.write("\n")

    # Read YAML file
    with open('config.yaml', 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)

    # Write YAML content to a text file
    with open(f'{full_path}/metadata.txt', 'w') as text_file:
        yaml.dump(config_data, text_file, default_flow_style=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # for i in range(config['total_simulation_runs']):
    #     run_simulations(i)

    # values from 0.0 to 1.0 in steps of 0.2
    #piw_steps = [0.6]
    #sn_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cn_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sn_steps = [0.55]
    #cn_steps = [0.2]
    piw_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    #piw_steps = [0.6]
    #sn_steps = [0.4]
    # cn_steps = [0.4,0.6]
    start_time = time.time()
    #swarm_sizes = [40,60,80,100,120,140,160,180,200]
    swarm_sizes = [20]
    total_runs = (len(cn_steps) * len(piw_steps) * len(sn_steps) * len(swarm_sizes))
    # Nested loops to iterate over all combinations
    for s in swarm_sizes:
        for cn_main in cn_steps:
            for sn_main in sn_steps:
                for piw_main in piw_steps:
                    total_runs -= 1
                    print(f"Running N= {s} piw: {piw_main}, sn: {sn_main} cn:{cn_main}")
                    for i in range(config['total_simulation_runs']):
                        run_simulations_all(i, piw_main, sn_main, cn_main, s)

                    print("Saving Plots")
                    max_runtime_r = int(config['max_timestep'] * (s / 100))
                    switch_timestep_r = int(config['color_change_timestep'] * (s / 100))
                    plot_the_graph = PlotGraph(max_runtime_r, s,
                                               config['majority_color_ratio'],
                                               config['informed_ratio'], piw_main, sn_main, cn_main,
                                               switch_timestep_r,config['neighbourhood_size'])
                    plot_the_graph.execute_and_plot()
                    print(f"Now {total_runs} runs left")
                    print("----")

        ht = HeatmapMaker(os.path.join("output_data", f"{s}_robots_{config['neighbourhood_size']}_neighbours",""), s, config['neighbourhood_size'])
        ht.execute_heatmaps_creation()
    print("Time taken (s):", round((time.time() - start_time), 2))

import os.path

import pandas as pd
import matplotlib.pyplot as plt
import glob
import yaml


class PlotGraph:
    def __init__(self, max_timestep, agent_count, majority_color_ratio, informed_ratio, personal_info_weight
                 , sensor_noise, communication_noise, color_change_timestep, neighbourhood_size):
        self.max_timestep = max_timestep
        self.agent_count = agent_count
        self.majority_color_ratio = majority_color_ratio
        self.informed_ratio = informed_ratio
        self.personal_info_weight = personal_info_weight
        self.sensor_noise = sensor_noise
        self.communication_noise = communication_noise
        self.color_change_timestep = color_change_timestep
        self.neighbourhood_size = neighbourhood_size

        self.nested_folder = (f"run_mt{self.max_timestep}_ac{self.agent_count}_mcr_{self.majority_color_ratio}_"
                              f"ir{self.informed_ratio}_piw{self.personal_info_weight}_sn{self.sensor_noise}_"
                              f"cn{self.communication_noise}_cct{self.color_change_timestep}")

        self.main_file_path = os.path.join("output_data", f"{agent_count}_robots_{self.neighbourhood_size}_neighbours",self.nested_folder)

    # Function to read and parse the data
    def read_data(self, file_path):
        return pd.read_csv(file_path, sep='\t')

    def print_timestep_data(self):
        # Function to read and parse data from the timestep_data file
        def read_timestep_data(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()

                total_switches = int(lines[0].split(':')[1].strip())
                avg_switch_timestep = float(lines[1].split(':')[1].strip())
                switch_timesteps = list(map(int, lines[2].split(':')[1].strip().split()))
                # print(f"Switch Timestamps: {switch_timesteps}")

            return total_switches, avg_switch_timestep, switch_timesteps

        # Read data from all relevant files
        file_paths = glob.glob(f'{self.main_file_path}/timestep_data*.txt')
        total_switches_values = []
        avg_switch_timesteps = []
        switch_timesteps_lists = []

        for file in file_paths:
            total_switches, avg_switch_timestep, switch_timesteps = read_timestep_data(file)
            total_switches_values.append(total_switches)
            avg_switch_timesteps.append(avg_switch_timestep)
            switch_timesteps_lists.append(len(switch_timesteps))

        # Calculate the average avg_switch_timestep
        average_avg_switch_timestep = sum(avg_switch_timesteps) / len(avg_switch_timesteps)
        print(f"Switch Timesteps List {switch_timesteps_lists}")
        # Calculate the average length of switch_timesteps lists
        average_length_switch_timesteps = sum(switch_timesteps_lists) / len(switch_timesteps_lists)

        # Get the total_switches value (assuming it remains the same in all files)
        total_switches_value = total_switches_values[0] if total_switches_values else None

        # Print the results
        print(f'The average of avg_switch_timestep values is: {average_avg_switch_timestep}')
        print(f'The average length of switch_timesteps lists is: {average_length_switch_timesteps}')
        print(f'The total_switches value is: {total_switches_value - 1}')

    def execute_and_plot(self):
        # Read data from all files
        file_paths = glob.glob(f'{self.main_file_path}/output_run*.txt')
        data_frames = [self.read_data(file) for file in file_paths]

        # self.print_timestep_data()

        # Merge data frames on 'timestep'
        merged_data = pd.concat(data_frames).groupby('timestep').mean()


        # # Select the last 1000 timesteps
        # last_1000_timesteps = merged_data.tail(1000)
        #
        # # Calculate the average swarm_quality for the last 1000 timesteps
        # average_swarm_quality = last_1000_timesteps['swarm_quality'].mean()
        #
        # # Print the result
        # print(f"Average swarm_quality for the last 1000 timesteps: {average_swarm_quality}")
        #
        # with open(f"output_data/average_quality_ir{self.informed_ratio}_w{self.personal_info_weight}_sn{self.sensor_noise}"
        #           f"_cn{self.communication_noise}.txt", "w") as file:
        #     file.write(f"ir:{self.informed_ratio}\n")
        #     file.write(f"w:{self.personal_info_weight}\n")
        #     file.write(f"sn:{self.sensor_noise}\n")
        #     file.write(f"cn:{self.communication_noise}\n")
        #     file.write(f"avg_swarm_quality:{average_swarm_quality}\n")
        # file.close()
        #TODO: Filtering out first 1000 timestep, remove this later.
        merged_data = merged_data.iloc[1000:]
        # Plot averaged swarm_quality over timesteps
        plt.figure(figsize=(10, 5))
        plt.plot(merged_data.index, merged_data['swarm_quality'], label='Swarm Quality')
        plt.ylim(0.0, 1.0)
        plt.xlabel('Timestep')
        plt.ylabel('Swarm Quality')
        plt.title(
            f"Averaged Swarm Quality ir:{self.informed_ratio} w:{self.personal_info_weight} sn:{self.sensor_noise}"
            f" cn:{self.communication_noise}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.main_file_path}/swarm_quality_graphs.png')
        # plt.show()
        plt.close()

        # Plot averaged color_x values over timesteps
        plt.figure(figsize=(10, 5))
        plt.ylim(0, self.agent_count)
        color_map = {'color_0': 'red', 'color_1': 'blue', 'color_2': 'green', 'color_3': 'purple', 'color_4': 'orange',
                     'color_5': 'aqua', 'color_6': 'teal', 'color_7': 'indigo', 'color_8': 'crimson', 'color_9': 'olive',
                     'color_10': 'rosybrown', 'color_11': 'lime', 'color_12': 'papayawhip', 'color_13': 'tan', 'color_14': 'thistle',
                     'color_15': 'saddlebrown', 'color_16': 'maroon', 'color_17': 'darkseagreen'}
        for color_col in merged_data.columns:
            if 'color' in color_col:
                plt.plot(merged_data.index, merged_data[color_col], label=color_col, color=color_map[color_col])

        plt.xlabel('Timestep')
        plt.ylabel('Number of Robots in Favour')
        plt.title(
            f"Averaged Color Values ir:{self.informed_ratio} w:{self.personal_info_weight} sn:{self.sensor_noise}"
            f" cn:{self.communication_noise}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.main_file_path}/color_count_graphs.png')
        plt.close()
        # plt.show()

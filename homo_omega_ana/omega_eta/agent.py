import numpy as np
import random
import itertools


class Agent:
    id_iter = itertools.count()

    def __init__(self, config, informed, personal_info_weight):
        self.id = next(self.id_iter)
        self.informed = informed
        self.config = config
        self.neighbours = list()
        self.personal_info_weight = personal_info_weight
        self.current_colour = -1
        self.personal_info_counts = dict()
        self.social_info_counts = dict()
        self.sensor_color = -1

    def get_id(self):
        return self.id

    def assign_neighbours(self, neighbours):
        # print(f"Id {self.id}. Neighbours: {[n.get_id() for n in neighbours]}")
        self.neighbours = neighbours

    def set_current_colour(self, current_colour):
        self.current_colour = current_colour

    def get_color_opinion(self):
        return self.current_colour

    def detect_color(self, color_list, m_color_idx, noise=0.0):
        # When it's not noise, just keep the same color
        # so when it's a noisy reading
        # No need of count. Just personal opinion is just your opinion of color, no need of count.
        self.sensor_color = color_list[m_color_idx]
        if random.random() < noise:
            other_colors_idx = [c_idx for c_idx in range(len(color_list)) if
                                color_list[c_idx] != color_list[m_color_idx]]
            m_color_idx = np.random.choice(other_colors_idx)
            self.sensor_color = color_list[m_color_idx]

    def majority_vote(self, color_list, noise=0.0):
        self.social_info_counts.clear()
        # We only do majority voting
        for n in self.neighbours:
            neighbour_color = n.get_color_opinion()
            # We have a noisy reading
            if random.random() < noise:
                other_colors = [c for c in color_list if c != neighbour_color]
                neighbour_color = random.choice(other_colors)
            if neighbour_color in self.social_info_counts:
                self.social_info_counts[neighbour_color] += 1
            else:
                self.social_info_counts[neighbour_color] = 1
        # Return key with max value
        # return max(self.social_info_counts, key=self.social_info_counts.get)

    def update_color(self, color_list):
        # if self.id == 0:
        #     print("---------------")
        #     print("Update color count")
        m = len(self.neighbours)
        decision_dict = dict()
        for color in color_list:
            if self.informed:
                mi = self.social_info_counts.get(color, 0)
                # Personal Opinion weight should be higher than 1 means saying you weigh your personal opinion
                # more than the social opinion and vice versa
                # if self.id == 0:
                #     print(f"mi: {mi}, m= {m} and decionsmaker = {color == self.sensor_color}")
                decision_dict[color] = mi + (self.personal_info_weight * m *
                                             (1 if color == self.sensor_color else 0))
                # mi = mi * (0 if color == self.current_colour else 1)
                # decision_dict[color] = (self.personal_info_counts.get(color, 0) + mi
                #                         + ((self.config['personal_info_weight'] * m *
                #                             (1 if color == self.sensor_color else 0))))

        # Decision-making time
        if self.informed:
            self.current_colour = self.get_highest_in_dict(decision_dict)
        else:
            self.current_colour = self.get_highest_in_dict(self.social_info_counts)

        # if self.id == 0:
        #     print(f"Current Opinion color is {self.current_colour}, Current Sensor color is {self.sensor_color}")
        #     print("SI= ", self.social_info_counts)
        #     print(f"Decision Dict:{decision_dict}")
        #     print("----")
        # self.social_info_counts.clear()

    def get_highest_in_dict(self, the_dict):
        max_count = max(the_dict.values())
        keys_with_max_count = [key for key, value in the_dict.items() if value == max_count]

        if len(keys_with_max_count) > 1:
            return random.choice(keys_with_max_count)
        else:
            return keys_with_max_count[0]

    def agent_reset(self):
        self.current_colour = -1
        self.personal_info_counts.clear()
        self.social_info_counts.clear()

    def __str__(self):
        return f"ID: {self.id}, Informed: {self.informed}, Current Color: {self.current_colour}"

mport pickle
import numpy as np
import sys
import pandas as pd
import cityflow as engine
from copy import deepcopy
import envs.env_config
import copy
import argparse
import os
import shutil
import json
import time

class RoadNet:

    def __init__(self, roadnet_file):
        self.roadnet_dict = json.load(open(roadnet_file,"r"))
        self.net_edge_dict = {}
        self.net_node_dict = {}
        self.net_lane_dict = {}

        self.generate_node_dict()
        self.generate_edge_dict()
        self.generate_lane_dict()

    def generate_node_dict(self):
        '''
        node dict has key as node id, value could be the dict of input nodes and output nodes
        :return:
        '''

        for node_dict in self.roadnet_dict['intersections']:
            node_id = node_dict['id']
            road_links = node_dict['roads']
            input_nodes = []
            output_nodes = []
            input_edges = []
            output_edges = {}
            for road_link_id in road_links:
                road_link_dict = self._get_road_dict(road_link_id)
                if road_link_dict['startIntersection'] == node_id:
                    end_node = road_link_dict['endIntersection']
                    output_nodes.append(end_node)
                    # todo add output edges
                elif road_link_dict['endIntersection'] == node_id:
                    input_edges.append(road_link_id)
                    start_node = road_link_dict['startIntersection']
                    input_nodes.append(start_node)
                    output_edges[road_link_id] = set()
                    pass

            # update roadlinks
            actual_roadlinks = node_dict['roadLinks']
            for actual_roadlink in actual_roadlinks:
                output_edges[actual_roadlink['startRoad']].add(actual_roadlink['endRoad'])

            net_node = {
                'node_id': node_id,
                'input_nodes': list(set(input_nodes)),
                'input_edges': list(set(input_edges)),
                'output_nodes': list(set(output_nodes)),
                'output_edges': output_edges# should be a dict, with key as an input edge, value as output edges
            }
            if node_id not in self.net_node_dict.keys():
                self.net_node_dict[node_id] = net_node

    def _get_road_dict(self, road_id):
        for item in self.roadnet_dict['roads']:
            if item['id'] == road_id:
                return item
        print("Cannot find the road id {0}".format(road_id))
        sys.exit(-1)
        # return None

    def generate_edge_dict(self):
        '''
        edge dict has key as edge id, value could be the dict of input edges and output edges
        :return:
        '''
        for edge_dict in self.roadnet_dict['roads']:
            edge_id = edge_dict['id']
            input_node = edge_dict['startIntersection']
            output_node = edge_dict['endIntersection']

            net_edge = {
                'edge_id': edge_id,
                'input_node': input_node,
                'output_node': output_node,
                'input_edges': self.net_node_dict[input_node]['input_edges'],
                'output_edges': self.net_node_dict[output_node]['output_edges'][edge_id],

            }
            if edge_id not in self.net_edge_dict.keys():
                self.net_edge_dict[edge_id] = net_edge

    def generate_lane_dict(self):
        lane_dict = {}
        for node_dict in self.roadnet_dict['intersections']:
            for road_link in node_dict["roadLinks"]:
                lane_links = road_link["laneLinks"]
                start_road = road_link["startRoad"]
                end_road = road_link["endRoad"]
                for lane_link in lane_links:
                    start_lane = start_road + "_" + str(lane_link['startLaneIndex'])
                    end_lane = end_road + "_" +str(lane_link["endLaneIndex"])
                    if start_lane not in lane_dict:
                        lane_dict[start_lane] = {
                            "output_lanes": [end_lane],
                            "input_lanes": []
                        }
                    else:
                        lane_dict[start_lane]["output_lanes"].append(end_lane)
                    if end_lane not in lane_dict:
                        lane_dict[end_lane] = {
                            "output_lanes": [],
                            "input_lanes": [start_lane]
                        }
                    else:
                        lane_dict[end_lane]["input_lanes"].append(start_lane)

        self.net_lane_dict = lane_dict

    def hasEdge(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return True
        else:
            return False

    def getEdge(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return edge_id
        else:
            return None

    def getOutgoing(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return self.net_edge_dict[edge_id]['output_edges']
        else:
            return []


class Intersection:
    DIC_PHASE_MAP = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        -1: 0
    }
    def __init__(self, inter_id, dic_traffic_env_conf, eng, light_id_dict,path_to_log,max_waiting_time):
        self.inter_id = inter_id

        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])

        self.eng = eng


        self.max_waiting_time = max_waiting_time

        self.controlled_model  = dic_traffic_env_conf['MODEL_NAME']
        self.path_to_log = path_to_log

        # =====  intersection settings =====
        self.list_approachs = ["W", "E", "N", "S"]
        self.dic_approach_to_node = {"W": 2, "E": 0, "S": 3, "N": 1}
        # self.dic_entering_approach_to_edge = {
        #    approach: "road{0}_{1}_{2}".format(self.dic_approach_to_node[approach], light_id) for approach in self.list_approachs}

        self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])}
        self.dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])})
        self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1)})
        self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)})

        self.dic_exiting_approach_to_edge = {
            approach: "road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach]) for
        approach in self.list_approachs}
        self.dic_entering_approach_lanes = {"W": [0], "E": [0], "S": [0], "N": [0]}
        self.dic_exiting_approach_lanes = {"W": [0], "E": [0], "S": [0], "N": [0]}

        # grid settings
        self.length_lane = 300
        self.length_terminal = 50
        self.length_grid = 5
        self.num_grid = int(self.length_lane // self.length_grid)

        self.list_phases = dic_traffic_env_conf["PHASE"][dic_traffic_env_conf['SIMULATOR_TYPE']]


        # generate all lanes
        self.list_entering_lanes = []      # 当前交叉口的进路
        for approach in self.list_approachs:
            self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + '_' + str(i) for i in
                                         range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]
        self.list_exiting_lanes = []
        for approach in self.list_approachs:
            self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + '_' + str(i) for i in
                                        range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.adjacency_row = light_id_dict['adjacency_row']
        self.neighbor_ENWS = light_id_dict['neighbor_ENWS']
        self.neighbor_lanes_ENWS = light_id_dict['entering_lane_ENWS']

        def _get_top_k_lane(lane_id_list, top_k_input):
            top_k_lane_indexes = []
            for i in range(top_k_input):
                lane_id = lane_id_list[i] if i < len(lane_id_list) else None
                top_k_lane_indexes.append(lane_id)
            return top_k_lane_indexes

        self._adjacency_row_lanes = {}
        # _adjacency_row_lanes is the lane id, not index
        for lane_id in self.list_entering_lanes:
            if lane_id in light_id_dict['adjacency_matrix_lane']:
                self._adjacency_row_lanes[lane_id] = light_id_dict['adjacency_matrix_lane'][lane_id]
            else:
                self._adjacency_row_lanes[lane_id] = [_get_top_k_lane([], self.dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"]),
                                                 _get_top_k_lane([], self.dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"])]
        # order is the entering lane order, each element is list of two lists

        self.adjacency_row_lane_id_local = {}
        for index, lane_id in enumerate(self.list_entering_lanes):
            self.adjacency_row_lane_id_local[lane_id] = index

        # self.vehicle_waiting_time = {}  # key: vehicle_id, value: the waiting time of this vehicle since last halt.

        # previous & current
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}

        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step = []
        self.list_lane_vehicle_current_step = []

        self.lane_first_vehicle_current_step = {}
        self.lane_first_vehicle_previous_step = {}
        self.lane_first_vehicle_waiting_time = {}

        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
        path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
        df = [self.get_current_time(), self.current_phase_index]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode='a', header=False, index=False)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

        self.dic_vehicle_min_speed = {}  # this second
        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second

        #### fwq
        self.previous_time = self.get_current_time()

    def build_adjacency_row_lane(self, lane_id_to_global_index_dict):
        self.adjacency_row_lanes = [] # order is the entering lane order, each element is list of two lists
        for entering_lane_id in self.list_entering_lanes:
            _top_k_entering_lane, _top_k_leaving_lane = self._adjacency_row_lanes[entering_lane_id]
            top_k_entering_lane = []
            top_k_leaving_lane = []
            for lane_id in _top_k_entering_lane:
                top_k_entering_lane.append(lane_id_to_global_index_dict[lane_id] if lane_id is not None else -1)
            for lane_id in _top_k_leaving_lane:
                top_k_leaving_lane.append(lane_id_to_global_index_dict[lane_id]
                                          if (lane_id is not None) and (lane_id in lane_id_to_global_index_dict.keys())  # TODO leaving lanes of system will also have -1
                                          else -1)
            self.adjacency_row_lanes.append([top_k_entering_lane, top_k_leaving_lane])

    # set
    def set_signal(self, action, action_pattern, yellow_time, all_red_time):
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time: # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index) # if multi_phase, need more adjustment
                path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode='a', header=False, index=False)
                self.all_yellow_flag = False
            else:
                pass
        else:
            # determine phase
            if action_pattern == "switch": # switch by order
                if action == 0: # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1: # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases) # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set": # set to certain phase
                self.next_phase_to_set_index = self.DIC_PHASE_MAP[action[0]] # if multi_phase, need more adjustment

            # set phase
            if self.current_phase_index == self.next_phase_to_set_index: # the light phase keeps unchanged
                pass
            else: # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.eng.set_tl_phase(self.inter_name, 0) # !!! yellow, tmp
                path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode='a', header=False, index=False)
                #traci.trafficlights.setRedYellowGreenState(
                #    self.node_light, self.all_yellow_phase_str)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    # update inner measurements
    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index

        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step


    def update_current_measurements_map(self, simulator_state):
        ## need change, debug in seeing format
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []

            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)

            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]       #### 当前时间步的车辆为该交叉口进路上的车
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][lane]

        for lane in self.list_exiting_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][lane]

        self.dic_vehicle_speed_current_step = simulator_state['get_vehicle_speed']
        self.dic_vehicle_distance_current_step = simulator_state['get_vehicle_distance']

        # get vehicle list
        self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)

        list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))
        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []
        for l in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l

        self.lane_first_vehicle_waiting_time = self._get_lane_first_vehicle_waiting_time(self.list_entering_lanes)

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left_entering_lane)

        #### fwq
        self._update_waiting_time()    #### 当前时间步的车辆为该交叉口进路上的车

        # update vehicle minimum speed in history, # to be implemented
        #self._update_vehicle_min_speed()

        # update feature
        self._update_feature_map(simulator_state)

    def _update_leave_entering_approach_vehicle(self):

        list_entering_lane_vehicle_left = []

        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])
                current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[lane])

            list_entering_lane_vehicle_left.append(
                    list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
                )

        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicle_arrive):

        ts = self.get_current_time()
        # get dic vehicle enter leave time

        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                #### fwq
                self.dic_vehicle_arrive_leave_time[vehicle] = {"enter_time": ts, "leave_time": np.nan, "cost_time": 0, "waiting_time": 0}
            else:
                #print("vehicle: %s already exists in entering lane!"%vehicle)
                #sys.exit(-1)
                pass

    def _update_left_time(self, list_vehicle_left):

        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
                # print('1')
                self.dic_vehicle_arrive_leave_time[vehicle]["cost_time"] = ts - self.dic_vehicle_arrive_leave_time[vehicle]["enter_time"]

            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    #### fwq
    def _update_waiting_time(self):
        for lane in self.list_entering_lanes:
            if self.dic_lane_vehicle_current_step[lane]:
                vehs = list(set(self.dic_lane_vehicle_current_step[lane]) & set(self.dic_lane_vehicle_previous_step[lane]))  # 求上一刻在该路上且这一刻还在该路上的车
                waiting_veh = [veh for veh in vehs if self.dic_vehicle_speed_current_step[veh] < 0.1]
                for vehicle in waiting_veh:
                    try:
                        self.dic_vehicle_arrive_leave_time[vehicle]["waiting_time"] += 1
                    except KeyError:
                        print("vehicle not recorded when entering")
                        sys.exit(-1)

    def update_neighbor_info(self, neighbors, dic_feature):
        # print(dic_feature)
        none_dic_feature = deepcopy(dic_feature)
        for key in none_dic_feature.keys():
            if none_dic_feature[key] is not None:
                if "cur_phase" in key:
                    none_dic_feature[key] = [1] * len(none_dic_feature[key])
                else:
                    none_dic_feature[key] = [0] * len(none_dic_feature[key])
            else:
                none_dic_feature[key] = None
        for i in range(len(neighbors)):
            neighbor = neighbors[i]
            example_dic_feature = {}
            if neighbor is None:
                example_dic_feature["cur_phase_{0}".format(i)] = none_dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = none_dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = none_dic_feature["lane_num_vehicle"]
            else:
                example_dic_feature["cur_phase_{0}".format(i)] = neighbor.dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = neighbor.dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = neighbor.dic_feature["lane_num_vehicle"]
            dic_feature.update(example_dic_feature)
        return dic_feature

    def _update_feature_map(self, simulator_state):

        dic_feature = dict()

        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]

        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes)

        dic_feature["coming_vehicle"] = self._get_coming_vehicles(simulator_state)
        dic_feature["leaving_vehicle"] = self._get_leaving_vehicles(simulator_state)

        dic_feature["lane_num_vehicle_been_stopped_thres1"] = self._get_lane_num_vehicle_been_stopped(1, self.list_entering_lanes)
        dic_feature['lane_first_vehicle_waiting_time'] = list(self.lane_first_vehicle_waiting_time.values())

        dic_feature["adjacency_matrix"] = self._get_adjacency_row() # TODO this feature should be a dict? or list of lists

        dic_feature["adjacency_matrix_lane"] = self._get_adjacency_row_lane() #row: entering_lane # columns: [inputlanes, outputlanes]

        self.dic_feature = dic_feature

    # ================= calculate features from current observations ======================

    def _get_adjacency_row(self):
        return self.adjacency_row

    def _get_adjacency_row_lane(self):
        return self.adjacency_row_lanes

    def lane_position_mapper(self, lane_pos, bins):
        lane_pos_np = np.array(lane_pos)
        digitized = np.digitize(lane_pos_np, bins)
        position_counter = [len(lane_pos_np[digitized == i]) for i in range(1, len(bins))]
        return position_counter

    def _get_coming_vehicles(self, simulator_state):
        ## TODO f vehicle position   eng.get_vehicle_distance()  ||  eng.get_lane_vehicles()

        coming_distribution = []
        ## dimension = num_lane*3*num_list_entering_lanes

        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        ## TODO LANE LENGTH = 300
        bins = np.linspace(0, 300, 4).tolist()

        for lane in self.list_entering_lanes:
            coming_vehicle_position = []
            vehicle_position_lane = lane_vid_mapping_dict[lane]
            for vehicle in vehicle_position_lane:
                coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
            coming_distribution.extend(self.lane_position_mapper(coming_vehicle_position, bins))


        return coming_distribution

    def _get_leaving_vehicles(self, simulator_state):
        leaving_distribution = []
        ## dimension = num_lane*3*num_list_entering_lanes

        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        ## TODO LANE LENGTH = 300
        bins = np.linspace(0, 300, 4).tolist()

        for lane in self.list_exiting_lanes:
            coming_vehicle_position = []
            vehicle_position_lane = lane_vid_mapping_dict[lane]
            for vehicle in vehicle_position_lane:
                coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
            leaving_distribution.extend(self.lane_position_mapper(coming_vehicle_position, bins))

        return leaving_distribution

    def _get_lane_queue_length(self, list_lanes):
        '''
        queue length for each lane
        '''
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in list_lanes]



    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]


    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()

    def get_dic_vehicle_arrive_leave_time(self):

        return self.dic_vehicle_arrive_leave_time

    def _get_lane_first_vehicle_waiting_time(self, list_lanes):
        all_lane_vehicles = self.dic_lane_vehicle_current_step  # {'road_0_1_0_0': [], 'road_0_1_0_1': [],......
        all_vehicle_speed = self.dic_vehicle_speed_current_step  # {'flow_0_0': 11.111,......
        all_vehicle_distance = self.dic_vehicle_distance_current_step  # {'flow_0_0': 260.2716249999998,......
        # 获取每条道路第一辆等待的车
        for lane in list_lanes:
            veh_dis = {}
            if all_lane_vehicles[lane] != []:
                for v in all_lane_vehicles[lane]:
                    dis = all_vehicle_distance[v]
                    veh_dis.update({v: dis})
                first_veh = max(veh_dis, key=veh_dis.get)  # 获取当前lane的第一辆车的编号
                if all_vehicle_speed[first_veh] <= 0.1:
                    self.lane_first_vehicle_current_step.update({lane: first_veh})
                else:
                    self.lane_first_vehicle_current_step.update({lane: 0})
            else:
                self.lane_first_vehicle_current_step.update({lane: 0})
        # 获取每条道路第一辆等待车的等待时间
        for lane in list_lanes:
            # if 现在的第一辆车和之前的第一辆车相同
            if self.lane_first_vehicle_current_step[lane] != 0 and self.lane_first_vehicle_current_step[lane] == self.lane_first_vehicle_previous_step[lane]:
                self.lane_first_vehicle_waiting_time[lane] += 1
            # 否则
            else:
                self.lane_first_vehicle_waiting_time[lane] = 0
        self.lane_first_vehicle_previous_step = deepcopy(self.lane_first_vehicle_current_step)
        return self.lane_first_vehicle_waiting_time

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        # dic_state = {state_feature_name: self.dic_feature[state_feature_name] for state_feature_name in list_state_features}
        dic_state = []
        for state_feature_name in ["cur_phase","lane_num_vehicle","lane_first_vehicle_waiting_time"]:
            dic_state.extend(self.dic_feature[state_feature_name])

        return np.array(dic_state)     # 修改类型

    def get_adjs(self):
        adjs = self.dic_feature["adjacency_matrix"]

        return np.array(adjs)

    def get_reward(self, dic_reward_info):
        # customize your own reward
        dic_reward = dict()

        dic_reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(self.dic_feature["lane_num_vehicle_been_stopped_thres1"])

        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward

    def get_cost(self):
        cost = self.dic_feature['lane_first_vehicle_waiting_time']
        # cost = list(self.lane_first_vehicle_waiting_time.values())
        # 每个交叉口，如果有道路超过最长等待时间，这个道上为1，否则为0'
        for i in range(len(cost)):
            if cost[i] > self.max_waiting_time:
                cost[i] = 1
            else:
                cost[i] = 0
        return cost

    def get_lane_waiting_time_count(self, vehicle_waiting_time):
        '''
        get_lane_waiting_time_count
        Get waiting time of vehicles in each lane.

        :param: None
        :return lane_waiting_time: waiting time of vehicles in each lane
        '''
        # the sum of waiting times of vehicles on the lane since their last halt.
        lane_waiting_time = {}
        lane_ind_waiting_time = {}
        lane_vehicles = self.eng.get_lane_vehicles()
        for lane in self.list_entering_lanes:
            lane_waiting_time[lane] = 0
            lane_ind_waiting_time[lane] = []
            for vehicle in lane_vehicles[lane]:
                lane_waiting_time[lane] += vehicle_waiting_time[vehicle]
                lane_ind_waiting_time[lane].append(vehicle_waiting_time[vehicle])
        return lane_waiting_time, lane_ind_waiting_time

class AnonEnv:
    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf, max_waiting_time):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.simulator_type = self.dic_traffic_env_conf["SIMULATOR_TYPE"]
        self.max_waiting_time = max_waiting_time
        self.vehicle_waiting_time = {}

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print ("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            #raise ValueError

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):

        print("# self.eng.reset() to be implemented")

        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": 0,
            "laneChange": False,
            "dir": self.path_to_work_directory+"/",
            "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],
            "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],
            "rlTrafficLight": self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
            "saveReplay": self.dic_traffic_env_conf["SAVEREPLAY"],
            "roadnetLogFile": "roadnetLogFile.json",
            "replayLogFile": "replayLogFile.txt"
        }
        print("=========================")
        # print(cityflow_config)
        with open(os.path.join(self.path_to_work_directory,"cityflow.config"), "w") as json_file:
            json.dump(cityflow_config, json_file)
        self.eng = engine.Engine(os.path.join(self.path_to_work_directory,"cityflow.config"), thread_num=1)

        # get adjacency
        if self.dic_traffic_env_conf["USE_LANE_ADJACENCY"]:
            self.traffic_light_node_dict = self._adjacency_extraction_lane()
        else:
            self.traffic_light_node_dict = self._adjacency_extraction()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i+1, j+1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict["intersection_{0}_{1}".format(i+1, j+1)],self.path_to_log, self.max_waiting_time)
                                  for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"])]

        self.list_inter_log = [[] for i in range(self.dic_traffic_env_conf["NUM_ROW"] *
                                                 self.dic_traffic_env_conf["NUM_COL"])]

        # set index for intersections and global index for lanes
        self.id_to_index = {}
        count_inter = 0
        for i in range(self.dic_traffic_env_conf["NUM_ROW"]):
            for j in range(self.dic_traffic_env_conf["NUM_COL"]):
                self.id_to_index['intersection_{0}_{1}'.format(i+1, j+1)] = count_inter
                count_inter += 1

        self.lane_id_to_index = {}
        count_lane = 0
        for i in range(len(self.list_intersection)): # TODO
            for j in range(len(self.list_intersection[i].list_entering_lanes)):
                lane_id = self.list_intersection[i].list_entering_lanes[j]
                if lane_id not in self.lane_id_to_index.keys():
                    self.lane_id_to_index[lane_id] = count_lane
                    count_lane += 1

        # build adjacency_matrix_lane in index from _adjacency_matrix_lane
        for inter in self.list_intersection:
            inter.build_adjacency_row_lane(self.lane_id_to_index)


        # get new measurements
        system_state_start_time = time.time()
        self.system_states = {
                              "get_vehicles": self.eng.get_vehicles(include_waiting=False),
                              "get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }
        # print("Get system state time: ", time.time()-system_state_start_time)

        update_start_time = time.time()
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states)
        # print("Update_current_measurements_map time: ", time.time()-update_start_time)


        state, done = self.get_state()
        adjs = self.get_adjs()
        # print(state)
        return state, adjs

    def step(self, action):
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action_list = [0]*len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()
            # state = self.get_state()

            self._inner_step(action_in_sec)

        reward = self.get_reward()

        # log
        self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)

        next_state, done = self.get_state()

        cost = self.get_cost()   # cost:每个路口中若有道路超过最长等待时间，给-1惩罚； sum_cost:每个路口的总违反次数

        avg_travel_time = self.eng.get_average_travel_time()
        all_vehicle_waiting_time = list(self.vehicle_waiting_time.values())
        lane_veh, lane_veh_ind = self.get_lane_waiting_time_count()
        # print("veh", veh)
        # print("veh_ind", veh_ind)
        # print("all_vehicle_waiting_time", all_vehicle_waiting_time)
        return next_state, reward, cost, done, avg_travel_time, all_vehicle_waiting_time, lane_veh, lane_veh_ind

    def _inner_step(self, action):

        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_conf["ALL_RED_TIME"]
            )

        # run one step
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()

        system_state_start_time = time.time()
        self.system_states = {
                              "get_vehicles": self.eng.get_vehicles(include_waiting=False),
                              "get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }

        # print("Get system state time: ", time.time()-system_state_start_time)

        if self.dic_traffic_env_conf['DEBUG']:
            print("Get system state time: {}".format(time.time()-start_time))
        # get new measurements

        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()

        update_start_time = time.time()
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states)

        self.get_vehicle_waiting_time(self.system_states)

        # print("Update_current_measurements_map time: ", time.time()-update_start_time)

        if self.dic_traffic_env_conf['DEBUG']:
            print("Update measurements time: {}".format(time.time()-start_time))

        #self.log_lane_vehicle_position()
        # self.log_first_vehicle()
        #self.log_phase()


    def _check_episode_done(self, list_state):

        # ======== to implement ========

        return False

    def get_vehicle_waiting_time(self, simulator_state):
        '''
        get_vehicle_waiting_time
        Get waiting time of vehicles according to vehicle's speed.
        If a vehicle's speed less than 0.1m/s, then its waiting time would be added 1s.

        :param: None
        :return vehicle_waiting_time: waiting time of vehicles
        '''
        # the waiting time of vehicle since last halt.
        vehicles = simulator_state["get_vehicles"]
        vehicle_speed = simulator_state["get_vehicle_speed"]
        for vehicle in vehicles:
            if vehicle not in self.vehicle_waiting_time.keys():
                self.vehicle_waiting_time[vehicle] = 0
            if vehicle_speed[vehicle] < 0.1:
                self.vehicle_waiting_time[vehicle] += 1
            else:
                self.vehicle_waiting_time[vehicle] = 0
        return self.vehicle_waiting_time

    def get_lane_waiting_time_count(self):
        list_veh_waiting_time = [inter.get_lane_waiting_time_count(self.vehicle_waiting_time)[0] for inter in self.list_intersection]
        list_veh_ind_waiting_time = [inter.get_lane_waiting_time_count(self.vehicle_waiting_time)[1] for inter in self.list_intersection]
        return list_veh_waiting_time, list_veh_ind_waiting_time

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):
        # consider neighbor info
        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
        done = [self._check_episode_done(list_state) for inter in self.list_intersection]   ##### fwq此处原本只有一个值

        # print(list_state)
        # print("list_state", list_state)
        return list_state, done

    def get_adjs(self):
        list_adjs = [inter.get_adjs() for inter in self.list_intersection]
        return list_adjs

    @staticmethod
    def _reduce_duplicates(feature_name_list):
        new_list = set()
        for feature_name in feature_name_list:
            if feature_name[-1] in ["0","1","2","3"]:
                new_list.add(feature_name[:-2])
        return list(new_list)

    def get_reward(self):

        list_reward = [[inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"])] for inter in self.list_intersection]

        return list_reward

    def get_cost(self):
        list_cost = [inter.get_cost() for inter in self.list_intersection]
        penalty = []
        for i in list_cost:
            # # 对每个路口，如果有道路超过最长等待时间，就给一个-1惩罚
            if all(j == 0 for j in i):
                penalty.append([0])
            else:
                penalty.append([1])

        return penalty

    def get_first_time(self):
        list_cost = [inter.get_first_time() for inter in self.list_intersection]
        return list_cost

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                    "state": before_action_feature[inter_ind],
                                                    "action": action[inter_ind]})

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            if int(inter_ind)%100 == 0:
                print("Batch log for inter ",inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle,orient='index')
            df.to_csv(path_to_log_file, na_rep="nan")

            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            self.batch_log(start, stop)

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            # print(net)
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                       'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,
                                                            "entering_lane_ENWS": None,}


            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()


            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1


            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['inter_id_to_index'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    neighboring_node = edge_id_dict[road_id]['to']
                    # calculate the neighboring intersections
                    if neighboring_node not in traffic_light_node_dict.keys(): # virtual node
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_node)
                    # calculate the entering lanes ENWS
                    for key, value in edge_id_dict.items():
                        if value['from'] == neighboring_node and value['to'] == i:
                            neighboring_road = key

                            neighboring_lanes = []
                            for k in range(value['num_of_lane']):
                                neighboring_lanes.append(neighboring_road+"_{0}".format(k))

                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])


            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                # TODO return with Top K results
                if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']: # use geo-distance
                    row = np.array([0]*total_inter_num)
                    # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                    for j in traffic_light_node_dict.keys():
                        location_2 = traffic_light_node_dict[j]['location']
                        dist = AnonEnv._cal_distance(location_1,location_2)
                        row[inter_id_to_index[j]] = dist
                    if len(row) == top_k:
                        adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                    elif len(row) > top_k:
                        adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                    else:
                        adjacency_row_unsorted = [k for k in range(total_inter_num)]
                    adjacency_row_unsorted.remove(inter_id_to_index[i])
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]] + adjacency_row_unsorted
                else: # use connection infomation
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]]
                    for j in traffic_light_node_dict[i]['neighbor_ENWS']: ## TODO
                        if j is not None:
                            traffic_light_node_dict[i]['adjacency_row'].append(inter_id_to_index[j])
                        else:
                            traffic_light_node_dict[i]['adjacency_row'].append(-1)


                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num

        return traffic_light_node_dict

    def _adjacency_extraction_lane(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])

        roadnet = RoadNet('{0}'.format(file))
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            # print(net)
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                       'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,
                                                            "entering_lane_ENWS": None,
                                                            "total_lane_num": None, 'adjacency_matrix_lane': None,
                                                            "lane_id_to_index": None,
                                                            "lane_ids_in_intersction": []
                                                            }

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            top_k_lane = self.dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"]
            total_inter_num = len(traffic_light_node_dict.keys())

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()


            # set inter id to index dict
            inter_id_to_index = {}
            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            # set the neighbor_ENWS nodes and entring_lane_ENWS for intersections
            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['inter_id_to_index'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    neighboring_node = edge_id_dict[road_id]['to']
                    # calculate the neighboring intersections
                    if neighboring_node not in traffic_light_node_dict.keys(): # virtual node
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_node)
                    # calculate the entering lanes ENWS
                    for key, value in edge_id_dict.items():
                        if value['from'] == neighboring_node and value['to'] == i:
                            neighboring_road = key

                            neighboring_lanes = []
                            for k in range(value['num_of_lane']):
                                neighboring_lanes.append(neighboring_road+"_{0}".format(k))

                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])


            lane_id_dict = roadnet.net_lane_dict
            total_lane_num = len(lane_id_dict.keys())
            # output an adjacentcy matrix for all the intersections
            # each row stands for a lane id,
            # each column is a list with two elements: first is the lane's entering_lane_LSR, second is the lane's leaving_lane_LSR
            def _get_top_k_lane(lane_id_list, top_k_input):
                top_k_lane_indexes = []
                for i in range(top_k_input):
                    lane_id = lane_id_list[i] if i < len(lane_id_list) else None
                    top_k_lane_indexes.append(lane_id)
                return top_k_lane_indexes

            adjacency_matrix_lane = {}
            for i in lane_id_dict.keys(): # Todo lane_ids should be in an order
                adjacency_matrix_lane[i] = [_get_top_k_lane(lane_id_dict[i]['input_lanes'], top_k_lane),
                                            _get_top_k_lane(lane_id_dict[i]['output_lanes'], top_k_lane)]



            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                # TODO return with Top K results
                if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']: # use geo-distance
                    row = np.array([0]*total_inter_num)
                    # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                    for j in traffic_light_node_dict.keys():
                        location_2 = traffic_light_node_dict[j]['location']
                        dist = AnonEnv._cal_distance(location_1,location_2)
                        row[inter_id_to_index[j]] = dist
                    if len(row) == top_k:
                        adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                    elif len(row) > top_k:
                        adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                    else:
                        adjacency_row_unsorted = [k for k in range(total_inter_num)]
                    adjacency_row_unsorted.remove(inter_id_to_index[i])
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]] + adjacency_row_unsorted
                else: # use connection infomation
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]]
                    for j in traffic_light_node_dict[i]['neighbor_ENWS']: ## TODO
                        if j is not None:
                            traffic_light_node_dict[i]['adjacency_row'].append(inter_id_to_index[j])
                        else:
                            traffic_light_node_dict[i]['adjacency_row'].append(-1)


                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num
                traffic_light_node_dict[i]['total_lane_num'] = total_lane_num
                traffic_light_node_dict[i]['adjacency_matrix_lane'] = adjacency_matrix_lane



        return traffic_light_node_dict



    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a-b)**2))

def parse_args():
    parser = argparse.ArgumentParser()
    # The file folder to create/log in
    parser.add_argument("--env", type=int, default=1)  # env=1 means you will run CityFlow
    parser.add_argument("--gui", type=bool, default=False)

    parser.add_argument("--mod", type=str, default='CoLight')  # SimpleDQN,SimpleDQNOne,GCN,CoLight,Lit
    parser.add_argument("--cnt", type=int, default=3600)  # 3600
    parser.add_argument("-all", action="store_true", default=False)

    global TOP_K_ADJACENCY
    TOP_K_ADJACENCY = 5
    global TOP_K_ADJACENCY_LANE
    TOP_K_ADJACENCY_LANE = 5
    global EARLY_STOP
    EARLY_STOP = False
    global SAVEREPLAY  # if you want to relay your simulation, set it to be True
    SAVEREPLAY = True
    global ADJACENCY_BY_CONNECTION_OR_GEO
    # TAKE CARE
    ADJACENCY_BY_CONNECTION_OR_GEO = False

    # modify:TOP_K_ADJACENCY in line 154
    global PRETRAIN
    PRETRAIN = False

    global THRESHOLD
    THRESHOLD = 2

    global ANON_PHASE_REPRE
    tt = parser.parse_args()
    if 'CoLight_Signal' in tt.mod:
        # 12dim
        ANON_PHASE_REPRE = {
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
        }
    else:
        # 12dim
        ANON_PHASE_REPRE = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0]
        }
    print('agent_name:%s', tt.mod)
    # print('ANON_PHASE_REPRE:', ANON_PHASE_REPRE)

    return parser.parse_args()

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

def copy_conf_file(dic_path, dic_exp_conf, dic_traffic_env_conf):
    # write conf files

    path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
              indent=4)
    json.dump(dic_traffic_env_conf,
              open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

def copy_anon_file(dic_path, dic_exp_conf):
    # hard code !!!

    path = dic_path["PATH_TO_WORK_DIRECTORY"]
    # copy sumo files
    #
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_exp_conf["TRAFFIC_FILE"][0]),
                os.path.join(path, dic_exp_conf["TRAFFIC_FILE"][0]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_exp_conf["ROADNET_FILE"]),
                os.path.join(path, dic_exp_conf["ROADNET_FILE"]))

def path_check(dic_path):
    # check path
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])

def main(memo, env, road_net, gui, volume, suffix, mod, cnt, max_waiting_time):
    NUM_COL = int(road_net.split('_')[0])
    NUM_ROW = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)

    ENVIRONMENT = ["sumo", "anon"][env]

    traffic_file_list = ["{0}_{1}_{2}_{3}".format(ENVIRONMENT, road_net, volume, suffix)]
    traffic_file_list = [i + ".json" for i in traffic_file_list]

    global PRETRAIN
    global EARLY_STOP
    for traffic_file in traffic_file_list:
        dic_exp_conf_extra = {

            "RUN_COUNTS": cnt,
            "MODEL_NAME": mod,
            "TRAFFIC_FILE": [traffic_file],  # here: change to multi_traffic

            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "PRETRAIN": PRETRAIN,  #

            "AGGREGATE": False,
            "DEBUG": False,
            "EARLY_STOP": EARLY_STOP,
        }

        global TOP_K_ADJACENCY
        global TOP_K_ADJACENCY_LANE
        global SAVEREPLAY
        global ADJACENCY_BY_CONNECTION_OR_GEO
        global ANON_PHASE_REPRE
        dic_traffic_env_conf_extra = {
            "USE_LANE_ADJACENCY": True,
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": "set",
            "MEASURE_TIME": 10,
            "IF_GUI": gui,
            "DEBUG": False,
            "TOP_K_ADJACENCY": TOP_K_ADJACENCY,
            "ADJACENCY_BY_CONNECTION_OR_GEO": ADJACENCY_BY_CONNECTION_OR_GEO,
            "TOP_K_ADJACENCY_LANE": TOP_K_ADJACENCY_LANE,
            "SIMULATOR_TYPE": ENVIRONMENT,
            "BINARY_PHASE_EXPANSION": True,
            "FAST_COMPUTE": False,
            "MODEL_NAME": mod,

            "SAVEREPLAY": SAVEREPLAY,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "VOLUME": volume,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "phase_expansion": {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            },

            "phase_expansion_4_lane": {
                1: [1, 1, 0, 0],
                2: [0, 0, 1, 1],
            },

            "LIST_STATE_FEATURE": [
                "cur_phase",
                "lane_num_vehicle",
                "lane_first_vehicle_waiting_time",
            ],

            "DIC_FEATURE_DIM": dict(
                D_LANE_QUEUE_LENGTH=(4,),
                D_LANE_NUM_VEHICLE=(4,),
                D_LANE_FIRST_VEHICLE_WAITING_TIME=(4,),

                D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                D_CUR_PHASE=(1,),
            ),

            "DIC_REWARD_INFO": {
                "sum_num_vehicle_been_stopped_thres1": -0.25,
            },

            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": {
                "anon": ANON_PHASE_REPRE,
            }
        }

        ## ==================== multi_phase ====================
        if volume == 'jinan':
            template = "Jinan"
        elif volume == 'mydata':
            template = "mydata"
        elif volume == 'hangzhou':
            template = 'Hangzhou'
        elif volume == 'newyork':
            template = 'NewYork'
        else:
            raise ValueError

        if mod in ['CoLight', 'GCN', 'SimpleDQNOne']:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = 1
            dic_traffic_env_conf_extra['ONE_MODEL'] = False
            if "adjacency_matrix" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                    "adjacency_matrix_lane" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                    mod not in ['SimpleDQNOne']:
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix")
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix_lane")
                if dic_traffic_env_conf_extra['ADJACENCY_BY_CONNECTION_OR_GEO']:
                    TOP_K_ADJACENCY = 5
                    dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("connectivity")
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CONNECTIVITY'] = (5,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = (5,)
                else:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = (
                    dic_traffic_env_conf_extra['TOP_K_ADJACENCY'],)

                if dic_traffic_env_conf_extra['USE_LANE_ADJACENCY']:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX_LANE'] = (
                    dic_traffic_env_conf_extra['TOP_K_ADJACENCY_LANE'],)
        else:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

        if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,)
        # print(traffic_file)
        prefix_intersections = str(road_net)
        deploy_dic_path = {
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo,
                                                   traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S',
                                                                                      time.localtime(time.time()))),
            "PATH_TO_DATA": os.path.join("data", template, prefix_intersections),
        }

        deploy_dic_exp_conf = merge(envs.env_config.DIC_EXP_CONF, dic_exp_conf_extra)
        deploy_dic_traffic_env_conf = merge(envs.env_config.dic_traffic_env_conf, dic_traffic_env_conf_extra)

    # deploy_dic_path = dic_path_extra
    path_check(deploy_dic_path)

    copy_conf_file(deploy_dic_path, deploy_dic_exp_conf, deploy_dic_traffic_env_conf)
    copy_anon_file(deploy_dic_path, deploy_dic_exp_conf)

    path_to_log = os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")

    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)

    env = AnonEnv(
        path_to_log=path_to_log,
        path_to_work_directory=deploy_dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=deploy_dic_traffic_env_conf,
        max_waiting_time=max_waiting_time)

    return env

def get_env(memo, road_net, volume, suffix, max_waiting_time):
    args = parse_args()
    env = main(memo, args.env, road_net, args.gui, volume,
               suffix, args.mod, args.cnt, max_waiting_time)
    return env
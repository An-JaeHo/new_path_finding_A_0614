import numpy as np
import heapq
import math
import queue
import time
import matplotlib.pyplot as plt
from .a_star_pathfinder import AStarPathfinder
import traceback

class Path:
    def __init__(self):
        self.initial_obstacles = []
        self.obstacle_set = set()
        self.map_size = 300
        self.grid_size = 5
        self.grid = np.zeros((self.map_size, self.map_size), dtype=int)
        self.path = []
        self.current_path_index = -1
        self.SAFETY_MARGIN = 1
        self.time =0
        self.last_calculated_target = None
        self.MIN_NEXT_POINT_DISTANCE = 10
        self.path_check = True

    def update_obstacle(self, obstacle_data):

        obstacles = obstacle_data.get('obstacles', [])
        
        if not isinstance(obstacles, list):
            print(f"Error: 'obstacles' must be a list, got {type(obstacles)}")
            return

        for obstacle in obstacles:
            if not isinstance(obstacle, dict) or not all(key in obstacle for key in ['x_min', 'x_max', 'z_min', 'z_max']):
                print(f"Error: Invalid obstacle format: {obstacle}")
                continue
                
            try:
                x_min = int(max(0, math.floor(obstacle['x_min'])))
                x_max = int(min(self.map_size - 1, math.ceil(obstacle['x_max'])))
                z_min = int(max(0, math.floor(obstacle['z_min'])))
                z_max = int(min(self.map_size - 1, math.ceil(obstacle['z_max'] )))
                
                '''
                x, z축에서 떨어진 거리 계산
                시작위치를 바로 못 넣어서 처음 위치를 50 30 넣었습니다. info가 들어오기전에 obstacle를 먼저 실행하기 때문
                '''
                dx = max(x_min - 50, 0, 50 - x_max)
                dz = max(z_min - 30, 0, 30 - z_max)

                distance = math.sqrt(dx**2 + dz**2)

                '''
                이 부분이 거리에 따라 obstacles의 를 가지고 오는 부분입니다.
                전체 obstacles를 가지고 여기서 분류합니다.
                '''
                # if distance < 100:
                #     for x in range(x_min, x_max + 1):
                #         for z in range(z_min, z_max + 1):
                #             self.initial_obstacles.append({
                #             "x": x,
                #             "z": z,
                #             "radius": 4
                #         })
                
            except (ValueError, TypeError) as e:
                print(f"Error processing obstacle {obstacle}: {e}")
                continue
        
        # print(f"Obstacle count (with safety margin): {np.sum(self.grid)}")
    

    '''
    이건 사용 안하고 있습니다.
    '''
    def angle_diff(self,angle):
        """각도를 -180도 ~ 180도 범위로 정규화"""
        angle = math.fmod(angle + 180, 360)
        if angle < 0:
            angle += 360
        return angle - 180

    '''
    목표가 앵글 0이라는 기준으로 잡고 BODY X의 앵들을 구하는 용도
    '''
    def normalize_angle_360(self,angle):
        """각도를 0 ~ 360 범위로 정규화"""
        angle = math.fmod(angle, 360)
        if angle < 0:
            angle += 360
        return angle

    '''
    이건 사용 안하고 있습니다.
    '''
    def check_safety_margin(self, start, end, safety_margin=2):
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        
        num_samples = int(np.linalg.norm(end - start) * 4) + 1
        if num_samples < 2:
            return True
        
        t = np.linspace(0, 1, num_samples)
        for i in range(num_samples):
            point = start + t[i] * (end - start)
            if not self.is_point_safe(point, safety_margin):
                return False
        return True

    def get_action(self, log_data, target_point):

        # 최초 A* 계산 시간 기록
        if not hasattr(self, "last_path_time"):
            self.last_path_time = time.time() 

        try:
            player_pos = log_data.get("playerPos", {})
            self.tank_x = int(round(player_pos["x"]))
            self.tank_y = int(round(player_pos["y"]))
            self.tank_z = int(round(player_pos["z"]))
            self.target_point = target_point
            self.body_x = int(round(log_data.get("playerBodyX", 0.0)))
            self.body_y = int(round(log_data.get("playerBodyY", 0.0)))
            self.body_z = int(round(log_data.get("playerBodyZ", 0.0)))
            self.tank_yam = self.angle_diff(self.body_x)
            self.tank_speed = np.clip(int(round(log_data.get("playerSpeed", 0.0))), 0, 70)

            if log_data:
                latest_log_data = log_data.get('lidarPoints',{})

            if latest_log_data and self.path_check:
                self.current_path_index = 1
                self.find_path(latest_log_data,target_point)
                self.path_check = False
                self.last_path_time = time.time()  # 🔄 경로 갱신 시점 기록

        except Exception as e:
            print("Error in get_action:")
            traceback.print_exc()  # 전체 에러 추적 출력
            return [-0.9, 0.0]
        
        '''
        경로가 들어왔을 경우에만  실행하도록 합니다
        '''
        if self.path:
            dx = self.path[self.current_path_index]['x'] - self.tank_x
            dz = self.path[self.current_path_index]['z'] - self.tank_z
            
            # 현재 tank가 바라보고 있는 방향을 받아 옵니다.
            current_angle = log_data.get('playerBodyX', 0)
            
            target_angle_rad = math.atan2(dx, dz)
            target_angle = math.degrees(target_angle_rad)
            
            if target_angle < 0:
                target_angle += 360

            # 상대 방향: 내가 보는 방향이 목표 기준에서 얼마나 벗어났는지
            toward_angle = self.normalize_angle_360(current_angle - target_angle)

            if 90 <= toward_angle <= 180:
                movews = -10.0
                movead = 1.7
            elif 40 <= toward_angle < 90:
                movews = 0.1
                movead = 1.2
            elif 20 <= toward_angle < 40:
                movews = 0.3
                movead = 0.9
            elif 0 <= toward_angle < 20:
                movews = 1.0
                movead = 0.5

            elif 180 < toward_angle <= 270:
                movews = -10.0
                movead = -1.7
            elif 270 < toward_angle <= 320:
                movews = 0.1
                movead = -1.2
            elif 320 < toward_angle <= 340:
                movews = 0.3
                movead = -0.9
            elif 340 < toward_angle <= 360:
                movews = 1.0
                movead = -0.5

            if log_data.get("playerSpeed", 0.0) > 7:
                movews = -0.85

            
            '''1초 이상 지났다면 경로 초기화 
            아직 마지막 경로는 지정하지 않았습니다.'''
            if time.time() - self.last_path_time >= 1.0 :
                movews = -0.85
                movead = 0

                self.path = []
                self.path_check = True
            return movews, movead
        else:
            movews = -0.85
            movead = 0
            return movews, movead
    def simplify_path(self, path, tolerance=5.0):
        """
        Ramer–Douglas–Peucker 기반 경로 단순화.
        tolerance: 직선에서 얼마나 벗어나야 분기할지 허용 오차.
        """
        if len(path) < 3:
            return path

        simplified_path = [path[0]]  # 시작점 포함
        start_index = 0

        while True:
            max_deviation = 0.0
            furthest_index = -1
            p1 = path[start_index]
            
            for i in range(start_index + 1, len(path)):
                p2 = path[i]

                # 선분 거리 계산
                dx = p2['x'] - p1['x']
                dz = p2['z'] - p1['z']
                length = math.sqrt(dx**2 + dz**2)
                if length == 0:
                    continue

                # 선분과 점 간 거리 측정
                for j in range(start_index + 1, i):
                    px = path[j]['x']
                    pz = path[j]['z']
                    num = abs((dx) * (p1['z'] - pz) - (p1['x'] - px) * (dz))
                    deviation = num / length

                    if deviation > max_deviation:
                        max_deviation = deviation
                        furthest_index = j

                # 허용 편차 초과 시 해당 포인트까지 경로로 인정
                if max_deviation > tolerance:
                    simplified_path.append(path[furthest_index])
                    start_index = furthest_index
                    break
            else:
                # 끝까지 tolerance 이내면 마지막 점 추가하고 종료
                simplified_path.append(path[-1])
                break

        return simplified_path

    # def simplify_path(self,path, tolerance=5.0):
    #     """
    #     주어진 경로를 단순화하여 웨이포인트 수를 줄입니다.
    #     `tolerance`는 경로가 직선에서 벗어나는 허용치를 나타냅니다.
    #     값이 클수록 더 많이 단순화됩니다.
    #     """
    #     if len(path) < 3:
    #         return path

    #     simplified_path = [path[0]] # 시작점은 항상 포함

    #     start_index = 0
    #     end_index = 0

    #     while end_index < len(path) - 1:
    #         p1 = path[start_index]
    #         p2_candidate_index = start_index + 1
            
    #         furthest_point_in_segment_index = -1
    #         max_deviation_in_segment = -1.0

    #         for i in range(start_index + 1, len(path)):
    #             current_point = path[i]
    #             p2_candidate = current_point 
                
    #             current_max_dev = -1.0
    #             current_furthest_idx = -1
                
    #             for j in range(start_index + 1, i + 1): # start_index 다음부터 current_point까지
    #                 temp_point = path[j]
                    
    #                 numerator = abs((p2_candidate['x'] - p1['x']) * (p1['z'] - temp_point['z']) - \
    #                                 (p1['x'] - temp_point['x']) * (p2_candidate['z'] - p1['z']))
    #                 denominator = math.sqrt((p2_candidate['x'] - p1['x'])**2 + (p2_candidate['z'] - p1['z'])**2)
                    
    #                 distance = 0
    #                 if denominator != 0:
    #                     distance = numerator / denominator 
                    
    #                 if distance > current_max_dev:
    #                     current_max_dev = distance
    #                     current_furthest_idx = j

    #             if current_max_dev > tolerance:
    #                 simplified_path.append(path[furthest_point_in_segment_index])
    #                 start_index = furthest_point_in_segment_index
    #                 end_index = i 
    #                 break 
    #             else:
    #                 if current_max_dev > max_deviation_in_segment:
    #                     max_deviation_in_segment = current_max_dev
    #                     furthest_point_in_segment_index = current_furthest_idx

    #                 end_index = i 

    #         else: 
    #             simplified_path.append(path[len(path) - 1])
    #             break

    #     if path and path[-1] not in simplified_path:
    #         simplified_path.append(path[-1])

    #     return simplified_path
    
    
    '''
    기본적인 ASTAR와 PATH FINDING을 하는 곳입니다.
    '''
    def find_path(self, latest_log_data, target_point):
        
        try:
            ''' A STAR
            맵 크기와 grid size를 지정했습니다.
            start_world : 현재 나의 위치  ->  get action 할때 로그 데이터로 넣어줍니다.
            end_world : 최종 도착할 위치 
            '''
            map_width = self.map_size
            map_height = self.map_size
            grid_size = self.grid_size
            start_world = [self.tank_x, self.tank_z]
            end_world = target_point

            '''라이다 필터링 '''
            '''탱크가 저지대에 있을 경우와 기울어 졌을 경우'''
            if self.tank_y <8 or 1 < self.body_y  < 359:
                
                if 3 <= self.body_y <= 357:
                    filtered_lidar = [
                    point for point in latest_log_data
                    if point.get("channelIndex") ==10 and "position" in point
                    ]
                else:
                    filtered_lidar = [
                    point for point in latest_log_data
                    if point.get("channelIndex") == 8 and "position" in point
                    ]
            else:
                '''평범한 길을 가고 있을 때'''
                filtered_lidar = [
                point for point in latest_log_data
                if point.get("channelIndex") == 7 and "position" in point
                ]

            

            
            '''장애물 추가 (중복은 set으로 빠르게 필터링)'''
            for obs_raw in filtered_lidar:
                pos = obs_raw["position"]
                x, z = pos.get("x"), pos.get("z")

                if x is None or z is None:
                    continue

                truncated_x = round(x, 1)
                truncated_z = round(z, 1)
                key = (truncated_x, truncated_z)

                if key not in self.obstacle_set :
                    self.initial_obstacles.append({
                        "x": x,
                        "z": z,
                        "radius": 4
                    })
                    self.obstacle_set.add(key)

            '''Pathfinder 초기화 및 장애물 반영
            a_star_pathfinder파일의 AStarPathfinder 객체를 만들어 그 객체에 obstacles를 추가합니다.
            '''
            self.grid = np.zeros((self.map_size, self.map_size), dtype=int)
            pathfinder = AStarPathfinder(map_width, map_height, grid_size, self.grid)
            
            pathfinder.update_obstacles(self.initial_obstacles,start_world,end_world)

            print(f'장애물 개수: {len(self.initial_obstacles)}')

        except queue.Empty:
            print("Queue is empty.")
            return

        '''A* 경로 탐색'''
        try:
            if self.tank_x is not None and self.tank_z is not None:

                '''AStarPathfinder의 객체에 있는 find_path를 실행시켜 경로를 만들고
                    simplify_path를 통해 경로를 최적화 시킵니다.                
                '''
                raw_path = pathfinder.find_path(start_world, end_world)
                simplified_path = self.simplify_path(raw_path, tolerance=7.0)

                self.path = simplified_path

                print(f'경로 길이: {len(self.path)}')

                if simplified_path:
                    print("경로 시각화 가능.")
                    '''#####################################
                    ##########경로 시각화 하는 코드입니다#######
                    ########################################'''
                    # pathfinder.visualize_astar_grid(obstacles=self.initial_obstacles,path=self.path, start_pos=start_world, end_pos=end_world)
                else:
                    print("단순화된 경로 없음.")
            else:
                print("탱크 위치 없음 또는 Pathfinder 미초기화.")
        except queue.Empty:
            print("Queue error during path calculation.")
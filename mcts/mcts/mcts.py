# # adapted from https://github.com/jys5609/MC-LAVE-RL.git

# import numpy as np
# from tqdm import tqdm
# import mcts.mcts.utils as utils
# from collections import defaultdict
# from mcts.virtualhome.llm_policy import LLMPolicy

# DISCOUNT_FACTOR = 0.95

# class StateNode:
#     def __init__(self, reward=0, done=False):
#         self.ob = None
#         self.look = None
#         self.inv = None
#         self.state = None
#         self.prev_action = None
#         self.id = None
#         self.valid_actions = None
#         self.history = []
#         self.parent = None
#         self.parent_action_id = None
#         self.best_action_node = None
        

#         self.N = 0
#         self.children = []
#         self.children_probs = []
#         self.reward = reward/(1-DISCOUNT_FACTOR)
#         self.score = 0
#         self.done = done
#         self.predicted_reward = 0
#         self.use_llm = False


# class ActionNode:
#     def __init__(self, action):
#         self.action = action
#         self.N = 0
#         self.Q = 0
#         self.Q_hat = 0
#         self.Rs = []
#         self.children = None
#         self.children_id = None

# # 轨迹记录
# class SimulationHistoryTracker:
#     def __init__(self):
#         self.histories = {}
        
#     def start_simulation(self, sim_id):
#         self.histories[sim_id] = []
        
#     def add_step(self, sim_id, depth, action, history):
#         # self.histories[sim_id].append({
#         #     'depth': depth,
#         #     'action': action,
#         #     'history': history.copy()
#         # })
#         if action != ' ':  # 不记录初始状态的标记
#             self.histories[sim_id].append({
#                 'depth': depth,
#                 'action': action,
#                 'history': history.copy() if history else []
#             })
        
#     # def print_simulation(self, sim_id):
#     #     print(f"\n=== 模拟 #{sim_id} 的完整历史 ===")

#     #     if not self.histories[sim_id]:
#     #         print("没有执行任何动作")
#     #         return
        
#     #     for step in self.histories[sim_id]:
#     #         print(f"深度 {step['depth']}: {step['action']}")
#     #     print(f"最终历史: {self.histories[sim_id][-1]['history']}\n")

#     # 因为有重复，所以修改
#     def print_simulation(self, sim_id):
#         print(f"\n=== 模拟 #{sim_id} 的完整历史 ===")

#         if not self.histories[sim_id]:
#             print("没有执行任何动作")
#             return
        
#         # 优化输出格式，合并相同深度的连续记录
#         optimized_steps = []
#         current_depth = None
        
#         for step in self.histories[sim_id]:
#             if step['depth'] != current_depth:
#                 optimized_steps.append(step)
#                 current_depth = step['depth']
#             else:
#                 # 用新步骤替换相同深度的最后一个步骤
#                 optimized_steps[-1] = step
        
#         for step in optimized_steps:
#             print(f"深度 {step['depth']}: {step['action']}")
        
#         print(f"最终历史: {self.histories[sim_id][-1]['history']}\n")

# class MCTSAgent:
#     def __init__(self, args, env, policy=None, name='MCTS', 
#                 uct_type='PUCT', valid_action_dict=None, actions_info=None,
#                   log_dir=None, visited_transitions=None, replay_file=None,
#                   use_llm=True):
#         self.env = env
#         self.name = name
#         # self.num_actions = env.action_num
#         self.best_action_node = None
#         self.uct_type = uct_type
#         # self.seed = args.seed
#         self.seed = 1234
#         self.round = args.round
#         self.root = None


#         self.exploration_constant = args.exploration_constant
#         self.bonus_constant = args.bonus_constant
#         self.max_depth = args.max_depth
#         self.simulation_per_act = args.simulation_per_act
#         self.discount_factor = args.discount_factor
#         self.visited_transitions = visited_transitions

#         self.action_selection_temp = 0.1 / (self.round + 1)

#         self.policy = policy
#         self.actions = [] if actions_info is None else actions_info[0]
#         self.actions_e = [] if actions_info is None else actions_info[1]

#         self.action_values = defaultdict(set)   # Ex: {north: [3.01, 2.00, 5.01]}

#         self.maxlen_obs = 150
#         self.maxlen_look = 150
#         self.maxlen_inv = 50
#         self.maxlen_action = 12
#         self.simulation_num = args.simulation_num
#         self.use_llm = use_llm
#         if use_llm:
#             self.llm_policy = LLMPolicy(device="cuda:0", model=args.model) 
#         self.q_network = None
#         # self.valid_action_dict = env.action_dict
#         # self.valid_action_dict = {} if valid_action_dict is None else valid_action_dict
#         self.state_dict = {}
#         self.action_embedding = {}
#         self.replay_file = replay_file

#         # 新增
#         self.history_tracker = SimulationHistoryTracker()

#     def search(self, ob, history, cur_depth, valid_actions, done):
#         '''
#         Search the best action with probs
#         :return: best action
#         '''
#         init_history = history.copy()
#         # if '*** You have won ***' in next_state_text or '*** You have died ***' in next_state_text:
#         #     score = int(next_state_text.split('you scored ')[1].split(' out of')[0])
#         #     reward = score - state_node.score
#         #     info['score'] = score

#         # self.write_buffer(state_node, best_action_node, ob, reward, done, info)

#         # if self.root is not None and self.root.best_action_node.children is not None:
#         #     self.root = self.root.best_action_node.children
#         #     self.root.parent = None
#         # else:
#         self.root = self.build_state(ob, history, valid_actions, done, use_llm=self.use_llm)

#         # for _ in tqdm(range(self.simulation_num)):
#         # 新修改
#         for i in tqdm(range(self.simulation_num)):
#         # for _ in tqdm(range(self.simulation_per_act * len(self.root.children))):

#             # 新增
#             self.history_tracker.start_simulation(i)
            
#             self.env.reset()
#             self.env.history = init_history.copy()
#             # _, root = self.simulate(self.root, 0)
#             # 新修改
#             _, root = self.simulate(self.root, 0, i)
#             self.history_tracker.print_simulation(i)
            
#             self.root = root
#         # select best action by Q-value
#         best_action_node_idx = self.greedy_action_node(self.root, 0, 0, if_print=True)
#         # select best action by Count
#         # best_action_node = self.max_visit_action_node(self.root)
#         best_action_node = self.root.children[best_action_node_idx]
#         self.root.best_action_node = best_action_node
#         return self.root.best_action_node.action

#     @staticmethod
#     def state_id(history: list):
#         return ' '.join(history)

#     def rebuild_state(self, state, ob, history, valid_actions, done, reward=0, prev_action=' ', use_llm=False):
#         state.id = self.state_id(history)
#         # state.id = ob + info['look'] + info['inv'] + str(reward) + str(info['score']) + prev_action
#         state.valid_actions = valid_actions
#         state.use_llm = use_llm

#         if not use_llm:
#             state.children_probs = np.ones((len(state.valid_actions),)) / len(state.valid_actions)

# # 修改6取消注释
#         elif state.id in self.state_dict.keys():
#             state.children_probs = self.state_dict[state.id].children_probs

#         # 并新增
#         else:
#             # 使用LLM计算概率
#             state.children_probs, state.predicted_reward = self.llm_policy._calculate_emperical_prob(
#                 history, ob, valid_actions, self.env.get_goal(), 10, 0, 0.95)
#             # 归一化处理
#             state.children_probs = np.array(state.children_probs)
#             if np.sum(state.children_probs) != 1:
#                 state.children_probs /= np.sum(state.children_probs)

#         self.state_dict[state.id] = state
#         for valid_action in state.valid_actions:
#             if isinstance(state.valid_actions, dict):
#                 state.children.append(ActionNode(state.valid_actions[valid_action]))
#             else:
#                 state.children.append(ActionNode(valid_action))

#         return state

#     def build_state(self, ob, history, valid_actions, done, reward=0, prev_action=' ', use_llm=False):
#         state = StateNode()
#         state.ob = ob
#         # state.look = info['look']
#         # state.inv = info['inv']
#         state.state = ob
#         state.done = done
#         # state.state = ob + info['look'] + info['inv']
#         state.reward = reward
#         # state.score = info['score']
#         state.prev_action = prev_action
#         state.history = history
#         state.id = self.state_id(history)
#         # state.id = ob + info['look'] + info['inv'] + str(reward) + str(info['score']) + prev_action
#         state.valid_actions = valid_actions
#         state.use_llm = use_llm

#         # if not use_llm:
#             # state.children_probs = np.ones((len(state.valid_actions),)) / len(state.valid_actions)

#            
#         # else:
#         #     state.children_probs, state.predicted_reward = self.llm_policy._calculate_emperical_prob(
#         #         history, ob, valid_actions, self.env.get_goal(), 10, 0, 0.95)
#            
#         state.children_probs, state.predicted_reward = self.llm_policy._calculate_emperical_prob(
#             history, ob, valid_actions, self.env.get_goal(), 10, 0, 0.95)

#         self.state_dict[state.id] = state
#         for valid_action in state.valid_actions:
#             if isinstance(state.valid_actions, dict):
#                 state.children.append(ActionNode(state.valid_actions[valid_action]))
#             else:
#                 state.children.append(ActionNode(valid_action))

#         return state

#         # 修改3：归一化
#         # ✅ 归一化处理，防止概率之和不等于1
#         # state.children_probs = np.array(state.children_probs)
#         # if np.sum(state.children_probs) != 1:
#         #     state.children_probs /= np.sum(state.children_probs)

#         # self.state_dict[state.id] = state
#         # for valid_action in state.valid_actions:
#         #     if isinstance(state.valid_actions, dict):
#         #         state.children.append(ActionNode(state.valid_actions[valid_action]))
#         #     else:
#         #         state.children.append(ActionNode(valid_action))

#         # return state

#        
#     # def simulate(self, state_node, depth):
#     # 新增
#     def simulate(self, state_node, depth, sim_count=0):
#         """
#         Simulate from a state node
#         Args:
#             state_node: current state node
#             depth: current depth
#             sim_count: current simulation count
#         Returns:
#             reward, next state node
#         """
#         if state_node.done or depth == self.max_depth:
#             return 0, state_node

#         best_action_node_idx = self.greedy_action_node(state_node, self.exploration_constant, self.bonus_constant)
#         best_action_node = state_node.children[best_action_node_idx]
#         rollout_next = False
#         ob, reward, done, history, valid_actions = self.env.step(best_action_node.action)
#         
#         next_state_id = self.state_id(history)
#         # path_of_nodes.append((state_node, best_action_node))
#         if next_state_id == best_action_node.children_id:
#             next_state_node = best_action_node.children
#             if next_state_node.use_llm == False:
#                 next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=best_action_node.action, use_llm=self.use_llm)
#                 # best_action_node.children[index] = next_state_node
#                 next_state_node.parent = state_node
#                 rollout_next = True
#         else: 
#             next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=best_action_node.action, use_llm=self.use_llm)
#             next_state_node.parent = state_node
#             best_action_node.children = next_state_node
#             best_action_node.children_id = next_state_node.id
#             rollout_next = True


#         if rollout_next:
#             if self.use_llm:
#                 rollout_r = []
#                 for _ in range(1):
#                     random_r = reward + self.discount_factor * self.rollout(next_state_node, depth+1, sim_count)
#                     rollout_r.append(random_r)  
#                 R = sum(rollout_r)/len(rollout_r)
#             else:
#                 rollout_r = []
#                 for _ in range(1):
#                     random_r = reward + self.discount_factor * self.rollout(next_state_node, depth+1, sim_count)
#                     rollout_r.append(random_r)  
#                 R = sum(rollout_r)/len(rollout_r)
#         else:
#             r, next_state_node = self.simulate(next_state_node, depth+1, sim_count)
#             R = reward + self.discount_factor * r

#         state_node.N += 1
#         best_action_node.N += 1
#         best_action_node.children = next_state_node
#         best_action_node.Rs.append(R)
#         best_action_node.Q = np.sum(np.array(best_action_node.Rs) * utils.softmax(best_action_node.Rs, T=10))
#         state_node.best_action_node = best_action_node       
#         return R, state_node

#     def max_visit_action_node(self, state_node):
#         children_count = []

#         for i in range(len(state_node.children)):
#             child = state_node.children[i]
#             children_count.append(child.N)

#         children_count = children_count / np.max(children_count)
#         count_based_probs = children_count ** (1/self.action_selection_temp) / (np.sum(children_count ** (1/self.action_selection_temp)))
#         return np.random.choice(state_node.children, p=count_based_probs)

#     def greedy_action_node(self, state_node, exploration_constant, bonus_constant, if_print=False):
#         best_value = -np.inf
#         best_children = []
#         best_children_prob = []
#         for i in range(len(state_node.children)):
#             child = state_node.children[i]
#             assert len(state_node.children_probs) == len(state_node.children), print(state_node.children_probs)
#             child_prob = state_node.children_probs[i]
            
#             if exploration_constant == 0:
#                 ucb_value = child.Q
#             elif self.uct_type == 'UCT':
#                 ucb_value = child.Q + exploration_constant * np.sqrt(np.log(state_node.N + 1) / (child.N + 1))
#                 # print(child.Q, exploration_constant * np.sqrt(np.log(state_node.N + 1) / (child.N + 1)))
#             elif self.uct_type == 'PUCT':
#                 # print(child_prob)
#                 ucb_value = child.Q + exploration_constant * child_prob * np.sqrt(state_node.N) / (child.N + 1)
#             elif self.uct_type == 'MC-LAVE':
#                 if child.action in self.action_embedding.keys():
#                     action_e = self.action_embedding[child.action]
#                 else:
#                     action_e = utils.vectorize(child.action)
#                     self.action_embedding[child.action] = action_e

#                 actions = list(self.action_values.keys())
#                 if child.action in actions:
#                     actions.pop(actions.index(child.action))

#                 actions_e = []
#                 for a in actions:
#                     actions_e.append(self.action_embedding[a])

#                 near_act, near_idx = utils.find_near_actions(action_e, actions, np.array(actions_e), threshold=0.8)
#                 if len(near_idx) == 0:
#                     child.Q_hat = 0
#                 else:
#                     near_Qs = set()
#                     for a in near_act:
#                         near_Qs.add(np.mean(list(self.action_values[a])))
#                     near_Qs = list(near_Qs)
#                     child.Q_hat = utils.softmax_value(near_Qs)

#                 ucb_value = child.Q \
#                             + exploration_constant * np.sqrt(state_node.N + 1) / (child.N + 1) * child_prob \
#                             + bonus_constant * np.sqrt(state_node.N + 1) / (child.N + 1) * child.Q_hat

#             else:
#                 raise NotImplementedError

#             if ucb_value == best_value:
#                 best_children.append(i)
#                 best_children_prob.append(child_prob)
#             elif ucb_value > best_value:
#                 best_value = ucb_value
#                 best_children = [i]
#                 best_children_prob = [child_prob]
#         if if_print:
#             for c in state_node.children:
#                 if c.N > 0:
#                     print(c.action, c.Q, c.N)
#         best_children_prob = np.array(best_children_prob) / np.sum(best_children_prob)
#         output_action_index = np.argmax(best_children_prob)
#         return best_children[output_action_index]

#     # def rollout(self, state_node, depth):
#     #     if state_node.done or depth == self.max_depth:
#     #         return 0
#     #     # action_node = np.random.choice(state_node.children, 1)[0]

#     #      # 改动1：用 children_probs 控制 Rollout 策略
#     #     action_node = np.random.choice(
#     #         state_node.children,
#     #         p=state_node.children_probs
#     #     )

#     #     action = action_node.action

#     #     ob, reward, done, history, valid_actions = self.env.step(action)
#     #     if done:
#     #         print("Done!")
#     #     next_state_id = self.state_id(history)


#     #     if next_state_id == action_node.children_id:
#     #         next_state_node = action_node.children
#     #     else:
#     #         next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=action)
#     #         next_state_node.parent = state_node
#     #         action_node.children = next_state_node
#     #         action_node.children_id = next_state_node.id
#     #     r = reward + self.discount_factor * self.rollout(next_state_node, depth+1)
#     #     return r


#     # 新修改2
#     def rollout(self, state_node, depth, sim_count=0):
#         if state_node.done or depth == self.max_depth:
#             return 0
#         # action_node = np.random.choice(state_node.children, 1)[0]
#          # 改动1：用 children_probs 控制 Rollout 策略
#         # action_node = np.random.choice(
#         #     state_node.children,
#         #     p=state_node.children_probs
#         # )

#         # 去掉随机性
#         action_idx = np.argmax(state_node.children_probs)
#         action_node = state_node.children[action_idx]

#         action = action_node.action

#         ob, reward, done, history, valid_actions = self.env.step(action)

#         # 新增
#         # 添加轨迹记录
#         if sim_count is not None:
#             self.history_tracker.add_step(
#                 sim_count,
#                 depth,
#                 action,
#                 history
#             )

#         if done:
#             print("Done!")
#         next_state_id = self.state_id(history)


#         if next_state_id == action_node.children_id:
#             next_state_node = action_node.children
#         else:
#             next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=action)
#             next_state_node.parent = state_node
#             action_node.children = next_state_node
#             action_node.children_id = next_state_node.id
#         # r = reward + self.discount_factor * self.rollout(next_state_node, depth+1)
#         # 新修改
#         # 在递归调用时传递simulation_id参数
#         r = reward + self.discount_factor * self.rollout(next_state_node, depth+1, sim_count)

#         # 如果是最深层的rollout，则保存轨迹到文件
#         if depth == self.max_depth - 1 or next_state_node.done:
#             # 创建文件名 task_step_simulation.txt
#             filename = f"{self.task_count}_{self.step_count}_{sim_count}.txt"
#             filepath = os.path.join(self.trajectory_dir, filename)
            
#             # 保存轨迹到文件
#             with open(filepath, "w", encoding="utf-8") as f:
#                 f.write(f"Task {self.task_count}, Step {self.step_count}, Simulation {sim_count}\n")
#                 f.write("Complete trajectory:\n")
#                 f.write("\n".join(history))
                
#             print(f"轨迹已保存到: {filepath}")

#         return r# adapted from https://github.com/jys5609/MC-LAVE-RL.git

import numpy as np
from tqdm import tqdm
import mcts.mcts.utils as utils
from collections import defaultdict
from mcts.virtualhome.llm_policy import LLMPolicy
import os

DISCOUNT_FACTOR = 0.95

class StateNode:
    def __init__(self, reward=0, done=False):
        self.ob = None
        self.look = None
        self.inv = None
        self.state = None
        self.prev_action = None
        self.id = None
        self.valid_actions = None
        self.history = []
        self.parent = None
        self.parent_action_id = None
        self.best_action_node = None
        

        self.N = 0
        self.children = []
        self.children_probs = []
        self.reward = reward/(1-DISCOUNT_FACTOR)
        self.score = 0
        self.done = done
        self.predicted_reward = 0
        self.use_llm = False


class ActionNode:
    def __init__(self, action):
        self.action = action
        self.N = 0
        self.Q = 0
        self.Q_hat = 0
        self.Rs = []
        self.children = None
        self.children_id = None


class MCTSAgent:
    def __init__(self, args, env, policy=None, name='MCTS', 
                uct_type='PUCT', valid_action_dict=None, actions_info=None,
                  log_dir=None, visited_transitions=None, replay_file=None,
                  use_llm=True):
        self.env = env
        self.name = name
        # self.num_actions = env.action_num
        self.best_action_node = None
        self.uct_type = uct_type
        self.seed = 1234
        self.round = args.round
        self.root = None


        self.exploration_constant = args.exploration_constant
        self.bonus_constant = args.bonus_constant
        self.max_depth = args.max_depth
        self.simulation_per_act = args.simulation_per_act
        self.discount_factor = args.discount_factor
        self.visited_transitions = visited_transitions

        self.action_selection_temp = 0.1 / (self.round + 1)

        self.policy = policy
        self.actions = [] if actions_info is None else actions_info[0]
        self.actions_e = [] if actions_info is None else actions_info[1]

        self.action_values = defaultdict(set)   # Ex: {north: [3.01, 2.00, 5.01]}

        self.maxlen_obs = 150
        self.maxlen_look = 150
        self.maxlen_inv = 50
        self.maxlen_action = 12
        self.simulation_num = args.simulation_num
        self.use_llm = use_llm
        if use_llm:
            self.llm_policy = LLMPolicy(device="cuda:1", model=args.model) 
        self.q_network = None
        # self.valid_action_dict = env.action_dict
        # self.valid_action_dict = {} if valid_action_dict is None else valid_action_dict
        self.state_dict = {}
        self.action_embedding = {}
        self.replay_file = replay_file

        # 添加用于追踪任务和步骤的计数器
        self.task_count = 0  # 当前是第几个任务
        self.step_count = 0  # 当前任务的第几步
        
        # 添加轨迹保存路径
        self.trajectory_dir = "trajectories17"
        if not os.path.exists(self.trajectory_dir):
            os.makedirs(self.trajectory_dir)

        # 用于保存 few-shot 反思内容
        self.memory = []
        # 定义并创建 memory 文件夹
        self.memory_dir = "memory14"
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)

    def few_shot_reflection(self, history):
        """
        从 few_shot_examples.txt 中读取示例，然后与当前轨迹拼接构成 prompt, 最后调用 LLM 生成反思内容
        """

        goal = str(self.env.get_goal())  # 直接调用 env.get_goal() 获取目标

        # 读取 few-shot 示例
        with open("/mnt/wsy/test2/llm-mcts/mcts/mcts/few_shot_examples.txt", "r", encoding="utf-8") as f:
            few_shot_examples = f.read()
        
        # 构造 prompt：包含 few-shot 示例和当前轨迹
        prompt = f"""\nYou will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Start your plan with exactly "New plan:" followed by your detailed plan in a SINGLE PARAGRAPH without numbered steps or bullet points. Here are some examples:
        \n{few_shot_examples}
        \n{goal}
        {''.join(history)}"""

        # 调用 LLM 生成反思内容
        reflection = self.llm_policy.generate_reflection(prompt)
        
        # 确保反思以"New plan:"开头
        if not reflection.startswith("New plan:"):
            reflection = "New plan:" + reflection
            
        return reflection

    def search(self, ob, history, cur_depth, valid_actions, done):
        '''
        Search the best action with probs
        :return: best action
        '''
        init_history = history.copy()
        
        # # 添加打印语句
        # print("初始history:")
        # print(history)
        
        # if '*** You have won ***' in next_state_text or '*** You have died ***' in next_state_text:
        #     score = int(next_state_text.split('you scored ')[1].split(' out of')[0])
        #     reward = score - state_node.score
        #     info['score'] = score

        # self.write_buffer(state_node, best_action_node, ob, reward, done, info)

        # if self.root is not None and self.root.best_action_node.children is not None:
        #     self.root = self.root.best_action_node.children
        #     self.root.parent = None
        # else:
        self.root = self.build_state(ob, history, valid_actions, done, use_llm=self.use_llm,memory=self.memory)

        for i in tqdm(range(self.simulation_num)):
            # print(f"\n---------------------- 模拟 #{i+1}/{self.simulation_num} ----------------------")
            self.root = self.build_state(ob, history, valid_actions, done, use_llm=self.use_llm, memory=self.memory)
            self.env.reset()
            self.env.history = init_history.copy()
            _, root = self.simulate(self.root, 0, i)  # 直接传递i作为sim_count
            self.root = root
            
        self.step_count += 1  # 每次search完成后增加step计数
        
        # select best action by Q-value
        best_action_node_idx = self.greedy_action_node(self.root, 0, 0, if_print=True)
        # select best action by Count
        # best_action_node = self.max_visit_action_node(self.root)
        best_action_node = self.root.children[best_action_node_idx]
        self.root.best_action_node = best_action_node
        
        # # 添加打印语句
        # print(f"最终选择的动作: {best_action_node.action}")
        # print(f"当前history: {self.root.history}")
        
        return self.root.best_action_node.action

    @staticmethod
    def state_id(history: list):
        return ' '.join(history)

    def rebuild_state(self, state, ob, history, valid_actions, done, reward=0, prev_action=' ', use_llm=False):
        state.id = self.state_id(history)
        # state.id = ob + info['look'] + info['inv'] + str(reward) + str(info['score']) + prev_action
        state.valid_actions = valid_actions
        state.use_llm = use_llm

        if not use_llm:
            state.children_probs = np.ones((len(state.valid_actions),)) / len(state.valid_actions)
            # print("nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
        # elif state.id in self.state_dict.keys():
        #     state.children_probs = self.state_dict[state.id].children_probs
        self.state_dict[state.id] = state
        for valid_action in state.valid_actions:
            if isinstance(state.valid_actions, dict):
                state.children.append(ActionNode(state.valid_actions[valid_action]))
            else:
                state.children.append(ActionNode(valid_action))

        return state

    def build_state(self, ob, history, valid_actions, done, reward=0, prev_action=' ', use_llm=False,memory=None):
        # 添加打印语句
        # print(f"构建状态时的history (prev_action={prev_action}):")
        # print(history)
        
        state = StateNode()
        state.ob = ob
        # state.look = info['look']
        # state.inv = info['inv']
        state.state = ob
        state.done = done
        # state.state = ob + info['look'] + info['inv']
        state.reward = reward
        # state.score = info['score']
        state.prev_action = prev_action
        state.history = history
        state.id = self.state_id(history)
        # state.id = ob + info['look'] + info['inv'] + str(reward) + str(info['score']) + prev_action
        state.valid_actions = valid_actions
        state.use_llm = use_llm

        if not use_llm:
            state.children_probs = np.ones((len(state.valid_actions),)) / len(state.valid_actions)
            print("nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
            
        else:
            state.children_probs, state.predicted_reward = self.llm_policy._calculate_emperical_prob(
                history, ob, valid_actions, self.env.get_goal(), 10, 0, 0.95, memory=self.memory)
            # 利用大语言模型计算各个动作的先验概率，并预测奖励
            
        self.state_dict[state.id] = state
        for valid_action in state.valid_actions:
            if isinstance(state.valid_actions, dict):
                state.children.append(ActionNode(state.valid_actions[valid_action]))
            else:
                state.children.append(ActionNode(valid_action))

        return state

        
    def simulate(self, state_node, depth, sim_count=0):
        """
        Simulate from a state node
        Args:
            state_node: current state node
            depth: current depth
            sim_count: current simulation count
        Returns:
            reward, next state node
        """
        if state_node.done or depth == self.max_depth:
            return 0, state_node

        best_action_node_idx = self.greedy_action_node(state_node, self.exploration_constant, self.bonus_constant)
        best_action_node = state_node.children[best_action_node_idx]
        rollout_next = False
        ob, reward, done, history, valid_actions = self.env.step(best_action_node.action)
        
        next_state_id = self.state_id(history)

        if next_state_id == best_action_node.children_id:
            next_state_node = best_action_node.children
            if next_state_node.use_llm == False:
                next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=best_action_node.action, use_llm=self.use_llm,memory=self.memory)
                next_state_node.parent = state_node
                rollout_next = True
        else: 
            next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=best_action_node.action, use_llm=self.use_llm,memory=self.memory)
            next_state_node.parent = state_node
            best_action_node.children = next_state_node
            best_action_node.children_id = next_state_node.id
            rollout_next = True

        if rollout_next:
            if self.use_llm:
                rollout_r = []
                for _ in range(1):
                    random_r = reward + self.discount_factor * self.rollout(next_state_node, depth+1, sim_count)
                    rollout_r.append(random_r)  
                R = sum(rollout_r)/len(rollout_r)
            else:
                rollout_r = []
                for _ in range(1):
                    random_r = reward + self.discount_factor * self.rollout(next_state_node, depth+1, sim_count)
                    rollout_r.append(random_r)  
                R = sum(rollout_r)/len(rollout_r)
        else:
            r, next_state_node = self.simulate(next_state_node, depth+1, sim_count)
            R = reward + self.discount_factor * r

        state_node.N += 1
        best_action_node.N += 1
        best_action_node.children = next_state_node
        best_action_node.Rs.append(R)
        best_action_node.Q = np.sum(np.array(best_action_node.Rs) * utils.softmax(best_action_node.Rs, T=10))
        state_node.best_action_node = best_action_node       
        return R, state_node

    def max_visit_action_node(self, state_node):
        children_count = []

        for i in range(len(state_node.children)):
            child = state_node.children[i]
            children_count.append(child.N)

        children_count = children_count / np.max(children_count)
        count_based_probs = children_count ** (1/self.action_selection_temp) / (np.sum(children_count ** (1/self.action_selection_temp)))
        return np.random.choice(state_node.children, p=count_based_probs)

    def greedy_action_node(self, state_node, exploration_constant, bonus_constant, if_print=False):
        best_value = -np.inf
        best_children = []
        best_children_prob = []
        for i in range(len(state_node.children)):
            child = state_node.children[i]
            assert len(state_node.children_probs) == len(state_node.children), print(state_node.children_probs)
            child_prob = state_node.children_probs[i]
            
            if exploration_constant == 0:
                ucb_value = child.Q
            elif self.uct_type == 'UCT':
                ucb_value = child.Q + exploration_constant * np.sqrt(np.log(state_node.N + 1) / (child.N + 1))
                # print(child.Q, exploration_constant * np.sqrt(np.log(state_node.N + 1) / (child.N + 1)))
            elif self.uct_type == 'PUCT':
                # print(child_prob)
                ucb_value = child.Q + exploration_constant * child_prob * np.sqrt(state_node.N) / (child.N + 1)
            elif self.uct_type == 'MC-LAVE':
                if child.action in self.action_embedding.keys():
                    action_e = self.action_embedding[child.action]
                else:
                    action_e = utils.vectorize(child.action)
                    self.action_embedding[child.action] = action_e

                actions = list(self.action_values.keys())
                if child.action in actions:
                    actions.pop(actions.index(child.action))

                actions_e = []
                for a in actions:
                    actions_e.append(self.action_embedding[a])

                near_act, near_idx = utils.find_near_actions(action_e, actions, np.array(actions_e), threshold=0.8)
                if len(near_idx) == 0:
                    child.Q_hat = 0
                else:
                    near_Qs = set()
                    for a in near_act:
                        near_Qs.add(np.mean(list(self.action_values[a])))
                    near_Qs = list(near_Qs)
                    child.Q_hat = utils.softmax_value(near_Qs)

                ucb_value = child.Q \
                            + exploration_constant * np.sqrt(state_node.N + 1) / (child.N + 1) * child_prob \
                            + bonus_constant * np.sqrt(state_node.N + 1) / (child.N + 1) * child.Q_hat

            else:
                raise NotImplementedError

            if ucb_value == best_value:
                best_children.append(i)
                best_children_prob.append(child_prob)
            elif ucb_value > best_value:
                best_value = ucb_value
                best_children = [i]
                best_children_prob = [child_prob]
        if if_print:
            for c in state_node.children:
                if c.N > 0:
                    print(c.action, c.Q, c.N)
        best_children_prob = np.array(best_children_prob) / np.sum(best_children_prob)
        output_action_index = np.argmax(best_children_prob)
        return best_children[output_action_index]

    def rollout(self, state_node, depth, sim_count=0):
        if state_node.done or depth == self.max_depth:
            return 0
        # action_node = np.random.choice(state_node.children, 1)[0]
        # # 使用概率加权选择，而不是完全随机
        # probs = state_node.children_probs / np.sum(state_node.children_probs)
        # action_node = np.random.choice(state_node.children, 1, p=probs)[0]

        best_idx = np.argmax(state_node.children_probs)
        action_node = state_node.children[best_idx]


        action = action_node.action

        ob, reward, done, history, valid_actions = self.env.step(action)
        
        # 添加打印语句
        # print(f"Rollout中执行动作 {action} 后的history:")
        # print(history)
        
        if done:
            print("Done!")
        next_state_id = self.state_id(history)


        if next_state_id == action_node.children_id:
            next_state_node = action_node.children
        else:
            # next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=action)
            next_state_node = self.build_state(ob, history, valid_actions, done, reward, prev_action=action, use_llm=self.use_llm,memory=self.memory)

            next_state_node.parent = state_node
            action_node.children = next_state_node
            action_node.children_id = next_state_node.id
        r = reward + self.discount_factor * self.rollout(next_state_node, depth+1, sim_count)

        # 如果是最深层的rollout，则保存轨迹到文件
        if depth == self.max_depth - 1 or next_state_node.done:
            # 创建文件名 task_step_simulation.txt
            filename = f"{self.task_count}_{self.step_count}_{sim_count}.txt"
            filepath = os.path.join(self.trajectory_dir, filename)
            
            # 保存轨迹到文件
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Task {self.task_count}, Step {self.step_count}, Simulation {sim_count}\n")
                f.write("Goal: " + str(self.env.get_goal()) + "\n")
                f.write("Complete trajectory:\n")
                f.write("\n".join(history))
                f.write("\n")
                f.write(str(done) + "\n")
                
            # print(f"轨迹已保存到: {filepath}")

            # 如果模拟失败，进行 few-shot 反思
            if not done:
                reflection = self.few_shot_reflection(history)
                self.memory.append(reflection)
                # ——立刻更新当前节点的先验——
                # state_id = next_state_node.id
                # node = self.state_dict[state_id]
                # node.children_probs, _ = self.llm_policy._calculate_emperical_prob(
                #     node.history, node.state, node.valid_actions,
                #     self.env.get_goal(), 10, 0, 0.95,
                #     memory=self.memory
                # )
                # 2. 重新计算先验和预测价值
                node = self.state_dict[next_state_node.id]
                children_probs, pred_value = self.llm_policy._calculate_emperical_prob(
                    history=node.history,
                    observation=node.state,
                    valid_action_list=node.valid_actions,
                    instruction=str(self.env.get_goal()),
                    done_reward=reward,            # 用 rollout 得到的真实 reward
                    step_reward=0,                 # 或者你定义的 step reward
                    discount_factor=self.discount_factor,
                    memory=self.memory
                )
                node.children_probs   = children_probs
                node.predicted_reward = pred_value
                # 单独保存反思到 memory 文件夹
                memory_filename = f"{self.task_count}_{self.step_count}_reflection.txt"
                memory_filepath = os.path.join(self.memory_dir, memory_filename)
                with open(memory_filepath, "a", encoding="utf-8") as mem_file:
                    mem_file.write(f"Simulation {sim_count} Reflection:\n")
                    mem_file.write("reflection:\n")
                    mem_file.write(reflection + "\n")

        return r


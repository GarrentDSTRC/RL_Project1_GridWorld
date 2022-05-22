import logging
import random
import gym
import numpy as np


#logger = logging.getLogger(__name__)


class GridEnv1_nn(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        self.states = list(range(16))  # 状态空间 0-15


        self.terminate_states = np.zeros(len(self.states))  # 终止状态为np格式
        self.terminate_states[4] = 1
        self.terminate_states[3] = 1

        self.actions = [-4, 4, -1, 1, 0]  #上下左右不动


        self.size = 4



        #self.states.remove(6)
        #self.states.remove(9)
        #self.states.remove(8)

        #self.vtab=np.zeros([16,],dtype = float)

        #self.gamma = 0.8  # 折扣因子
        self.viewer = None
        self.state = 15


    def step(self,action):             #action 01234
        # 系统当前状态
        state = self.state
        as_n = self.state + self.actions[action]

        if  ((as_n not in self.states))or\
                (as_n ==6 or as_n== 8 or as_n== 9) or(action==2 and  self.state % self.size == 0) or (action == 3 and self.state % self.size == 3) or action==4:
            self.state = state
            return state, -2, False

        elif self.terminate_states[as_n]:
            self.state=as_n
            return as_n, 100, True

        else:
            self.state = as_n
            return as_n, -1,False


    def reset(self):
        #self.state = self.states[int(random.random() * len(self.states))]
        self.state=15
        return self.state

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 600

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # 创建网格世界
            self.line1 = rendering.Line((100, 100), (500, 100))
            self.line2 = rendering.Line((100, 200), (500, 200))
            self.line3 = rendering.Line((100, 300), (500, 300))
            self.line4 = rendering.Line((100, 400), (500, 400))
            self.line5 = rendering.Line((100, 500), (500, 500))
            self.line6 = rendering.Line((100, 100), (100, 500))
            self.line7 = rendering.Line((200, 100), (200, 500))
            self.line8 = rendering.Line((300, 100), (300, 500))
            self.line9 = rendering.Line((400, 100), (400, 500))
            self.line10 = rendering.Line((500, 100), (500, 500))

            # #创建石柱
            # self.shizhu = rendering.make_circle(40)
            # self.circletrans = rendering.Transform(translation=(250,350))
            # self.shizhu.add_attr(self.circletrans)
            # self.shizhu.set_color(0.8,0.6,0.4)

            # 创建第一个火坑
            self.fire1 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(250, 250))
            self.fire1.add_attr(self.circletrans)
            self.fire1.set_color(1, 0, 0)

            # 创建第二个火坑
            self.fire2 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(150, 250))
            self.fire2.add_attr(self.circletrans)
            self.fire2.set_color(1, 0, 0)

            # 创建第三个火坑
            self.fire3 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(350, 350))
            self.fire3.add_attr(self.circletrans)
            self.fire3.set_color(1, 0, 0)

            # 创建宝石
            self.diamond = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(150, 350))
            self.diamond.add_attr(self.circletrans)
            self.diamond.set_color(0, 0, 1)

            # 创建机器人
            self.robot = rendering.make_circle(30)
            at=(self.state%4+1)*100+50
            bt=(4-self.state//4)*100+50
            self.robotrans = rendering.Transform(translation=(at,bt))
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0, 1, 0)

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            # self.viewer.add_geom(self.shizhu)
            self.viewer.add_geom(self.fire1)
            self.viewer.add_geom(self.fire2)
            self.viewer.add_geom(self.fire3)
            self.viewer.add_geom(self.diamond)
            self.viewer.add_geom(self.robot)

        if self.state is None:
            return None
        x = (self.state % 4 + 1) * 100 + 50
        y = (4 - self.state // 4) * 100 + 50
        self.robotrans.set_translation(x, y)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
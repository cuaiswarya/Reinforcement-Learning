import numpy as np
from numpy.random import uniform
from typing import *

import gym
from gym import spaces

from pyrlprob.mdp import AbstractMDP

from environment.landing_dyn import *
#memory = 10**9


class Landing2DEnv(AbstractMDP):
    """
    One-Dimensional Landing Problem.
    Reference: https://doi.org/10.2514/6.2008-6615
    """

    def __init__(self, config) -> None:
        """
        Definition of observation and action spaces
        """

        super().__init__(config=config)


        #Time step
        self.time_step = self.tf/(float(self.H))

        #Maximum episode steps
        self.max_episode_steps = self.H

        #Observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        #Action space
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)



    def get_observation(self,
                        state,
                        control) -> np.ndarray:
        """
        Get current observation: height and velocity
        """

        observation = np.array([state["x"], state["h"], state["u"], state["v"], state["m"], state["t"]], dtype=np.float32)

        return observation

    

    def get_control(self,
                    action,
                    state) -> float:
        """
        Get current control: thrust value
        """

        control = 0.5*(action + 1.)*self.Tmax

        return control



    def next_state(self,
                   state, 
                   control,
                   time_step) -> Dict[str, float]:
        """
        Propagate state: integration of system dynamics
        """

        #State at current time-step
        s = np.array([state["x"],state["h"], state["u"], state["v"], state["m"]])

        #Integration of equations of motion
        data = np.concatenate((self.g, control, self.c), axis = None)
        
        # Integration
        t_eval = np.array([state['t'], state['t'] + time_step])
        sol = rk4(dynamics, s, t_eval, data)

        #State at next time-step
        self.success = True
        s_new = {"x": sol[0],"h": sol[1], "u": sol[2], "v": sol[3], "m": sol[4], "t": t_eval[-1], "step": state["step"] + 1}
        
        return s_new
    

    def collect_reward(self,
                       prev_state, 
                       state, 
                       control) -> Tuple[float, bool]:
        """
        Get current reward and done signal.
        """

        done = False

        reward = state["m"] - prev_state["m"]

        if state["step"] == self.H:
            done = True
        if not self.success:
            done = True
        if state["h"] <= 0. or state["m"] <= 0.:
            done = True
        
        
        if done:
            cstr_viol = max(abs(state["x"] - self.xf), abs(state["h"] - self.hf), abs(state["u"] - self.uf), abs(state["v"] - self.vf) - 0.005)
            state["cstr_viol"] = max(cstr_viol, 0.)

            reward = reward - 10.*cstr_viol
        
        return reward, done
    

    def get_info(self,
                 prev_state,
                 state,
                 control,
                 observation,
                 reward,
                 done) -> Dict[str, float]:
        """
        Get current info.
        """

        info = {}
        info["episode_step_data"] = {}
        info["episode_step_data"]["x"] = [prev_state["x"]]
        info["episode_step_data"]["h"] = [prev_state["h"]] 
        info["episode_step_data"]["u"] = [prev_state["u"]]
        info["episode_step_data"]["v"] = [prev_state["v"]] 
        info["episode_step_data"]["m"] = [prev_state["m"]] 
        info["episode_step_data"]["t"] = [prev_state["t"]] 
        info["episode_step_data"]["Tx"] = [control[0]]
        info["episode_step_data"]["Th"] = [control[1]]
        if done:
            info["custom_metrics"] = {}
            info["episode_end_data"] = {}
            info["episode_step_data"]["x"].append(state["x"])
            info["episode_step_data"]["h"].append(state["h"]) 
            info["episode_step_data"]["u"].append(state["u"])
            info["episode_step_data"]["v"].append(state["v"]) 
            info["episode_step_data"]["m"].append(state["m"]) 
            info["episode_step_data"]["t"].append(state["t"]) 
            info["episode_step_data"]["Tx"].append(control[0])
            info["episode_step_data"]["Th"].append(control[1])
            info["episode_end_data"]["xf"] = state["x"]
            info["episode_end_data"]["hf"] = state["h"]
            info["episode_end_data"]["uf"] = state["u"]
            info["episode_end_data"]["vf"] = state["v"]
            info["episode_end_data"]["mf"] = state["m"]
            info["custom_metrics"]["cstr_viol"] = state["cstr_viol"]

        return info
    

    def reset(self) -> np.ndarray:
        """ 
        Reset the environment
        """

        self.state = {}
        self.state["x"] = uniform(self.x0_min, self.x0_max)
        self.state["h"] = uniform(self.h0_min, self.h0_max)
        self.state["u"] = uniform(self.u0_min, self.u0_max)
        self.state["v"] = uniform(self.v0_min, self.v0_max)
        self.state["m"] = self.m0
        self.state["t"] = 0.
        self.state["step"] = 0

        control = 0.

        observation = self.get_observation(self.state, control)

        return observation


    

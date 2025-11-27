import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import utils

import gc


PI = 3.1415926535
MAX_STEPS = 3000 #per episode

# seed
seed = 206265
np.random.seed(seed)
random.seed(seed)

############################## 25.11.17 ##############################
# Refactorized Code only for 1 lc - 1 Env case
# All bugs are fixed & optimized --> reference of optimization
# (also for sub-classes)
# for KPC
######################################################################


class CutterSphere():
    def __init__(self, ast, random = True, mode = 'Rxyz_assign', *args):
        """
        initialize
        - if random == True, use random parameters
        - if random == False, use parameters from *args
        *args = (R, x1, y1, z1)
            if mode == 'ratio_assign' *args = (phi, theta, r_cen_ratio, r_cut_ratio)
        ast : Asteroid_Model#class
        """
        self.k = 0.1#7e-3 #cut ratio 0.2
        self.min_cen = 7 #3 - for generating asteroid
        self.max_cen = 13 #10

        if random == False and mode == 'Rxyz_assign':
            self.radi = args[0]
            self.x1 = args[1]
            self.y1 = args[2]
            self.z1 = args[3]

            self.r_cen, self.phi_cen, self.theta_cen = utils.cart2sph((self.x1, self.y1, self.z1))
            return
        
        elif random == False and mode == 'ratio_assign':
            self.phi_cen = 2*np.pi*args[0]
            self.theta_cen = np.pi*args[1]
            self.r_cen_ratio = args[2]
            self.R_cut_ratio = args[3]
        
        else:
            self.phi_cen = 2*np.pi*np.random.rand()
            self.theta_cen = np.pi*np.random.rand()

        self.j_cen = round(self.theta_cen/ast.dtheta)
        if self.j_cen%2 == 0:
            self.i_cen = round(self.phi_cen/ast.dphi)
        else:
            self.i_cen = round((self.phi_cen-ast.dphi/2)/ast.dphi)
        self.r_ast = ast.pos_sph_arr[self.i_cen, self.j_cen, 0]

        if random == False and mode == 'ratio_assign':
            self.r_cen = (self.min_cen + (self.max_cen-self.min_cen)*self.r_cen_ratio)*self.r_ast
            self.radi = self.k*self.r_ast*self.R_cut_ratio + self.r_cen - self.r_ast
            
        else:
            self.r_cen = (self.min_cen + (self.max_cen-self.min_cen)*np.random.rand())*self.r_ast
            self.radi = self.k*self.r_ast*np.random.rand() + self.r_cen - self.r_ast

        self.x1, self.y1, self.z1 = utils.sph2cart([self.r_cen, self.phi_cen, self.theta_cen])


    def f(self, cart_pos):
        """
        Equation of Sphere
        cart_pos : cartesian position coord.
        """
        x = cart_pos[0]
        y = cart_pos[1]
        z = cart_pos[2]

        f = (x-self.x1)**2 + (y-self.y1)**2 + (z-self.z1)**2 - self.radi**2
        return f
    
    def r_f(self, angle_pos):
        """
        <input> angle_pos = given [phi, theta]
        <output> : r coord. corr the input (the point on surface of the sphere)
        """
        phi = angle_pos[0]
        theta = angle_pos[1]
        
        r_f_unit = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        r_cen_unit = np.array([np.sin(self.theta_cen)*np.cos(self.phi_cen), np.sin(self.theta_cen)*np.sin(self.phi_cen), np.cos(self.theta_cen)])
        cosa = np.dot(r_f_unit, r_cen_unit)

        r_f = self.r_cen*cosa - ((self.r_cen*cosa)**2 - (self.r_cen**2 - self.radi**2))**0.5
        return r_f
    
class AsteroidModel():
    def __init__(self, axes, N_set, tilt_mode="assigned", tilt=(0, 0)):
        self.Nphi = N_set[0]
        self.Ntheta = N_set[1]
        self.dphi = 2*np.pi/self.Nphi
        self.dtheta = np.pi/self.Ntheta
        self.pos_sph_arr = np.zeros((self.Nphi+1, self.Ntheta+1, 3)) #last index = first index (circular)
        self.pos_cart_arr = np.zeros((self.Nphi+1, self.Ntheta+1, 3)) #last index = first index 
        self.surf_vec_arr = np.zeros((self.Nphi, self.Ntheta, 2, 3))
    
        self.axes_R = np.array([axes[0], axes[1], axes[2]])
        self.tilt = np.array([tilt[0], tilt[1]])

        if tilt_mode == "random":
            self.tilt = np.pi*np.array([2*np.random.rand(1)[0], np.random.rand(1)[0]])
        elif tilt_mode == "assigned":
            pass
        else:
            raise ValueError("Unimplemented tilt_mode")
        
    # necessary calclutating functions
    def __circular(self, index):
        """
        for circular pos_arr
        index = 'i' : i-axis
              = 'j' : j-axis
              = 'all' : i-axis & j-axis
        """
        if index in ('i', 'all'):
            for j in range(self.Ntheta+1):
                self.pos_sph_arr[self.Nphi, j] = self.pos_sph_arr[0, j]
                self.pos_cart_arr[self.Nphi, j] = self.pos_cart_arr[0, j]
        
        if index in ('j', 'all'):
            for i in range(self.Nphi+1):
                self.pos_sph_arr[i, 0] = self.pos_sph_arr[0, 0]
                self.pos_cart_arr[i, 0] = self.pos_cart_arr[0, 0]

                self.pos_sph_arr[i, self.Ntheta] = self.pos_sph_arr[0, self.Ntheta]
                self.pos_cart_arr[i, self.Ntheta] = self.pos_cart_arr[0, self.Ntheta]


    # generating with basic frame
    def base_fitting_generator(self, mode="ellipsoid"):
        if mode == "ellipsoid":
            generating_frame = self.__ellipsoid_frame 

        for i in range(self.Nphi):
            for j in range(self.Ntheta+1):
                phi_ij = (j%2)*(self.dphi/2) + i*self.dphi
                theta_ij = j*self.dtheta
                r_ij = generating_frame([phi_ij, theta_ij])

                x_ij = r_ij*np.sin(theta_ij)*np.cos(phi_ij)
                y_ij = r_ij*np.sin(theta_ij)*np.sin(phi_ij)
                z_ij = r_ij*np.cos(theta_ij)
                
                self.pos_sph_arr[i, j] = np.array([r_ij, phi_ij, theta_ij])
                self.pos_cart_arr[i, j] = np.array([x_ij, y_ij, z_ij])

                if i == 0:
                    self.pos_sph_arr[self.Nphi, j] = np.array([r_ij, phi_ij, theta_ij])
                    self.pos_cart_arr[self.Nphi, j] = np.array([x_ij, y_ij, z_ij])
        

    def __ellipsoid_frame(self, direction, radi=[-1, -1, -1], tilt_angle=[-1, -1]):
        """
        ellipsoid generator
        a, b, c : radius corr. axis (default = axes_R)
        """
        
        a = self.axes_R[0] if radi[0] == -1 else radi[0]
        b = self.axes_R[1] if radi[1] == -1 else radi[1]
        c = self.axes_R[2] if radi[2] == -1 else radi[2]

        """
        tilt_angle = [longitude, latitude]
        * longitude angle : z-axis rotation
        * latitude angle : x-axis rotation
        """
        if tilt_angle[0] == -1:
            long = self.tilt[0]
        else:
            long = tilt_angle[0]
        if tilt_angle[1] == -1:
            lat = self.tilt[1]
        else:
            lat = tilt_angle[1]

        self.tilt = np.array([long, lat])
        long_rot_arr = utils.rotArr(-long, "z")
        lat_rot_arr = utils.rotArr(-lat, "y")
        R_arr = lat_rot_arr@long_rot_arr

        A_arr = np.array([[1/a**2, 0, 0],
                          [0, 1/b**2, 0],
                          [0, 0, 1/c**2]])
        
        """
        coordinate direction : [phi, theta]
        output : corr. r value
        """
        phi_temp = direction[0]
        theta_temp = direction[1]
        u_vec = np.array([np.sin(theta_temp)*np.cos(phi_temp), np.sin(theta_temp)*np.sin(phi_temp), np.cos(theta_temp)]).T
        r_temp = 1 / np.sqrt(u_vec.T@R_arr.T@A_arr@R_arr@u_vec)

        return r_temp
    
    def surf_vec_cal(self):
        for i in range(self.Nphi):
            for j in range(self.Ntheta):
                if j%2 == 0:
                    v11 = self.pos_cart_arr[i+1, j] - self.pos_cart_arr[i, j+1]
                    v12 = self.pos_cart_arr[i+1, j+1] - self.pos_cart_arr[i, j+1]
                    v21 = self.pos_cart_arr[i+1, j] - self.pos_cart_arr[i, j]
                    v22 = self.pos_cart_arr[i, j+1] - self.pos_cart_arr[i, j]
                elif j%2 == 1:
                    v11 = self.pos_cart_arr[i+1, j+1] - self.pos_cart_arr[i, j]
                    v12 = self.pos_cart_arr[i, j+1] - self.pos_cart_arr[i, j]
                    v21 = self.pos_cart_arr[i+1, j] - self.pos_cart_arr[i, j]
                    v22 = self.pos_cart_arr[i+1, j+1] - self.pos_cart_arr[i, j]
                self.surf_vec_arr[i, j, 0] = -0.5*np.cross(v11, v12)
                self.surf_vec_arr[i, j, 1] = -0.5*np.cross(v21, v22)
    
    def cut_ast(self, sph_num, pla_num, assigned=False, mode='Rxyz_assign', **kwargs):
        """
        cut asteroid with specific shape

        sph_num : cutting spherical num
        pla_num : cutting plane num
        """
        pos_sph = kwargs['pos_sph']
        self.__sph_cut(sph_num, assigned=not(assigned), mode=mode, pos_sph=pos_sph)

    def __sph_cut(self, sph_num, **kwargs):
        """
        cutting with sphere - CutterSphere#class
        """
        for k in range(sph_num):
            sph_temp = CutterSphere(self, kwargs['assigned'], kwargs['mode'], kwargs['pos_sph'][0], kwargs['pos_sph'][1], kwargs['pos_sph'][2], kwargs['pos_sph'][3])
            for i in range(self.Nphi+1):
                for j in range(self.Ntheta+1):
                    if sph_temp.f(self.pos_cart_arr[i, j]) < 0:
                        self.pos_sph_arr[i, j, 0] = sph_temp.r_f(self.pos_sph_arr[i, j, 1:])
                        self.pos_cart_arr[i, j] = utils.sph2cart(self.pos_sph_arr[i, j])

    def copy(self):
        ast_copy = AsteroidModel((1, 1, 1), (self.Nphi, self.Ntheta))
        ast_copy.pos_sph_arr = self.pos_sph_arr.copy()
        ast_copy.pos_cart_arr = self.pos_cart_arr.copy()
        ast_copy.surf_vec_arr = self.surf_vec_arr.copy()
        ast_copy.axes_R = self.axes_R.copy()
        ast_copy.tilt = self.tilt.copy()
        
        return ast_copy

class AstEnv():
    def __init__(self, target_lc:np.ndarray, lc_info:np.ndarray, reward_domain, N_set=(40, 20), lc_unit_len=200, ell_init=(False, False)):
        """
        target_lc
        lc_info : [sun_dir, earth_dir, rot_axis]
        lc_num
        obs_set = (obs_lc, obs_time) : what lc to obs, where to obs
        N_set = (Nphi, Ntheta) : asteroid model grid splitting number
        """
        self.lc_unit_len = lc_unit_len
        self.target_lc = target_lc
        self.lc_info = lc_info

        self.rot_axis = lc_info[-3:]
        self.initial_eps = np.empty(2)
        self.initial_eps[0] = np.arctan2(self.rot_axis[1], self.rot_axis[0])
        self.initial_eps[1] = np.arccos(self.rot_axis[2]/LA.norm(self.rot_axis))
        self.R_eps = utils.rotArr(-self.initial_eps[1], "y")@utils.rotArr(-self.initial_eps[0], "z")

        self.ast_obs_unit_step = 1 #2
        self.lc_obs_unit_step = 1 #2

        self.Nphi, self.Ntheta = N_set[0], N_set[1]
        self.dphi, self.dtheta = 2*PI/self.Nphi, PI/self.Ntheta
        
        self.reward_threshold = reward_domain[1] #70
        self.total_threshold = reward_domain[1] #70
        self.err_min = reward_domain[0]
        self.ell_err = False

        # Initialize asteroid
        self.reward0 = 999
        self.max_reward = -9e+8
        self.lc_pred = np.ones(self.lc_unit_len)
        self.ast_backup = None
        self.reset(True, ell_init)
        self.ast_backup = self.ast.copy()


    def orb2geo(self, vec_orb, rot_angle):
        return utils.rotArr(-rot_angle, "z")@self.R_eps@vec_orb
    
    def obs(self):
        #r_arr obs
        obs_r_arr_temp = self.ast.pos_sph_arr[:-1, :-1, 0].copy()
        obs_r_arr = obs_r_arr_temp[::self.ast_obs_unit_step, ::self.ast_obs_unit_step] + 0
        obs_r_arr = obs_r_arr.flatten()

        obs_tensor = np.concatenate((obs_r_arr, self.target_lc, self.lc_pred, self.lc_info[:6]))
        return obs_tensor
    
    def step(self, action, mode='ratio_assign', update=True):
        """
        action = [R_cut, r, phi, theta]
        if mode == 'ratio_assign'
            action = [phi, theta, r_cen_ratio, R_cut_ratio]
        """
        done = False
        passed = False
        if not (action[-1] == 0 and action[-2] == 0):
            if mode == 'ratio_assign':
                self.ast.cut_ast(1, 0, True, mode='ratio_assign', pos_sph=action)
            elif mode == 'coord_assign':
                cut_sph_pos = utils.sph2cart((action[1], action[2], action[3]))
                self.ast.cut_ast(1, 0, True, mode='Rxyz_assign', pos_sph=(action[0], cut_sph_pos[0], cut_sph_pos[1], cut_sph_pos[2]))
            else:
                raise NotImplementedError

        # Maintaining the radius mean of asteroid r_arr
        mean0 = 10
        r_arr_mean = np.mean(self.ast.pos_sph_arr[:, :, 0])
        self.ast.pos_sph_arr[:, :, 0] = self.ast.pos_sph_arr[:, :, 0] * mean0 / r_arr_mean
        self.ast.pos_cart_arr = self.ast.pos_cart_arr * mean0 / r_arr_mean

        self.ast.surf_vec_cal()

        reward = self.reward(init=100.0, relative=True)
        
        if reward > self.reward_threshold:
            done = True
            passed = True
        elif reward < self.max_reward - 3.5:#min(-4e+2, self.reward0):
            done = True
            passed = False

        observation = self.obs()

        if reward > self.max_reward and update:
            self.max_reward = reward + 0.0
            self.ast_backup = self.ast.copy()
            
        return observation, reward, done, passed
        
    def reward(self, init=100, relative=True):
        target_lc_temp = self.target_lc.copy()
        target_lc_mean = self.__lc_mean(target_lc_temp)

        lc_temp = self.__lc_gen(self.lc_info) #generate lc
        lc_temp = lc_temp * target_lc_mean / self.__lc_mean(lc_temp) #scaling lc_temp compared with target_lc_temp
        self.lc_pred[:] = lc_temp

        # Normalization for Loss Calculation
        target_lc_temp = target_lc_temp - target_lc_mean
        lc_temp = lc_temp - target_lc_mean
        
        if relative:
            amp = self.__amp_lc(target_lc_temp)
            loss = np.mean((80*(target_lc_temp - lc_temp)/amp)**2) #40

            loss_i = 60*np.trapezoid(np.abs(target_lc_temp-lc_temp))/(100*amp)
            loss_d = np.mean((40*(np.diff(target_lc_temp)-np.diff(lc_temp)))**2)
            #loss = (loss + loss_i + loss_d)*3/10
            loss = (1.2*loss + loss_i + loss_d)*2/10
        else:
            loss = np.mean((target_lc_temp - lc_temp)**2)
            
        return init - loss
    
    def reset(self, passed, ell_init=(False, False)):
        if self.ast_backup == None:
            max_try = 20 # original : 5, this is changed value for Ellipsoid_Approx_Data 
            for i in range(max_try+1):
                if ell_init[0]:
                    ell_arr = ell_init[1]
                    self.R_set = ell_arr[:3]
                    self.tilt = ell_arr[3:]
                else:
                    raise NotImplementedError
                self.ast = AsteroidModel(axes=self.R_set, N_set=(self.Nphi, self.Ntheta), tilt_mode="assigned", tilt=self.tilt)
                self.ast.base_fitting_generator(mode="ellipsoid")
                self.lc_pred = np.ones(self.lc_unit_len)
                _, self.reward0, _, _ = self.step((0, 0, 0, 0)) #initialize/recalculate lc_pred
                
                if self.reward0 > self.err_min and self.reward0 < self.total_threshold:#self.err_min+30:
                    break

                if i == max_try:
                    self.ell_err = True
        else:
            if not passed:
                self.ast = self.ast_backup.copy()
            self.step((0, 0, 0, 0)) #initialize/recalculate lc_pred

        return self.obs()

    def __lc_gen(self, lc_info, flux0=10):
        Sdir = lc_info[0:3]
        Edir = lc_info[3:6]
        #rot_axis = lc_info[6:9]
        N_arr = self.ast.surf_vec_arr / np.sqrt(np.abs(self.ast.surf_vec_arr)+1e-15)
        N_arr = N_arr.reshape(-1, 3)

        generated_lc = np.zeros(self.lc_unit_len)
        for t in range(self.lc_unit_len):
            theta_t = 2*PI*t/self.lc_unit_len
            Edir_t = self.R_eps.T@self.orb2geo(Edir.T, theta_t) #Edir(0) -> Edir(t)    ### 여기 R_eps 검토하기!) 했음 25.11.17
            Sdir_t = self.R_eps.T@self.orb2geo(Sdir.T, theta_t) #Sdir(0) -> Sdir(t)    ###
            Edir_t = Edir_t / LA.norm(Edir_t)
            Sdir_t = Sdir_t / LA.norm(Sdir_t)
            generated_lc[t] = utils.ReLU(N_arr@Edir_t).T@utils.ReLU(N_arr@Sdir_t)
        generated_lc = flux0 * generated_lc

        return generated_lc
    
    # Utils for LC-Related Calculation
    def __lc_mean(self, input_lc):
        """
        input_lc = [LC Length]
        """
        lc_len = input_lc.shape[-1]
        lc_mean0 = (np.sum(input_lc, axis=-1) - (input_lc[..., 0] + input_lc[..., -1])/2) / lc_len
        return lc_mean0
    
    def __amp_lc(self, input_lc):
        lc_max = np.max(input_lc)
        lc_min = np.min(input_lc)
        return lc_max - lc_min
    
    def show(self, path, name="None"):
        fig = plt.figure(figsize=(13, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        lim_set = (-10, 10)

        gridX = self.ast.pos_cart_arr[:, :, 0]
        gridY = self.ast.pos_cart_arr[:, :, 1]
        gridZ = self.ast.pos_cart_arr[:, :, 2]

        ax1.plot(self.target_lc, color='coral', linestyle='solid') #black
        ax1.plot(self.lc_pred, color='coral', linestyle='dashed')
        ax1.set_ylim([np.min(self.target_lc)-5, np.max(self.target_lc)+5])
        ax1.set_title("Lightcurve")

        ax2.set_box_aspect((1, 1, 1))
        ax2.set_xlim(lim_set)
        ax2.set_xlabel('X')
        ax2.set_ylim(lim_set)
        ax2.set_ylabel('Y')
        ax2.set_zlim(lim_set)
        ax2.set_zlabel('Z')
        ax2.set_title("Predicted Model")

        ax2.plot_surface(gridX, gridY, gridZ)
        
        plt.savefig(path+name)
        #plt.show()

class Runner():
    def __init__(self, env:AstEnv, state_dim, action_dim, prec = 5.0):
        self.env = env
        self.done = True
        self.passed = False
        self.prec = prec
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.data_set_arr = np.zeros((1, self.state_dim+self.action_dim+2))

    def reset(self, passed):
        self.state = self.env.reset(passed)
        self.done = False
        self.passed = False

    def run(self, env_no, random=True, save=False):
        prec = self.prec

        #30
        get_env_num = int(20 * (self.env.reward_threshold - max(self.env.reward0, 0)) / self.env.reward_threshold)#int(30 * (self.env.reward_threshold - self.env.reward0) / self.env.reward_threshold)
        get = 0
        reward_threshold_list = np.linspace(max(self.env.reward0, 0), self.env.reward_threshold, get_env_num)
        if reward_threshold_list.shape[0] == 0:
            return
        
        show_bool = True
        with tqdm(total=get_env_num, desc="Reward Map Generation") as pbar:
            for t in range(MAX_STEPS):
                if self.done:
                    self.reset(self.passed)

                actions = np.random.uniform(-prec, prec, 4)
                actions[:2] = np.mod(actions[:2], 1.0)
                actions[2:] = 0.1

                self.state, reward, self.done, self.passed = self.env.step(actions)

                if reward >= reward_threshold_list[get]:
                    self.make_map(reward, env_no, random, save)
                    get += 1
                    pbar.update(1)

                if get == get_env_num: break

                if t%4 == 0 and get == 0:
                    #print(" | Reward : {:7.5g}".format(reward), end='')
                    #print(" | actions : [{:6.05g}, {:6.05g}, {:6.05g}, {:6.05g}]".format(actions[0], actions[1], actions[2], actions[3]), end='')
                    #print(" | done/pass : "+str(self.done)+"/"+str(self.passed)+" "+str(self.env.max_reward))
                    if t%20 == 0:
                        show_bool = True

                if show_bool:
                    #print("show_passed : "+str(reward)+" | obs_lc_num : "+str(self.env.obs_lc_num))
                    #self.env.show(str(data_num)+"_"+str(et)+"_"+str(k)+"_"+str(int(reward*100)/100)+"_"+"0402ast.png")
                    #plt.close()
                    show_bool = False

                if self.done and self.passed: break
                
    def make_map(self, ref_reward, env_no, random, save):
        ratio_action_set = [(0.1, 0.1)]
        ref_ast = self.env.ast.copy()
        resol = 1

        rot_axis = self.env.initial_eps * 180/np.pi
        rot_axis[0] = rot_axis[0]%360
        rot_axis[1] = rot_axis[1]%180

        path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/reward_maps/"

        for ratio_actions in ratio_action_set:
            delta_map_temp = np.zeros((resol*self.env.Nphi, resol*self.env.Ntheta))
            #print("\nGenerating Map... (at Reward="+str(int(ref_reward*100)/100)+")", end='')
            for idx in range(self.env.Nphi*self.env.Ntheta*resol*resol):
                i = idx//int(resol*self.env.Ntheta)
                j = idx%int(resol*self.env.Ntheta)

                if random:
                    phi_action = (i/(resol*self.env.Nphi) + np.random.normal(0, 0.05, 1)[0])%1
                    theta_action = (j/(resol*self.env.Ntheta) + np.random.normal(0, 0.05, 1)[0])%1
                else:
                    phi_action = (i/(resol*self.env.Nphi))%1
                    theta_action = (j/(resol*self.env.Ntheta))%1
                actions = np.array([phi_action, theta_action, ratio_actions[0], ratio_actions[1]])
                
                _, reward, _, _ = self.env.step(actions, update=False)
                delta_map_temp[i, j] = reward - ref_reward
                self.data_set_arr = np.concatenate(
                    (self.data_set_arr, np.array([np.concatenate(
                            (self.state, actions, np.array([delta_map_temp[i, j], ref_reward]))
                        )])
                    ), axis=0
                )
                
                self.env.ast = ref_ast.copy()

            if save:
                phi_ticks = self.__map_tick_list(resol, self.env.Nphi, 360)
                theta_ticks = self.__map_tick_list(resol, self.env.Ntheta, 180)
        
                circle_points = 200
                Edirs = np.zeros((2, circle_points))
                Sdirs = np.zeros((2, circle_points))
                for t in range(circle_points):
                    Edir = self.env.R_eps.T@self.env.orb2geo((self.env.lc_info[3:6]).T, 2*np.pi*t/circle_points)   
                    Edirs[0, t] = (np.arctan2(Edir[1], Edir[0]) * 180/np.pi)%360
                    Edirs[1, t] = (np.arccos(Edir[2]/LA.norm(Edir)) * 180/np.pi)%180
                    Sdir = self.env.R_eps.T@self.env.orb2geo((self.env.lc_info[0:3]).T, 2*np.pi*t/circle_points)
                    Sdirs[0, t] = (np.arctan2(Sdir[1], Sdir[0]) * 180/np.pi)%360
                    Sdirs[1, t] = (np.arccos(Sdir[2]/LA.norm(Sdir)) * 180/np.pi)%180

                delta_map_temp = delta_map_temp.T
                
                grady, gradx = np.gradient(delta_map_temp)
                x = np.arange(delta_map_temp.shape[1])
                y = np.arange(delta_map_temp.shape[0])
                X, Y = np.meshgrid(x, y)


                plt.figure(figsize=(20, 11))
                plt.imshow(delta_map_temp)

                plt.plot(rot_axis[0]*resol*self.env.Nphi/360, rot_axis[1]*resol*self.env.Ntheta/180, color='red', marker='X', markersize=10)
                plt.plot(((rot_axis[0]+180)%360)*resol*self.env.Nphi/360, (180-rot_axis[1])*resol*self.env.Ntheta/180, color='blue', marker='X', markersize=10)
                
                for i in range(circle_points):
                    plt.plot(Edirs[0, i]*resol*self.env.Nphi/360, Edirs[1, i]*resol*self.env.Ntheta/180, color='blue', marker='.', markersize=6)
                    plt.plot(Sdirs[0, i]*resol*self.env.Nphi/360, Sdirs[1, i]*resol*self.env.Ntheta/180, color='red', marker='.', markersize=6)
                
                plt.colorbar()
                plt.quiver(X, Y, gradx, grady, color='gold', angles='xy', headwidth=2, headlength=4)
                name = "Env No."+str(env_no)+" (ref_reward="+str(int(ref_reward*1000)/1000)+") Delta+Grad Reward MAP"
                name_ratio = "(Ratio actions = ["+str(ratio_actions[0])+", "+str(ratio_actions[1])+"])"
                name_rot_axis = "(Rot_Axis = ["+str(int(100*rot_axis[0])/100)+", "+str(int(100*rot_axis[1])/100)+"])"
                plt.title(name+"\n"+name_ratio)
                plt.xticks(phi_ticks[0], phi_ticks[1])
                plt.yticks(theta_ticks[0], theta_ticks[1])
                plt.savefig(path+name+name_ratio+".png", dpi=300)
                plt.close()            

    def __map_tick_list(self, resol:int, N_ang:int, max_ang):
        tick_num = 12
        tick_value = []
        tick_label = []

        for i in range(tick_num):
            tick_value.append(i*(resol*N_ang)/tick_num)
            tick_label.append(str(i*int((max_ang*100)//tick_num)/100))

        return tick_value, tick_label
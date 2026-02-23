import numpy as np
import numpy.linalg as LA
import torch
from torch import nn
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import utils

import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                
                # 26.02.22 
                self.lc_pred0 = self.lc_pred.copy()

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
            ### orb -> geo로 갖고 와서 geo에서 계산해야 하는거 아님? (surf_vec_arr가 geo frame이잖아) 26.02.18
            #Edir_t = self.R_eps.T@self.orb2geo(Edir.T, theta_t) #Edir(0) -> Edir(t)    ### 여기 R_eps 검토하기!) 했음 25.11.17
            #Sdir_t = self.R_eps.T@self.orb2geo(Sdir.T, theta_t) #Sdir(0) -> Sdir(t)    ###
            Edir_t = self.orb2geo(Edir.T, theta_t) #Edir(0) -> Edir(t)
            Sdir_t = self.orb2geo(Sdir.T, theta_t) #Sdir(0) -> Sdir(t)
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
    
    def show(self, res_params, path, name="None"):
        # ---------- res_params : parameters to plot, calculated from running ----------
        reward_list = res_params[0]
        reward_map = res_params[1]
        best_action = res_params[2]
        sel_actions = res_params[3]
        K = res_params[4]

        # ---------- Internal Parameter Processing ----------
        r_arr = self.ast.pos_sph_arr[:-1, :-1, 0]

        Sdir = self.lc_info[0:3]
        Edir = self.lc_info[3:6]
        Stheta = np.arccos(Sdir[-1]) * 20 / np.pi
        Etheta = np.arccos(Edir[-1]) * 20 / np.pi

        # ---------- Main Plotting Part ----------
        fig = plt.figure(figsize=(15, 10), dpi=150)
        ax1 = fig.add_subplot(2, 3, 1)                      # LC
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')     # Asteroid (View1)
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')     # Asteroid (View2)
        ax4 = fig.add_subplot(2, 3, 4)                      # t-reward graph
        ax5 = fig.add_subplot(2, 3, 5)                      # r_arr
        ax6 = fig.add_subplot(2, 3, 6)                      # (predicted) reward_map + selected region
        
        lim_set = (-10, 10)
        view1 = (30, -60)
        view2 = (-30, 120)

        gridX = self.ast.pos_cart_arr[:, :, 0]
        gridY = self.ast.pos_cart_arr[:, :, 1]
        gridZ = self.ast.pos_cart_arr[:, :, 2]

        # ax1 : Lightcurves
        ax1.plot(self.target_lc, color='coral', linestyle='solid', label='target') #black
        ax1.plot(self.lc_pred, color='coral', linestyle='dashed', label='pred.')
        ax1.plot(self.lc_pred0, color='gray', alpha=0.3, linestyle='dotted')
        ax1.set_title("Lightcurve (Reward = %.2f)"%(reward_list[-1]))
        ax1.legend()
        ax1.set_ylim([np.min(self.target_lc)-5, np.max(self.target_lc)+5])

        # ax2 : Asteroid Polyhedron (View1)
        self._plotAsteroid(ax2, gridX, gridY, gridZ,
                           elev=view1[0], azim=view1[1], lim_set=lim_set)
        
        # ax3 : Asteroid Polyhedron (View2)
        self._plotAsteroid(ax3, gridX, gridY, gridZ,
                           elev=view2[0], azim=view2[1], lim_set=lim_set)

        # ax4 : t - reward Graph
        t_last = len(reward_list)
        ax4.plot(np.arange(t_last), reward_list, color='royalblue')
        ax4.plot((0, t_last-1), (self.total_threshold, self.total_threshold), color='gray', alpha=0.3, linestyle='dotted')
        ax4.set_title("t - reward Graph")
        ax4.set_xlim((-1, max(15+1, t_last+1)))
        ax4.set_xlabel('t')
        ax4.set_ylabel('reward')

        # ax5 : r_arr
        r_arr_img = ax5.imshow(r_arr.T, vmax=8, vmin=12)
        ax5.set_title("r_arr")
        plt.colorbar(r_arr_img, ax=ax5, shrink=0.5)
        ax5.plot([0, 39], [Stheta, Stheta], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
        ax5.plot([0, 39], [Etheta, Etheta], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
        ax5.legend()

        # ax6 : (Predicted) Reward Map + Selected Actions
        reward_map_img = ax6.imshow(reward_map.T, vmax=np.max(np.abs(reward_map)), vmin=-np.max(np.abs(reward_map)))
        ax6.scatter(40*sel_actions[:, 0], 20*sel_actions[:, 1], s=80, facecolors='none', edgecolors='r')
        ax6.scatter(40*best_action[0], 20*best_action[1], marker='*', color='r')
        ax6.set_title("(Predicted) Reward Map + Selected Actions (K=%d)"%(K))
        plt.colorbar(reward_map_img, ax=ax6, shrink=0.5)
        self._setRewardMapPlot(ax=ax6, Etheta=Etheta, Stheta=Stheta)
        
        plt.savefig(path+name)
        #plt.show()
        plt.close()

    def _plotAsteroid(self, ax:plt.Axes, gridX, gridY, gridZ, elev=30, azim=-60, lim_set=(-10, 10)):
        ax.plot_surface(gridX, gridY, gridZ)
        ax.set_title("Asteroid (Elev.=%ddeg, Azim.=%ddeg)"%(elev, azim))
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(lim_set)
        ax.set_xlabel('X')
        ax.set_ylim(lim_set)
        ax.set_ylabel('Y')
        ax.set_zlim(lim_set)
        ax.set_zlabel('Z')

    def _setRewardMapPlot(self, ax:plt.Axes, Etheta, Stheta):
        """
        Draw optional informations for reward_map type plotting to ax
        """
        ax.plot([ 0, 40], [Etheta, Etheta], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
        ax.plot([ 0, 40], [Stheta, Stheta], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
        ax.plot([ 0, 40], [ 0,  0], color='gold', linewidth=0.8, linestyle='dotted')
        ax.plot([ 0, 40], [20, 20], color='gold', linewidth=0.8, linestyle='dotted')
        ax.plot([ 0,  0], [ 0, 20], color='gold', linewidth=0.8, linestyle='dotted')
        ax.plot([40, 40], [ 0, 20], color='gold', linewidth=0.8, linestyle='dotted')
        ax.set_xlim([0-0.5, 40-0.5])
        ax.set_ylim([20-0.5, 0-0.5])

# Updated for preserving spatial structure with 1x1 conv
class QValueNet_CNN_B1(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, activation=nn.ReLU, dropout=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # R_arr encoders (input: [B, C, 40, 20])
        self.r_arr_encoder1 = nn.Sequential(
            nn.Conv2d(1, 8, (9, 5)),  # -> [B, 8, 40, 20] # 1 channel / assumed input is already done padding=1 #(1, 16, 3)
            self.activation(),
            #nn.MaxPool2d(2)  # -> [B, 8, 20, 10] #deleted (260108) : to remain size 40x20
        )

        self.r_arr_encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, (5, 3)),  # assumed input is already done padding=1 #(16, 32, 3)
            self.activation(), # -> [B, 16, 20, 10] # -> [B, 16, 40, 20]

            # for preserving spatial structure, using size 1 kernal instead MLP
            nn.Conv2d(16, 64, 1), # -> [B, 64, 20, 10] # -> [B, 16, 40, 20]
            self.activation(),
            #nn.AdaptiveAvgPool2d(1) # -> [B, 64, 1, 1]#deleted (260108) : to remain size 40x20
        )

        # Info encoder (input: [B, 1, 6])
        self.info_encoder = nn.Sequential(
            nn.Linear(6, 32),
            self.activation(),
            nn.Linear(32, 64) # -> [B, 1, 64]
        )

        # RL encoder (input: [B, 1, 4])
        self.rl_encoder = nn.Sequential(
            nn.Linear(4, 32),
            self.activation(),
            nn.Linear(32, 64) # -> [B, 1, 64]
        )

        # Lightcurves encoder (input: [B, 2, 100])
        self.lc_encoder1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=15),
            self.activation(),
            nn.MaxPool1d(2),   # -> [B, 16, 50]
        )

        self.lc_encoder2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=9),
            self.activation(), # -> [B, 32, 50]

            nn.Conv1d(32, 64, 1), # -> [B, 64, 50]
            self.activation(),
            nn.AdaptiveAvgPool1d(8), # -> [B, 64, 8]

            nn.Flatten(1, -1), # -> [B, 512]
            nn.Linear(64*8, 64), # -> [B, 64]
            self.activation(),
            nn.Dropout(dropout)
        )

        # Fusion & Head
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 64, self.hidden_dim),
            self.activation(),
            nn.Dropout(dropout),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation(),
            nn.Dropout(dropout),

            nn.Linear(self.hidden_dim, 256),
            self.activation(),
            nn.Dropout(dropout),

            nn.Linear(256, 1)  # e.g., class count or regression value
        )

    def r_padding(self, x, pad=(1, 1)):
        N, C, H, W = x.shape
        pad_H = pad[0]
        pad_W = pad[1]

        out = torch.full((N, C, H + 2*pad_H, W + 2*pad_W), fill_value=0.0, dtype=x.dtype, device=x.device)
        out[:, :, pad_H:pad_H+H, pad_W:pad_W+W] = x
        out[:, :, :, :pad_W] = torch.roll(torch.flip(out[:, :, :, pad_W:pad_W+pad_W], (-2,)), 20, -1)
        out[:, :, :, -pad_W:] = torch.roll(torch.flip(out[:, :, :, -pad_W-pad_W:-pad_W], (-2,)), 20, -1)
        out[:, :, :pad_H, pad_W:pad_W+W] = x[:, :, -pad_H:, :]
        out[:, :, -pad_H:, pad_W:pad_W+W] = x[:, :, :pad_H, :]
        return out

    def lc_padding(self, x, pad=1):
        N, C, W = x.shape

        out = torch.full((N, C, W + 2*pad), fill_value=0.0, dtype=x.dtype, device=x.device)
        out[:, :, pad:pad+W] = x
        out[:, :, :pad] = x[:, :, -pad:]
        out[:, :, -pad:] = x[:, :, :pad]
        return out

    def shifter(self, img, dx=0, dy=0):
        PI = 3.14159265358979
        img_F = torch.fft.fft2(img)
        N, M = img.shape
        dev = img.device

        ky = torch.fft.fftfreq(N, device=dev)[:, None]
        kx = torch.fft.fftfreq(M, device=dev)[None, :]
        phase = torch.exp(-2j*PI*(kx*dx + ky*dy))
        new_img = torch.fft.ifft2(img_F*phase)
        return new_img.real

    def sphere_latlon_tensor(self, lon, lat, Nlon=40, Nlat=20):
        # Added w/ GPT (26.01.11)
        lon = lon.long()
        lat = lat.long()

        # lon은 항상 주기 wrap
        lon = torch.remainder(lon, Nlon)

        half = Nlon // 2
        m1 = lat < 0
        m2 = (~m1) & (lat >= Nlat)

        lat = torch.where(m1, -lat, lat)
        lon = torch.where(m1, lon + half, lon)

        lat = torch.where(m2, 2*(Nlat-1) - lat, lat)
        lon = torch.where(m2, lon + half, lon)

        # 보정 후에도 다시 wrap
        lon = torch.remainder(lon, Nlon)
        return lon, lat


    def r_a_gather(self, r_arr_feat, lon, lat, size=3):
        # Added w/ GPT (26.01.11)
        if size%2 == 0: raise ValueError("size must be odd")

        B = r_arr_feat.shape[0]
        b_idx = torch.arange(B, device=r_arr_feat.device)

        r_a_elems = []
        for i in range(-size//2, size//2+1):
            for j in range(-size//2, size//2+1):
                lon_temp, lat_temp = self.sphere_latlon_tensor(lon+i, lat+j)
                r_a_elems.append(r_arr_feat[b_idx, :, lon_temp, lat_temp])
        #r_a = torch.cat(r_a_elems, dim=1)
        r_a = r_a_elems[0]
        for i in range(1, len(r_a_elems)): r_a = r_a + r_a_elems[i]
        r_a = r_a / (size**2)

        return r_a

    def forward(self, X):
        if X.dim() == 3 and X.size(1) == 1:
            X = X.squeeze(1)  # [B, input_dim]
        PI = 3.14159265358979

        r_arr = X[..., :800].reshape((X.shape[0], 1, 40, 20))
        lc_target = X[..., 800:900].reshape((X.shape[0], 1, 100))
        lc_pred = X[..., 900:1000].reshape((X.shape[0], 1, 100))
        lc_info = X[..., 1000:1006]
        rl_info = X[..., 1006:]

        #r_arr_feat = torch.transpose(r_arr, -2, -1)
        #r_arr_feat = self.r_padding(r_arr_feat, pad=(4, 2))
        r_arr_feat = self.r_padding(r_arr, pad=(4, 2))
        r_arr_feat = self.r_arr_encoder1(r_arr_feat)
        r_arr_feat = self.r_padding(r_arr_feat, pad=(2, 1))
        r_arr_feat = self.r_arr_encoder2(r_arr_feat)
        #r_arr_feat = torch.squeeze(r_arr_feat, dim=-1)
        #r_arr_feat = torch.squeeze(r_arr_feat, dim=-1)

        ##############################################################
        # select action coord. from feature map (GPT, 260108)
        lon_raw = rl_info[..., 0]  # [B]  (아직 embedding 전, 0~1 가정)
        lat_raw = rl_info[..., 1]  # [B]

        lon_idx = torch.floor(lon_raw * 40).clamp(0, 39).long()
        lat_idx = torch.floor(lat_raw * 20).clamp(0, 19).long()

        # r_arr_feat가 [B, 64, 40, 20]일 때:
        B = r_arr_feat.shape[0]
        b_idx = torch.arange(B, device=r_arr_feat.device)
        r_a = r_arr_feat[b_idx, :, lon_idx, lat_idx]     # [B, 64]
        #r_a = self.r_a_gather(r_arr_feat, lon_idx, lat_idx, size=3) # [B, 64]
        ##############################################################

        lc = torch.cat([lc_target, lc_pred], dim=1)
        lc_feat = self.lc_padding(lc, pad=7)          # [B, 2, 114]
        lc_feat = self.lc_encoder1(lc_feat)           # [B, 16, 50]
        lc_feat = self.lc_padding(lc_feat, pad=4)     # [B, 16, 58]
        lc_feat = self.lc_encoder2(lc_feat)           # [B, 64, 1]
        #lc_feat = torch.squeeze(lc_feat, dim=-1)      # [B, 64]

        info_feat = self.info_encoder(lc_info)
        #info_feat = torch.squeeze(info_feat, dim=1)

        # action direction embedding
        lon_raw = rl_info[..., 0]  # [B]  (아직 embedding 전, 0~1 가정)
        lat_raw = rl_info[..., 1]  # [B]
        action_emb = torch.stack([
            torch.sin(2*PI*lon_raw),
            torch.cos(2*PI*lon_raw),
            torch.sin(PI*lat_raw),
            torch.cos(PI*lat_raw),
        ], dim=1)  # [B,4]
        rl_feat = self.rl_encoder(action_emb)
        #rl_info = torch.unsqueeze(rl_info, dim=1)
        #rl_feat = self.rl_encoder(rl_info)
        #rl_feat = torch.squeeze(rl_feat, dim=1)

        #fusion_feat = torch.cat((r_arr_feat, lc_feat, info_feat, rl_feat), dim=1)
        fusion_feat = torch.cat((r_a, lc_feat, info_feat, rl_feat), dim=1)
        out = self.head(fusion_feat)
        #shift_out = self.shift_head(fusion_feat)

        #self.x_shift = torch.unsqueeze(shift_out[..., 0], dim=1)
        #self.y_shift = torch.unsqueeze(shift_out[..., 1], dim=1)

        #out = self.shifter(out, dx=20*self.x_shift, dy=10*self.y_shift)

        out = 6 * 2 / PI * torch.atan(out/0.8) #out/0.8
        #out = 7 * 2 / PI * torch.atan(1.5 * out)

        return out

class AgentRunner():
    def __init__(self, env:AstEnv, model:QValueNet_CNN_B1):
        self.env = env
        self.done = True
        self.passed = False

        self.model = model

    def reset(self, passed):
        self.state = self.env.reset(passed)
        self.reward = self.env.reward0
        self.done = False
        self.passed = False

    def input_data(self, state):
        # GPU 최적화 필요 !!!!!!!!!!!!!!
        input_list = []
        for idx in range(800):
            i = idx//int(20)
            j = idx%int(20)
            phi_action = (i/40)%1
            theta_action = (j/20)%1
            actions = np.array([phi_action, theta_action, 0.1, 0.1])
            input = torch.tensor(np.concatenate((state, actions))).float().to(device)
            input_list.append(torch.unsqueeze(input, 0))
        total_input = torch.concat(input_list, dim=0)
        return total_input
    
    def action_selector(self, pred_map):
        # NEED TO IMPLEMENT
        
        ####################
        # TOP-K Selector
        ####################

        # pred_map : 20*40
        self.K = 20
        idxs = np.argsort(pred_map.reshape(-1))[::-1][:self.K]
        actions = np.stack([np.array([((idx%int(40))/40)%1, ((idx//int(40))/20)%1, 0.1, 0.1]) for idx in idxs], axis=0)

        test_rewards = np.zeros((self.K))
        ref_ast = self.env.ast.copy()
        for i in range(self.K):
            _, reward, _, _ = self.env.step(actions[i, :], update=False)
            self.env.ast = ref_ast.copy()
            test_rewards[i] = reward + 0
        
        if np.max(test_rewards) < self.reward: return None, actions
        return actions[np.argmax(test_rewards), :], actions
        
    def run(self, env_i, save_path):
        ref_ast = self.env.ast.copy()
        self.env.ast = ref_ast.copy()

        self.reset(self.passed)
        reward_list = []
        for t in tqdm(range(MAX_STEPS)):
            if self.done:
                if self.passed:
                    print("PASSED")
                    break
                #else: print("Did not converged to valid solution.")
                
            
            pred = np.zeros((20, 40))
            self.model.eval()
            with torch.no_grad():
                input = self.input_data(self.state[:1006])
                rewards = self.model(input)
                pred = rewards.cpu().numpy().reshape(40, 20).T

            best_action, actions = self.action_selector(pred)
            if best_action is None:
                print("No Improving Action Detected")
                break
            self.state, self.reward, self.done, self.passed = self.env.step(best_action)

            reward_list.append(self.reward)
            if t%1 == 0:
                self.env.show((reward_list, pred.T, best_action, actions, self.K), path=save_path, name='Env No.%02d t = %02d.png'%(env_i, t))

        print("Reward Change : %.4f -> %.4f"%(reward_list[0], reward_list[-1]))       
    
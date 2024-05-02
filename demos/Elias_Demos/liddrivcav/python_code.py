""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""
import math
import os

import numpy as np

from phi.jax.flow import *

math.set_global_precision(64)


def tgv_F(vis, t):
    return math.exp(-2 * vis * t)


def tgv_velocity(x, vis=0, t=0):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([cos.vector['x'] * sin.vector['y'] * tgv_F(vis, t),
                       - sin.vector['x'] * cos.vector['y'] * tgv_F(vis, t)], dim=channel('vector'))


def tgv_pressure(x, vis=0, t=0):
    return -1 / 4 * (math.sum(math.cos(2 * x), 'vector')) * (tgv_F(vis, t) ** 2)

def plane_poisseuille(x, vis, p_grad, h):
    return p_grad/(2*vis) * x*(h-x)

def plane_poisseuille_init(x, vis=1, p_grad=1, h=1):
    real_poisseuille = math.stack([p_grad/(2*vis) * x.vector['y']*(h-x.vector['y']), x.vector['y']*0], dim=channel(vector='x,y'))
    return real_poisseuille


class TestRun:

    def __init__(self, tges, gridtype, order, xynum, p_grad, vis, dt, name='data'):

        if order == 'phi':
            self.adv_diff_press = self.adp_phi_flow
            self.pressure_treatment = self.pt_phi_flow
        elif order == 'low':
            self.adv_diff_press = self.adp_low_ord
            self.pressure_treatment = self.pt_low_ord
        elif order == 'mid':
            self.adv_diff_press = self.adp_mid_ord
            self.pressure_treatment = self.pt_mid_ord
        if order == 'high':
            self.adv_diff_press = self.adp_high_ord
            self.pressure_treatment = self.pt_high_ord
        if order == 'high_impl':
            self.adv_diff_press = self.adp_high_ord_impl
            self.pressure_treatment = self.pt_high_ord_impl

        self.order = order
        self.name = name
        self.t_num = int(math.ceil(tges / dt))
        self.gridtype = gridtype
        self.timestep = self.fourth_ord_runge_kutta
        self.xynum = xynum
        self.p_grad = p_grad
        self.vis = vis
        self.dt = dt

    def phi_flow(self, velocity, pressure):
        velocity = advect.semi_lagrangian(velocity, velocity, self.dt)
        velocity, pressure = fluid.make_incompressible(velocity, solve=Solve('CG-adaptive', 1e-5, 0, x0=pressure))
        velocity = diffuse.explicit(velocity, 0.1, self.dt)
        return velocity, 0


    def fourth_ord_runge_kutta(self, velocity, pressure):
        dt = self.dt

        v_1, p_1 = velocity, pressure
        rhs_1 = self.adv_diff_press(v_1, p_1)
        v_2_old = velocity + (dt / 2) * rhs_1
        v_2, p_2 = self.pressure_treatment(v_2_old, p_1, dt / 2)
        # vis.plot(v_2_old.vector['x'], v_2_old.vector['y'], title=f'vel 1 old')
        # vis.show()
        # vis.plot(v_2.vector['x'], v_2.vector['y'], title=f'vel 1')
        # vis.show()
        # vis.plot(p_2, title=f'press 1')
        # vis.show()


        rhs_2 = self.adv_diff_press(v_2, p_2)
        v_3_old = velocity + (dt / 2) * rhs_2
        v_3, p_3 = self.pressure_treatment(v_3_old, p_2, dt / 2)
        # vis.plot(v_3_old.vector['x'], v_3_old.vector['y'], title=f'vel 2 old')
        # vis.show()
        # vis.plot(v_3.vector['x'], v_3.vector['y'], title=f'vel 2')
        # vis.show()
        # vis.plot(p_3, title=f'press 1')
        # vis.show()

        rhs_3 = self.adv_diff_press(v_3, p_3)
        v_4_old = velocity + dt * rhs_2
        v_4, p_4 = self.pressure_treatment(v_4_old, p_3, dt)
        # vis.plot(v_4_old.vector['x'], v_4_old.vector['y'], title=f'vel 3 old')
        # vis.show()
        # vis.plot(v_4.vector['x'], v_4.vector['y'], title=f'vel 3')
        # vis.show()
        # vis.plot(p_4, title=f'press 1')
        # vis.show()

        rhs_4 = self.adv_diff_press(v_4, p_4)
        v_p1_old = velocity + (dt / 6) * (rhs_1 + 2 * rhs_2 + 2 * rhs_3 + rhs_4)
        p_p1_old = (1 / 6) * (p_1 + 2 * p_2 + 2 * p_3 + p_4)
        v_p1, p_p1 = self.pressure_treatment(v_p1_old, p_p1_old, dt)
        # vis.plot(p_p1_old.vector['x'], p_p1_old.vector['y'], title=f'vel 4 old')
        # vis.show()
        # vis.plot(v_p1.vector['x'], v_p1.vector['y'], title=f'vel 4')
        # vis.show()
        # vis.plot(p_p1, title=f'press 1')
        # vis.show()

        return v_p1, p_p1

    def adp_high_ord_impl(self, v, p):
        adv_diff_press = (advect.finite_difference(v, v, order=6, implicit=Solve('scipy-GMres', 1e-12, 1e-12)))
        adv_diff_press += (diffuse.finite_difference(v, self.vis, order=6, implicit=Solve('scipy-GMres', 1e-12, 1e-12)))
        adv_diff_press -= field.spatial_gradient(p, type=self.gridtype, order=4, gradient_extrapolation=extrapolation.ZERO)
        return adv_diff_press

    def pt_high_ord_impl(self, v, p, dt_=None):
        if dt_ is None:
            dt_ = self.dt
        v, delta_p = \
            fluid.make_incompressible(v, order=6,
                                      solve=math.Solve('biCG-stab(2)', 1e-10, 1e-10))
        p += delta_p / dt_
        return v, p

    def adp_high_ord(self, v, p):
        adv_diff_press = (advect.finite_difference(v, v, order=6))
        diff = (diffuse.finite_difference(v, self.vis, order=6))
        adv_diff_press += diff
        # vis.plot(diff.vector['x'], diff.vector['y'], title=f'vel 4')
        # vis.show()
        adv_diff_press -= field.spatial_gradient(p, type=self.gridtype, order=6, gradient_extrapolation=extrapolation.ZERO)
        return adv_diff_press

    def pt_high_ord(self, v, p, dt_=None):
        if dt_ is None:
            dt_ = self.dt
        v, delta_p = \
            fluid.make_incompressible2(v, order=4,
                                      solve=math.Solve('biCG-stab(2)', 1e-10, 1e-10))
        p += delta_p / dt_
        return v, p


    def adp_mid_ord(self, v, p):
        adv = (advect.finite_difference(v, v, order=4))
        adv_diff_press = adv
        diff = (diffuse.finite_difference(v, self.vis, order=4))
        adv_diff_press += diff
        # vis.plot(adv.vector['x'], adv.vector['y'], title=f'vel 4')
        # vis.show()
        adv_diff_press -= field.spatial_gradient(p, type=self.gridtype, order=4, gradient_extrapolation=extrapolation.ZERO)
        return adv_diff_press

    def pt_mid_ord(self, v, p, dt_=None):
        if dt_ is None:
            dt_ = self.dt
        v, delta_p = \
            fluid.make_incompressible2(v,
                                      solve=math.Solve('biCG-stab(2)', 1e-10, 1e-10),
                                      order=4)
        p += delta_p / dt_
        return v, p


    def adp_low_ord(self, v, p):
        adv_diff_press = advect.finite_difference(v, v)
        diff = (diffuse.finite_difference(v, self.vis, order=2))
        adv_diff_press += diff
        # vis.plot(diff.vector['x'], diff.vector['y'], title=f'vel 4')
        # vis.show()
        adv_diff_press -= field.spatial_gradient(p, type=self.gridtype, gradient_extrapolation=extrapolation.ZERO)
        return adv_diff_press

    def pt_low_ord(self, v, p, dt_=None):
        if dt_ is None:
            dt_ = self.dt
        v, delta_p = \
            fluid.make_incompressible2(v, solve=math.Solve('biCG-stab(2)', 1e-10, 1e-10)) # "scipy-GMres" wirft fehler "biCG-stab(2)" geht
        p += delta_p / dt_
        return v, p

    def run(self, jit_compile=True, t_num=0, freq=100, eps=0, recap=False):
        print(f"run {self.name}:")
        if recap:
            self.name += "_rec"

        while(True):
            try:
                os.mkdir(f"data/{self.name}")
                break
            except FileExistsError:
                self.name = self.name + '_'

        if jit_compile:
            timestepper = math.jit_compile(self.timestep)
        else:
            timestepper = self.timestep

        if t_num > 0:
            self.t_num = t_num

        DOMAIN = dict(bounds=Box['x,y', 0:1, 0:1], x=self.xynum, y=self.xynum,
                      extrapolation=extrapolation.combine_sides(
                          x=extrapolation.ZERO,
                          y=(extrapolation.ZERO, extrapolation.combine_by_direction(extrapolation.ZERO, extrapolation.ConstantExtrapolation(-1)))))

        DOMAIN2 = dict(bounds=Box['x,y', 0:1, 0:1], x=self.xynum, y=self.xynum,
                       extrapolation=extrapolation.ZERO_GRADIENT)

        if recap:
            post_name = self.name[:-4]
            data = np.load(f"data/{post_name}/data.npz")
            t_num = data['t_num'].item()
            freq = data['freq'].item()
            velocity = field.read(f"data/{post_name}/vel_{t_num}.npz")
            pressure = field.read(f"data/{post_name}/press_{t_num}.npz")
        else:
            velocity = self.gridtype(tensor([0, 0], channel(vector='x, y')), **DOMAIN)
            pressure = CenteredGrid(0, **DOMAIN2)

        # velocity = field.read(f"data/fast_ldc_ord_low_120_re_1000_dt0.0001/vel_7000.npz")
        # pressure = field.read(f"data/fast_ldc_ord_low_120_re_1000_dt0.0001/press_7000.npz")

        field.write(velocity, f"data/{self.name}/vel_{0}")
        field.write(pressure, f"data/{self.name}/press_{0}")
        print(f"timestep: {0} of {self.t_num}")
        div = field.divergence(velocity, order=4)
        div_mean = math.mean(math.abs(div.values)).numpy().max()
        print(f"div mean: ", div_mean)
        div_max = math.max(math.abs(div.values)).numpy().max()
        print(f"div max: ", div_max)


        # for i in range(3):
        #     velocity, pressure = fluid.make_incompressible2(velocity, order=4, solve=math.Solve('biCG-stab(2)', 1e-7, 1e-7))
        #
        #     vis.plot(velocity, pressure, title=f'vel and press {i}')
        #     vis.show()
        #     vis.plot(velocity.vector['x'], velocity.vector['y'], title=f'vel x and vel y {i}')
        #     vis.show()
        #     div = field.divergence(velocity, order=2)
        #     vis.plot(div, title=f'div {i}')
        #     vis.show()
        #     print(f"div {i}: ", math.mean(math.abs(div.values)))
        #     print(f"max div {i}: ", math.max(math.abs(div.values)))

        for i in range(1, self.t_num+1):

            if i % freq == 0:
                print(f"timestep: {i} of {self.t_num}")
                div = field.divergence(velocity, order=4)
                div_mean = math.mean(math.abs(div.values))
                print(f"div mean: ", div_mean)
                div_max = math.max(math.abs(div.values))
                print(f"div max: ", div_max)

                # vis.plot(velocity, pressure, title=f'vel and p after timestep {i}')
                # vis.show()
                # vis.plot(velocity.vector['x'], velocity.vector['y'], title=f'vel x and y after timestep {i}')
                # vis.show()

                velocity_new, pressure_new = timestepper(velocity, pressure)

                max = math.max(math.abs(velocity_new.values - velocity.values)) / (self.dt * math.max(math.abs(velocity_new.values)))
                max = max.numpy().max()

                velocity, pressure = velocity_new, pressure_new
                field.write(velocity, f"data/{self.name}/vel_{i}")
                field.write(pressure, f"data/{self.name}/press_{i}")

                print("steady state degree: ", max)
                if max < eps:
                    break

                if math.is_nan(velocity.values).any:
                    break

            else:
                velocity, pressure = timestepper(velocity, pressure)



        np.savez(f"data/{self.name}/data", t_num=i, dt=self.dt, visc=self.vis, freq=freq, p_grad=self.p_grad, max_steady_state_diff=max, div_mean=div_mean, div_max=div_max)

        print()

    def print_fail_status(self):
        data = np.load(f"data/{self.name}/data.npz")
        t_num = data['t_num'].item()
        vel_data = field.read(f"data/{self.name}/vel_{t_num}.npz")
        print(f"{self.name}: {math.any(math.is_nan(vel_data.values))}")
        print(data['max_steady_state_diff'].item())


    def draw_plots(self):
        os.mkdir(f"plots/{self.name}")

        data = np.load(f"data/{self.name}/data.npz")
        t_num = data['t_num'].item()
        dt = data['dt'].item()
        visc = data['visc'].item()
        freq = data['freq'].item()
        vel_data = [field.read(f"data/{self.name}/vel_{i}.npz") for i in range(0, t_num+1, freq)]
        press_data = [field.read(f"data/{self.name}/press_{i}.npz") for i in range(0, t_num+1, freq)]

        max_steady_state_diff = data['max_steady_state_diff'].item()
        div_mean = data['div_mean'].item()
        div_max = data['div_max'].item()
        print(f"{self.name}: \t mssd: {max_steady_state_diff}, \t dmean: {div_mean}, \t dmax: {div_max}")
        final_vel = vel_data[-1].values.numpy('x,y,vector')
        mid_cell = int(final_vel[0,:,0].size/2)
        # print(f"{self.name}: \t 0.5,0.5_x: {final_vel[mid_cell][mid_cell][0]}, \t 0.5,0.5_y: {final_vel[mid_cell][mid_cell][1]}")
        # print(f"{self.name}: \t x_vel_max: {np.max(final_vel[mid_cell,:,0])}, \t y_vel_max:  {np.max(final_vel[:,mid_cell,1])}, \t y_vel_min:  {np.min(final_vel[:,mid_cell,1])}")


        for i in range(int(t_num / freq)+1):
            t = tensor(i * freq * dt)
            # f1 = vis.plot(vel[i], press[i], title=f'{i}: vel, press')._obj
            # timestamp = '{:07.4f}'.format(float(t))
            # vis.savefig(f"plots/{self.name}/v_and_p_{timestamp}.jpg", f1)
            # vis.close()
            timestamp = '{:07.4f}'.format(float(t))
            f1 = vis.plot(vel_data[i], title=f'{i}: vel')._obj
            vis.savefig(f"plots/{self.name}/v_{timestamp}.jpg", f1)
            vis.close()
            f2 = vis.plot(press_data[i], title=f'{i}: press')._obj
            vis.savefig(f"plots/{self.name}/p_{timestamp}.jpg", f2)
            vis.close()
            f3 = vis.plot(vel_data[i].vector['x'], vel_data[i].vector['y'], title=f'{i}: vel fields')._obj
            vis.savefig(f"plots/{self.name}/v_fields_{timestamp}.jpg", f3)
            vis.close()


    def more_plots(self):
        os.mkdir(f"plots/{self.name}")

        vel_data = field.read(f"data/{self.name}/vel.npz")
        press_data = field.read(f"data/{self.name}/press.npz")
        data = np.load(f"data/{self.name}/data.npz")

        t_num = data['t_num'].item()
        dt = data['dt'].item()
        visc = data['visc'].item()
        freq = data['freq'].item()
        p_grad = data['p_grad'].item()
        # p_grad = 0.01


        # vel = unstack(vel_data, 'time')
        press = unstack(press_data, 'time')


        # last_vel = vel[-1].vector['x']
        avg_profiles = math.sum(vel_data.vector['x'].values, 'x') / vel_data.values.shape.only('x').size

        last_avg_profile = avg_profiles.time[-1]
        vel_grid_profile_sample = vel_data.vector['x'].time[0].x[0]
        last_avg_profile_points = vel_grid_profile_sample.points.vector['y']

        ana_sol = plane_poisseuille(last_avg_profile_points, visc, p_grad, vel_grid_profile_sample.bounds.size.vector['y'])
        f1 = vis.plot([last_avg_profile, ana_sol, last_avg_profile - ana_sol], same_scale=False, title="last_avg_profile, ana_sol, difference")._obj
        # vis.show()
        vis.savefig(f"plots/{self.name}/velprofile_anasol_error_plot.jpg", f1)
        vis.close()

        error_dev = math.abs(math.sum(avg_profiles - ana_sol, 'y')) / ana_sol.shape.size
        f2 = vis.plot(vec(time=math.range_tensor(spatial(time=error_dev.time.size)), errors=error_dev.time.as_spatial()),
                      same_scale=False, title="errors over time", log_dims='errors')._obj
        # vis.show()
        vis.savefig(f"plots/{self.name}/errors_over_time.jpg", f2)
        vis.close()

        print('done')


def overview_plot(names_block, block_names=None, title='', folder_name='overview_plots'):

    try:
        os.mkdir(f"plots/{folder_name}")
    except FileExistsError:
        pass

    comparison_lines_block = []
    steadiness_lines_block = []
    convergence_lines = []
    for names in names_block:

        comparison_lines = []
        steadiness_lines = []
        convergence_line = []
        convergence_res = []
        for name in names:
            vel_data = field.read(f"data/{name}/vel.npz")
            data = np.load(f"data/{name}/data.npz")

            vel = unstack(vel_data, 'time')
            visc = data['visc'].item()
            freq = data['freq'].item()
            p_grad = data['p_grad'].item()
            t_num = data['t_num'].item()
            dt = data['dt'].item()

            last_vel = vel[-1].vector['x']
            avg_profile = math.sum(last_vel.values, 'x') / last_vel.values.shape.only('y').size
            profile_points = vel[-1].vector['x'].x[0].points.vector['y']
            ana_sol = plane_poisseuille(profile_points, visc, p_grad, last_vel.bounds.size.vector['y'])

            comparison_error_curves = (((avg_profile - ana_sol)).numpy(), ((avg_profile - ana_sol)/ana_sol).numpy())
            x_vals = profile_points.numpy()
            comparison_lines.append([x_vals, comparison_error_curves])

            convergence_line.append((avg_profile - ana_sol).y[int(avg_profile.y.size/2)])
            convergence_res.append(avg_profile.y.size)

            middle_cell_time_dev = vel_data.vector['x'].values

            rate_of_change_avg = []
            rate_of_change_mid_point = []
            for i in range(1, middle_cell_time_dev.time.size):
                diff = (middle_cell_time_dev.time[i-1] - middle_cell_time_dev.time[i])/freq
                rate_of_change_avg.append(math.mean(diff**2)/math.mean(abs(middle_cell_time_dev.time[i])))
                rate_of_change_mid_point.append(diff.vector['x'].x[int(vel_data.x.size/2)].y[int(vel_data.y.size/2)])

            time_scaling = np.linspace(0, dt*t_num, len(rate_of_change_avg))
            steadiness_lines.append([time_scaling, (np.array(rate_of_change_mid_point), np.array(rate_of_change_avg))])

        convergence_lines.append([convergence_res, convergence_line])
        comparison_lines_block.append(comparison_lines)
        steadiness_lines_block.append(steadiness_lines)


    import matplotlib.pyplot as plt
    linestyles = ['-', '--', ':', '-.']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    def plt_line_block(data_block, title, extra_param=None):
        fig = plt.figure(figsize=(9, 6))
        ax = plt.subplot(title=title)

        for block_nr, lines in enumerate(data_block):
            for line_nr, line in enumerate(lines):
                if extra_param is not None:
                    error_line = line[1][extra_param]
                else:
                    error_line = line[1]
                ax.plot(line[0], error_line, label=f"ord: {block_names[block_nr] if block_names is not None else '/'} - res: {line[0].size}",
                        color=colors[line_nr], linestyle=linestyles[block_nr])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        ax.set_xlabel('y axis position')
        ax.set_ylabel('error')
        ax.set_yscale('log')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              ncol=1, fancybox=True, shadow=True)
        fig.savefig(f"plots/{folder_name}/{title}.jpg")
        plt.close(fig)

    plt_line_block(comparison_lines_block, title + " error comparison", 0)
    plt_line_block(comparison_lines_block, title + " error comparison relative", 1)
    plt_line_block(steadiness_lines_block, title + " steadiness mid point", 0)
    plt_line_block(steadiness_lines_block, title + " steadiness avg", 1)

    def plt_lines(data_block, title, extra_param=None):
        fig = plt.figure(figsize=(9, 6))
        ax = plt.subplot(title=title)

        for line_nr, line in enumerate(data_block):
            if extra_param is not None:
                error_line = line[1][extra_param]
            else:
                error_line = line[1]
            ax.plot(line[0], error_line, label=f"ord: {block_names[line_nr] if block_names is not None else '/'}",
                    color=colors[line_nr])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        ax.set_xlabel('y axis position')
        ax.set_ylabel('error')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              ncol=1, fancybox=True, shadow=True)
        fig.savefig(f"plots/{folder_name}/{title}.jpg")
        plt.close(fig)

    plt_lines(convergence_lines, title + " convergence course")

# for re in [100, 1000]:
# # for re in [100]:
#     for ord in ['mid', 'low', 'high']:
#     # for ord in ['mid']:
#         for res in [15, 31]:
#         # for res in [15]:
#         #     if ord == 'mid' and re == 100 and res == 15:
#         #         pass
#         #     else:
#             eps = 1e-6
#             test = TestRun(0, CenteredGrid, ord, res, 0.05, 1/re, 0.001, name=f"paperldclast_re{re}_ord{ord}_res{res}")
#             # test.run(t_num=200000, freq=1000, jit_compile=True, eps=eps)
#             test.draw_plots()
#
# for re in [100, 1000]:
#     for ord in ['mid', 'low', 'high']:
#         res = 47
#         eps = 1e-6
#         test = TestRun(0, CenteredGrid, ord, res, 0.05, 1/re, 0.001, name=f"paperldclast_re{re}_ord{ord}_res{res}")
#         # test.run(t_num=200000, freq=1000, jit_compile=True, eps=eps)
#         test.draw_plots()

eps = 1e-6

# for re in [1000]:
#     for ord in ["low", "mid", "high"]:
#         for res in [8, 15, 30, 60, 120, 240]:
#             test = TestRun(0, CenteredGrid, ord, res, 0.05, 1 / re, 0.0001,
#                            name=f"fast_ldc_ord_{ord}_{res}_re_{re}_dt0.0001")
#             # test.run(t_num=300000, freq=1000, jit_compile=True, eps=eps)
#             test.print_fail_status()

# for re in [1000]:
#     for ord in ["low", "mid", "high"]:
#         for res in [8, 15, 30, 60, 120, 240]:
#             test = TestRun(0, CenteredGrid, ord, res, 0.05, 1 / re, 0.0003,
#                            name=f"fast_ldc_ord_{ord}_{res}_re_{re}_dt0.0001")
#             # test.draw_plots()
#             # test.run(t_num=300000, freq=1000, jit_compile=True, eps=eps)
#             test.print_fail_status()



# resols = [31, 61, 121]
# re = 1000
# for ord in ['low', 'mid', 'high']:
#     for res in resols:
#         test = TestRun(0, CenteredGrid, ord, res, None, 1 / re, 0.001,
#                            name=f"new_try_{ord}_{res}")
#         test.run(t_num=300000, freq=1000, jit_compile=True, eps=eps)
#         test.draw_plots()
#         # test.print_fail_status()


# re = 1000
# res = 121
# for ord in ['low', 'high']:
#     for dt in [0.003, 0.001, 0.0003, 0.0001]:
#             test = TestRun(0, CenteredGrid, ord, res, None, 1 / re, dt,
#                                name=f"timestep_invest2_{ord}_{res}_dt_{dt}")
#             test.run(t_num=3000000, freq=1000, jit_compile=True, eps=eps)
#             test.draw_plots()
#             # test.print_fail_status()

resols = [31, 61, 121]
re = 1000
for ord in ['low', 'mid', 'high']:
# for ord in ['low']:
    for res in resols:
        test = TestRun(0, CenteredGrid, ord, res, None, 1 / re, 0.001,
                           name=f"new_try_{ord}_{res}")
        test.run(t_num=300000, freq=1, jit_compile=True, eps=eps, recap=True)
        test.draw_plots()
        # test.print_fail_status()

print('done')
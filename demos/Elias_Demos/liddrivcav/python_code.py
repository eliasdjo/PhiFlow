""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""
import math
import os

from phi.jax.flow import *

math.set_global_precision(64)


class TestRun:

    def __init__(self, tges, at, order, xynum, p_grad, vis, dt, name='data'):

        if order == 'low_phi':
            self.ord = 2
        elif order == 'low':
            self.ord = 20
        elif order == 'mid':
            self.ord = 4
        if order == 'high':
            self.ord = 6

        self.order = order
        self.name = name
        self.t_num = int(math.ceil(tges / dt))
        self.at = at
        self.timestep = self.fourth_ord_runge_kutta
        self.xynum = xynum
        self.p_grad = p_grad
        self.vis = vis
        self.dt = dt

        self.diver = 0


    def fourth_ord_runge_kutta(self, velocity, pressure):
        dt = self.dt

        v_1, p_1 = velocity, pressure
        rhs_1 = self.adv_diff_press(v_1, p_1)
        v_2_old = velocity + (dt / 2) * rhs_1
        v_2, p_2 = self.pressure_treatment(v_2_old, p_1, dt / 2)

        rhs_2 = self.adv_diff_press(v_2, p_2)
        v_3_old = velocity + (dt / 2) * rhs_2
        v_3, p_3 = self.pressure_treatment(v_3_old, p_2, dt / 2)

        rhs_3 = self.adv_diff_press(v_3, p_3)
        v_4_old = velocity + dt * rhs_2
        v_4, p_4 = self.pressure_treatment(v_4_old, p_3, dt)

        rhs_4 = self.adv_diff_press(v_4, p_4)
        v_p1_old = velocity + (dt / 6) * (rhs_1 + 2 * rhs_2 + 2 * rhs_3 + rhs_4)
        p_p1_old = (1 / 6) * (p_1 + 2 * p_2 + 2 * p_3 + p_4)
        v_p1, p_p1 = self.pressure_treatment(v_p1_old, p_p1_old, dt)

        return v_p1, p_p1


    def adv_diff_press(self, v, p):
        adv_diff_press = (advect.finite_difference(v, v, order=self.ord))
        diff = (diffuse.finite_difference(v, self.vis, order=self.ord))
        adv_diff_press += diff
        adv_diff_press -= field.spatial_gradient(p, at=self.at, order=self.ord, gradient_extrapolation=extrapolation.ZERO)
        return adv_diff_press.with_extrapolation(0)


    def pressure_treatment(self, v, p, dt_=None):
        if dt_ is None:
            dt_ = self.dt
        v, delta_p = \
            fluid.make_incompressible(v, solve=math.Solve('biCG-stab(2)', 1e-10, 1e-10), order=self.ord)
        p += delta_p / dt_
        return v, p


    def run(self, jit_compile=True, t_num=0, freq=100, eps=0, recap=False):
        print(f"run {self.name}:")
        if recap:
            self.name += "_rez"

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

        DOMAIN_V = dict(bounds=Box['x,y', 0:1, 0:1], x=self.xynum, y=self.xynum,
                      extrapolation=extrapolation.combine_sides(
                          x=extrapolation.ZERO,
                          y=(extrapolation.ZERO, extrapolation.combine_by_direction(extrapolation.ZERO, extrapolation.ConstantExtrapolation(-1)))))

        DOMAIN_P = dict(bounds=Box['x,y', 0:1, 0:1], x=self.xynum, y=self.xynum,
                       extrapolation=extrapolation.ZERO_GRADIENT)

        if recap:
            post_name = self.name[:-4]
            data_ = np.load(f"data/{post_name}/data.npz")
            t_num_ = data_['t_num'].item()
            freq_ = data_['freq'].item()
            velocity = field.read(f"data/{post_name}/vel_{t_num_-freq_}.npz")
            pressure = field.read(f"data/{post_name}/press_{t_num_-freq_}.npz")
        else:
            velocity = CenteredGrid(tensor([0, 0], channel(vector='x, y')), **DOMAIN_V)
            pressure = CenteredGrid(0, **DOMAIN_P)


        field.write(velocity, f"data/{self.name}/vel_{0}")
        field.write(pressure, f"data/{self.name}/press_{0}")
        print(f"timestep: {0} of {self.t_num}")
        self.diver = math.jit_compile(field.divergence, "order, implicit, implicitness")
        # self.diver = field.divergence
        div = self.diver(velocity, order=self.ord)
        div_mean = math.mean(math.abs(div.values)).numpy().max()
        print(f"div mean: ", div_mean)
        div_max = math.max(math.abs(div.values)).numpy().max()
        print(f"div max: ", div_max)


        # for i in range(3):
        #     velocity, pressure = fluid.make_incompressible(velocity, order=4, solve=math.Solve('biCG-stab(2)', 1e-7, 1e-7))
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

        for i in range(1, self.t_num+1 if not recap else 2000):

            if i % freq == 0:
                print(f"timestep: {i} of {self.t_num}")
                div = self.diver(velocity, order=self.ord)
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

        for i in range(int(t_num / freq)+1):
            t = tensor(i * freq * dt)
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


eps = 1e-6
resols = [31, 61, 121]
# resols = [31]
ords = ['low', 'mid', 'high']
# ords = ['high']
re = 1000

for ord in ords:
    for res in resols:
        test = TestRun(0, 'center', ord, res, None, 1 / re, 0.001,
                           name=f"phi3.0_finale_{ord}_{res}")
        test.run(t_num=300000, freq=100, jit_compile=True, eps=eps)
        test.draw_plots()


# re = 1000
# res = 31
# for ord in ['high']:
#     for dt in [0.003]:
#             test = TestRun(0, 'center', ord, res, None, 1 / re, dt,
#                                name=f"timestep_invest2_{ord}_{res}_dt_{dt}")
#             test.run(t_num=3000000, freq=100, jit_compile=True, eps=eps)
#             test.draw_plots()

print('done')
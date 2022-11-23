""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""

import os
from phi.jax.flow import *

dt = 0.0025
visc = 0.01

math.set_global_precision(64)


def tgv_F(vis, t):
    return math.exp(-2 * vis * t)


def tgv_velocity(x, vis=0, t=0):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([cos.vector['x'] * sin.vector['y'] * tgv_F(vis, t),
                       - sin.vector['x'] * cos.vector['y'] * tgv_F(vis, t)], dim=channel('vector'))


def tgv_pressure(x, vis=0, t=0):
    return -1 / 4 * (math.sum(math.cos(2 * x), 'vector')) * (tgv_F(vis, t) ** 2)


class TestRun:

    def __init__(self, tges, gridtype, timestep, name='data'):

        if name is None:
            self.name = f"{timestep}"
        else:
            self.name = name

        self.t_num = int(math.ceil(tges / dt))
        self.gridtype = gridtype
        self.timestep = getattr(self, timestep)

    def phi_flow(self, velocity, pressure):
        velocity = advect.semi_lagrangian(velocity, velocity, dt)
        velocity, pressure = fluid.make_incompressible(velocity, solve=Solve('CG-adaptive', 1e-5, 0, x0=pressure))
        velocity = diffuse.explicit(velocity, 0.1, dt)
        return velocity, 0


    def high_order(self, velocity, pressure):

        v_1, p_1 = velocity, pressure
        rhs_1 = self.adp_high_ord(v_1, p_1)
        v_2_old = velocity + (dt / 2) * rhs_1
        v_2, p_2 = self.pt_high_ord(v_2_old, p_1, dt / 2)
        # vis.plot(velocity.vector['x'], velocity.vector['y'], title=f'vel 1')
        # vis.show()
        # vis.plot(pressure, title=f'press 1')
        # vis.show()


        rhs_2 = self.adp_high_ord(v_2, p_2)
        v_3_old = velocity + (dt / 2) * rhs_2
        v_3, p_3 = self.pt_high_ord(v_3_old, p_2, dt / 2)
        # vis.plot(velocity.vector['x'], velocity.vector['y'], title=f'vel 2')
        # vis.show()
        # vis.plot(pressure, title=f'press 2')
        # vis.show()

        rhs_3 = self.adp_high_ord(v_3, p_3)
        v_4_old = velocity + dt * rhs_2
        v_4, p_4 = self.pt_high_ord(v_4_old, p_3, dt)
        # vis.plot(velocity.vector['x'], velocity.vector['y'], title=f'vel 3')
        # vis.show()
        # vis.plot(pressure, title=f'press 3')
        # vis.show()

        rhs_4 = self.adp_high_ord(v_4, p_4)
        v_p1_old = velocity + (dt / 6) * (rhs_1 + 2 * rhs_2 + 2 * rhs_3 + rhs_4)
        p_p1_old = (1 / 6) * (p_1 + 2 * p_2 + 2 * p_3 + p_4)
        v_p1, p_p1 = self.pt_high_ord(v_p1_old, p_p1_old, dt)
        # vis.plot(velocity.vector['x'], velocity.vector['y'], title=f'vel 4')
        # vis.show()
        # vis.plot(pressure, title=f'press 4')
        # vis.show()

        return v_p1, p_p1


    def adp_high_ord(self, v, p):

        adv_diff_press = (advect.finite_difference(v, v, dt, scheme=Scheme(6, Solve('GMRES', 1e-5, 1e-5))) - v) / dt

        # vis.plot(adv_diff_press.vector['x'], adv_diff_press.vector['y'], title=f'adv')
        # vis.show()

        diff = (diffuse.finite_difference(v, visc, dt, scheme=Scheme(6, Solve('GMRES', 1e-5, 1e-5))) - v) / dt

        # vis.plot(diff.vector['x'], diff.vector['y'], title=f'diff')
        # vis.show()

        adv_diff_press += diff
        press = field.spatial_gradient(p, type=self.gridtype, scheme=Scheme(4), gradient_extrapolation=extrapolation.combine_sides(
            x=extrapolation.PERIODIC,
            y=extrapolation.combine_by_direction(extrapolation.ANTIREFLECT, extrapolation.ANTISYMMETRIC)))
        # press = field.spatial_gradient(p, type=self.gridtype, scheme=Scheme(4),
        #                                gradient_extrapolation=extrapolation.PERIODIC)

        # vis.plot(press.vector['x'], press.vector['y'], title=f'press')
        # vis.show()

        adv_diff_press -= press
        return adv_diff_press

    def pt_high_ord(self, v, p, dt_=dt):
        v, delta_p = fluid.make_incompressible(v, scheme=Scheme(4), solve=math.Solve('CG', 1e-5, 1e-5))
        p += delta_p / dt_
        return v, p


    def run(self, jit_compile=True, t_num=0, freq=100):
        print(f"run {self.name}:")

        # os.mkdir(f"data/{self.name}")

        if jit_compile:
            timestepper = math.jit_compile(self.timestep)
        else:
            timestepper = self.timestep

        if t_num > 0:
            self.t_num = t_num

        # DOMAIN = dict(bounds=Box['x,y', 0:1, 0:1], x=100, y=100, extrapolation=extrapolation.combine_sides(
        #     x=extrapolation.PERIODIC,
        #     y=extrapolation.combine_by_direction(extrapolation.ANTIREFLECT, extrapolation.ANTISYMMETRIC)))
        #
        # DOMAIN2 = dict(bounds=Box['x,y', 0:1, 0:1], x=100, y=100, extrapolation=extrapolation.combine_sides(x=extrapolation.PERIODIC, y=extrapolation.SYMMETRIC))

        DOMAIN = dict(bounds=Box['x,y', 0:1, 0:1], x=100, y=100, extrapolation=extrapolation.PERIODIC)

        DOMAIN2 = dict(bounds=Box['x,y', 0:1, 0:1], x=100, y=100, extrapolation=extrapolation.PERIODIC)

        # DOMAIN = dict(bounds=Box['x,y', 0:100, 0:100], x=50, y=20, extrapolation=extrapolation.PERIODIC)

        # DOMAIN2 = dict(bounds=Box['x,y', 0:100, 0:100], x=50, y=20, extrapolation=extrapolation.PERIODIC)


        velocity = StaggeredGrid(0, **DOMAIN)
        vals_x = velocity.values.vector['x']
        t = math.scatter(math.zeros(vals_x.shape.only('x')),
                     tensor([5], instance('points')),
                     tensor([5], instance('points')))
        vals_x = vals_x + t

        velocity = velocity.with_values(stack([vals_x, velocity.values.vector['y']], channel(vector='x,y')))
        pressure = CenteredGrid(0, **DOMAIN2)

        # vis.plot(velocity, pressure, title=f'vel and press')
        # vis.show()
        # vis.plot(velocity.vector[0], velocity.vector[1], title=f'vel x and vel y')
        # vis.show()

        velocity, pressure, solveinfo = fluid.make_incompressible(velocity, scheme=Scheme(4), solve=math.Solve('GMRES', 1e-5, 1e-5))
        vis.plot(solveinfo.residual, title=f'residual')
        vis.show()
        vis.plot(pressure, title=f'pressure')
        vis.show()
        vis.plot(velocity.vector[0], velocity.vector[1], title=f'vel x and vel y')
        vis.show()

        # velocity = StaggeredGrid(tgv_velocity, **DOMAIN)
        # pressure = CenteredGrid(tgv_pressure, **DOMAIN2)



        vel_data = []
        press_data = []

        vel_data.append(velocity)
        press_data.append(pressure)

        for i in range(self.t_num):

            if i % freq == 0:
                print(f"timestep: {i} of {self.t_num}")

                vel_data.append(velocity)
                press_data.append(pressure)

            velocity, pressure = timestepper(velocity, pressure)
            # vis.plot(velocity, title=f'{i} vel')
            # vis.show()
            # vis.plot(velocity.vector['x'], velocity.vector['y'], title=f'{i} vel')
            # vis.show()
            # vis.plot(pressure, title=f'{i} press')
            # vis.show()

        field.write(stack(vel_data, batch('time')), f"data/{self.name}/vel")
        field.write(stack(press_data, batch('time')), f"data/{self.name}/press")
        np.savez(f"data/{self.name}/data", t_num=self.t_num, dt=dt, visc=visc, freq=freq)

        print()

    def draw_plots(self):
        os.mkdir(f"plots/{self.name}")

        vel_data = field.read(f"data/{self.name}/vel.npz")
        press_data = field.read(f"data/{self.name}/press.npz")
        data = np.load(f"data/{self.name}/data.npz")

        t_num = data['t_num'].item()
        dt = data['dt'].item()
        visc = data['visc'].item()
        freq = data['freq'].item()

        vel = unstack(vel_data, 'time')
        press = unstack(press_data, 'time')

        for i in range(int((t_num + 1) / freq)):
            t = tensor(i * freq * dt)
            f1 = vis.plot(vel[i], press[i], title=f'{i}: vel, press')._obj
            timestamp = '{:07.4f}'.format(float(t))
            vis.savefig(f"plots/{self.name}/v_and_p_{timestamp}.jpg", f1)
            vis.close()
            f2 = vis.plot(vel[i].vector[0], vel[i].vector[1], title=f'{i}: vel fields')._obj
            timestamp = '{:07.4f}'.format(float(t))
            vis.savefig(f"plots/{self.name}/v_fields_{timestamp}.jpg", f2)
            vis.close()



# test = TestRun(0, StaggeredGrid, "high_order", name="real_symmetric")
# # test.run(t_num=10, freq=3, jit_compile=True)
# test.draw_plots()

# test = TestRun(0, StaggeredGrid, "high_order", name="test_with_init_vel_left_larger_periodic_low_vis_smaller_time")
# test.run(t_num=10, freq=1, jit_compile=True)
# test.draw_plots()
#
# test = TestRun(0, StaggeredGrid, "high_order", name="test")
# test.run(t_num=1, freq=1, jit_compile=False)
# test.draw_plots()

test = TestRun(0, StaggeredGrid, "high_order", name="tackle_inital_make_incomp")
test.run(t_num=0, freq=3, jit_compile=True)
# test.draw_plots()

""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""

import os
from phi.jax.flow import *

# dt = 0.0015
# visc = 0.01

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


class TestRun:

    def __init__(self, tges, gridtype, timestep, xynum, p_grad, vis, dt, name='data'):

        if name is None:
            self.name = f"{timestep}"
        else:
            self.name = name

        self.t_num = int(math.ceil(tges / dt))
        self.gridtype = gridtype
        self.timestep = getattr(self, timestep)
        self.xynum = xynum
        self.p_grad = p_grad
        self.vis = vis
        self.dt = dt

    def phi_flow(self, velocity, pressure):
        velocity = advect.semi_lagrangian(velocity, velocity, self.dt)
        velocity, pressure = fluid.make_incompressible(velocity, solve=Solve('CG-adaptive', 1e-5, 0, x0=pressure))
        velocity = diffuse.explicit(velocity, 0.1, self.dt)
        return velocity, 0


    def high_order(self, velocity, pressure):
        dt = self.dt

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

        adv_diff_press = (advect.finite_difference(v, v, self.dt, scheme=Scheme(6, Solve('GMRES', 1e-5, 1e-5))) - v) / self.dt

        # vis.plot(adv_diff_press.vector['x'], adv_diff_press.vector['y'], title=f'adv')
        # vis.show()

        diff = (diffuse.finite_difference(v, self.vis, self.dt, scheme=Scheme(6, Solve('GMRES', 1e-5, 1e-5))) - v) / self.dt

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

        # adv_diff_press -= press
        press = -press.with_values(stack([math.ones(press.vector['x'].values.shape), math.zeros(press.vector['y'].values.shape)], channel(vector='x,y'))*self.p_grad)

        # vis.plot(press.vector['x'], press.vector['y'], title=f'press')
        # vis.show()
        adv_diff_press -= press
        return adv_diff_press

    def pt_high_ord(self, v, p, dt_):
        v, delta_p = fluid.make_incompressible(v, scheme=Scheme(4), solve=math.Solve('CG', 1e-5, 1e-5))
        p += delta_p / dt_
        return v, p

    def adp_low_ord(self, v, p):
        adv_diff_press = advect.finite_difference(v, v, self.dt) - v
        adv_diff_press += (diffuse.finite_difference(v, self.vis, self.dt) - v) / self.dt
        adv_diff_press -= field.spatial_gradient(p, type=self.gridtype)
        return adv_diff_press

    def pt_low_ord(self, v, p, dt_):
        v, delta_p = \
            fluid.make_incompressible(v,
                                      solve=math.Solve('auto', 1e-12, 1e-12,
                                                       gradient_solve=math.Solve('auto', 1e-12, 1e-12)))
        p += delta_p / dt_
        return v, p

    def run(self, jit_compile=True, t_num=0, freq=100):
        print(f"run {self.name}:")

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

        DOMAIN = dict(bounds=Box['x,y', 0:0.5, 0:0.5], x=self.xynum, y=self.xynum, extrapolation=extrapolation.combine_sides(
            x=extrapolation.PERIODIC,
            y=extrapolation.combine_by_direction(extrapolation.ANTIREFLECT, extrapolation.ANTISYMMETRIC)))

        DOMAIN2 = dict(bounds=Box['x,y', 0:0.5, 0:0.5], x=self.xynum, y=self.xynum, extrapolation=extrapolation.combine_sides(x=extrapolation.PERIODIC, y=extrapolation.SYMMETRIC))

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

        # velocity, pressure, solveinfo = fluid.make_incompressible(velocity, scheme=Scheme(4), solve=math.Solve('GMRES', 1e-5, 1e-5))
        velocity, pressure = fluid.make_incompressible(velocity, scheme=Scheme(4), solve=math.Solve('GMRES', 1e-5, 1e-5))
        velocity, pressure = fluid.make_incompressible(velocity, scheme=Scheme(4), solve=math.Solve('GMRES', 1e-5, 1e-5))
        velocity, pressure = fluid.make_incompressible(velocity, scheme=Scheme(4), solve=math.Solve('GMRES', 1e-5, 1e-5))


        # vis.plot(solveinfo.residual, title=f'residual')
        # vis.show()
        # vis.plot(pressure, title=f'pressure')
        # vis.show()
        # vis.plot(velocity.vector[0], velocity.vector[1], title=f'vel x and vel y')
        # vis.show()

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
        np.savez(f"data/{self.name}/data", t_num=self.t_num, dt=self.dt, visc=self.vis, freq=freq, p_grad=self.p_grad)

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
            # f1 = vis.plot(vel[i], press[i], title=f'{i}: vel, press')._obj
            # timestamp = '{:07.4f}'.format(float(t))
            # vis.savefig(f"plots/{self.name}/v_and_p_{timestamp}.jpg", f1)
            # vis.close()
            # f2 = vis.plot(vel[i].vector[0], vel[i].vector[1], title=f'{i}: vel fields')._obj
            # timestamp = '{:07.4f}'.format(float(t))
            # vis.savefig(f"plots/{self.name}/v_fields_{timestamp}.jpg", f2)
            # vis.close()
            f1 = vis.plot(vel[i].vector[0], press[i], title=f'{i}: vel_x, press')._obj
            timestamp = '{:07.4f}'.format(float(t))
            vis.savefig(f"plots/{self.name}/v_and_p_{timestamp}.jpg", f1)
            vis.close()

    def more_plots(self):
        # os.mkdir(f"plots/{self.name}")

        vel_data = field.read(f"data/{self.name}/vel.npz")
        press_data = field.read(f"data/{self.name}/press.npz")
        data = np.load(f"data/{self.name}/data.npz")

        t_num = data['t_num'].item()
        dt = data['dt'].item()
        visc = data['visc'].item()
        freq = data['freq'].item()
        p_grad = data['p_grad'].item()
        # p_grad = 0.01


        vel = unstack(vel_data, 'time')
        press = unstack(press_data, 'time')


        last_vel = vel[-1].vector['x']
        avg_profile = math.sum(last_vel.values, 'x') / last_vel.values.shape.only('y').size
        profile_points = vel[-1].vector['x'].x[0].points

        ana_sol = plane_poisseuille(profile_points, visc, p_grad, last_vel.bounds.size.vector['y'])
        f1 = vis.plot([avg_profile, ana_sol, avg_profile - ana_sol], same_scale=False, title="avg_profile, ana_sol, avg_profile")._obj
        # vis.show()
        vis.savefig(f"plots/{self.name}/velprofile_anasol_error_plot.jpg", f1)
        vis.close()

        print('done')






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
#
# test = TestRun(0, StaggeredGrid, "high_order", 100, 0.01, 0.01, 0.0015, name="Poiseuille flow 5_")
# test.more_plots()
#
# test = TestRun(0, StaggeredGrid, "high_order", 100, 0.01, 0.01, 0.0015, name="Poiseuille flow 5__")
# test.more_plots()

# test = TestRun(0, StaggeredGrid, "high_order", 100, 0.01, 0.01, 0.0015, name="Poiseuille flow 5___")
# test.more_plots()
#
# test = TestRun(0, StaggeredGrid, "high_order", 100, 0.01, 0.01, 0.0015, name="Poiseuille flow 5____")
# test.more_plots()


# ----------------------------------------------------------------


# test = TestRun(0, StaggeredGrid, "high_order", 10, 0.01, 0.01, 0.0015, name="Poiseuille_flow_6_re10_dp0.01_vi0.01_dt0.0015")
# test.run(t_num=20000, freq=100, jit_compile=True)
# test.draw_plots()
# test.more_plots()
#
# test = TestRun(0, StaggeredGrid, "high_order", 25, 0.01, 0.01, 0.0015, name="Poiseuille_flow_6_re25_dp0.01_vi0.01_dt0.0015")
# test.run(t_num=20000, freq=100, jit_compile=True)
# test.draw_plots()
# test.more_plots()
#
# test = TestRun(0, StaggeredGrid, "high_order", 50, 0.01, 0.01, 0.0015, name="Poiseuille_flow_6_re50_dp0.01_vi0.01_dt0.0015")
# test.run(t_num=20000, freq=100, jit_compile=True)
# test.draw_plots()
# test.more_plots()

# test = TestRun(0, StaggeredGrid, "high_order", 100, 0.01, 0.01, 0.0015, name="Poiseuille_flow_6_re100_dp0.01_vi0.01_dt0.0015")
# test.run(t_num=20000, freq=100, jit_compile=True)
# test.draw_plots()
# test.more_plots()





test = TestRun(0, StaggeredGrid, "high_order", 10, 0.01, 0.003, 0.0005, name="Poiseuille_flow_6_re10_dp0.01_vi0.003_dt0.0005")
test.run(t_num=90000, freq=450, jit_compile=True)
test.draw_plots()
test.more_plots()

test = TestRun(0, StaggeredGrid, "high_order", 25, 0.01, 0.003, 0.0005, name="Poiseuille_flow_6_re25_dp0.01_vi0.003_dt0.0005")
test.run(t_num=90000, freq=450, jit_compile=True)
test.draw_plots()
test.more_plots()

test = TestRun(0, StaggeredGrid, "high_order", 50, 0.01, 0.003, 0.0005, name="Poiseuille_flow_6_re50_dp0.01_vi0.003_dt0.0005")
test.run(t_num=90000, freq=450, jit_compile=True)
test.draw_plots()
test.more_plots()

# test = TestRun(0, StaggeredGrid, "high_order", 100, 0.01, 0.003, 0.0005, name="Poiseuille_flow_6_re100_dp0.01_vi0.003_dt0.0005")
# test.run(t_num=60000, freq=300, jit_compile=True)
# test.draw_plots()
# test.more_plots()





test = TestRun(0, StaggeredGrid, "high_order", 10, 0.05, 0.01, 0.0003, name="Poiseuille_flow_6_re10_dp0.05_vi0.01_dt0.0003")
test.run(t_num=150000, freq=750, jit_compile=True)
test.draw_plots()
test.more_plots()

test = TestRun(0, StaggeredGrid, "high_order", 25, 0.05, 0.01, 0.0003, name="Poiseuille_flow_6_re25_dp0.05_vi0.01_dt0.0003")
test.run(t_num=150000, freq=750, jit_compile=True)
test.draw_plots()
test.more_plots()

test = TestRun(0, StaggeredGrid, "high_order", 50, 0.05, 0.01, 0.0003, name="Poiseuille_flow_6_re50_dp0.05_vi0.01_dt0.0003")
test.run(t_num=150000, freq=750, jit_compile=True)
test.draw_plots()
test.more_plots()

# test = TestRun(0, StaggeredGrid, "high_order", 100, 0.05, 0.01, 0.0003, name="Poiseuille_flow_6_re100_dp0.05_vi0.01_dt0.0003")
# test.run(t_num=100000, freq=500, jit_compile=True)
# test.draw_plots()
# test.more_plots()


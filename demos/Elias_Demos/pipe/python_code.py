""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""
import os

from phi.jax.flow import *

dt = 1 / 4
visc = 0.1


class TestRun:

    def __init__(self, tges, gridtype, timestep, name=None):

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

        rhs_2 = self.adp_high_ord(v_2, p_2)
        v_3_old = velocity + (dt / 2) * rhs_2
        v_3, p_3 = self.pt_high_ord(v_3_old, p_2, dt / 2)

        rhs_3 = self.adp_high_ord(v_3, p_3)
        v_4_old = velocity + dt * rhs_2
        v_4, p_4 = self.pt_high_ord(v_4_old, p_3, dt)

        rhs_4 = self.adp_high_ord(v_4, p_4)
        v_p1_old = velocity + (dt / 6) * (rhs_1 + 2 * rhs_2 + 2 * rhs_3 + rhs_4)
        p_p1_old = (1 / 6) * (p_1 + 2 * p_2 + 2 * p_3 + p_4)
        v_p1, p_p1 = self.pt_high_ord(v_p1_old, p_p1_old, dt)

        return v_p1, p_p1


    def adp_high_ord(self, v, p):
        adv_diff_press = advect.finite_difference(v, v, dt, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12))) - v
        adv_diff_press += (diffuse.finite_difference(v, visc, dt, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12))) - v) / dt
        adv_diff_press -= field.spatial_gradient(p, type=self.gridtype, scheme=Scheme(4), gradient_extrapolation=extrapolation.combine_sides(
            x=extrapolation.combine_by_direction(extrapolation.REFLECT, extrapolation.SYMMETRIC),
            y=extrapolation.PERIODIC))
        return adv_diff_press

    def pt_high_ord(self, v, p, dt_=dt):
        v, delta_p = fluid.make_incompressible(v, scheme=Scheme(4))
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

        DOMAIN = dict(x=50, y=32, extrapolation=extrapolation.combine_sides(
            x=extrapolation.combine_by_direction(extrapolation.REFLECT, extrapolation.SYMMETRIC),
            y=extrapolation.PERIODIC))

        DOMAIN2 = dict(x=50, y=32, extrapolation=extrapolation.combine_sides(x=extrapolation.SYMMETRIC, y=extrapolation.PERIODIC))
        velocity = StaggeredGrid(0, **DOMAIN)
        pressure = CenteredGrid(0, **DOMAIN2)

        # pressure = CenteredGrid(partial(tgv_pressure, vis=visc, t=0), **DOMAIN)
        # velocity = self.gridtype(partial(tgv_velocity, vis=visc, t=0), **DOMAIN)

        vx_data = []
        vy_data = []
        p_data = []

        for i in range(self.t_num):

            if i % freq == 0:
                print(f"timestep: {i} of {self.t_num}")

                vx_data.append(velocity.values.vector[0].numpy(('x', 'y')))
                vy_data.append(velocity.values.vector[1].numpy(('x', 'y')))

                p_data.append(pressure.values.numpy(('x', 'y')))

            velocity, pressure = timestepper(velocity, pressure)

            np.savez(f"data/{self.name}", vx=np.array(vx_data), vy=np.array(vy_data),
                     p=np.array(p_data), t_num=self.t_num, dt=dt, visc=visc, freq=freq)

        print()

    def draw_plots(self):
        os.mkdir(f"plots/{self.name}")

        file = np.load(f"data/{self.name}" + ".npz")

        vx = file['vx']
        vy = file['vy']
        p = file['p']
        t_num = file['t_num'].item()
        dt = file['dt'].item()
        visc = file['visc'].item()
        freq = file['freq'].item()

        for i in range(int(t_num / freq)):
            t = tensor(i * freq * dt)
            vis.plot(tensor(vx[i], spatial('x'), spatial('y')), tensor(vy[i], spatial('x'), spatial('y')))
            vis.savefig(f"plots/{self.name}/{t}.jpg")

tges = 10
high_order = TestRun(tges, StaggeredGrid, "high_order", name="high_order")
# high_order.run(t_num=100, freq=1)
high_order.draw_plots()

# from phi.flow import *
from phi.jax.flow import *
from functools import partial
import os

math.set_global_precision(64)


def tgv_F(vis, t):
    return math.exp(-2 * vis * t)


def tgv_velocity(x, vis=0, t=0):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([cos.vector['x'] * sin.vector['y'] * tgv_F(vis, t),
                       - sin.vector['x'] * cos.vector['y'] * tgv_F(vis, t)], dim=channel('vector'))


def tgv_pressure(x, vis=0, t=0):
    return -1 / 4 * (math.sum(math.cos(2 * x), 'vector')) * (tgv_F(vis, t) ** 2)


dt = 1 / 16 * 2 * math.pi / 264
visc = 0.1


class TestRun:

    def __init__(self, xy_nums, tges, gridtype, timestep, adv_diff_press, pressure_treatment, name=None):

        if name is None:
            self.name = f"{timestep} - {adv_diff_press} - {pressure_treatment}"
        else:
            self.name = name

        self.xy_nums = xy_nums
        self.t_num = int(math.ceil(tges / dt))
        self.gridtype = gridtype
        self.timestep = getattr(self, timestep)
        self.adv_diff_press = getattr(self, adv_diff_press)
        self.pressure_treatment = getattr(self, pressure_treatment)

    def fst_ord_time_step(self, velocity, pressure):
        v_old = self.adv_diff_press(velocity, pressure)
        v_new, p_new = self.pressure_treatment(v_old, pressure)
        return v_new, p_new

    def fourth_ord_runge_kutta(self, velocity, pressure):
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


    def adp_phi_flow(self, v, p):
        adv_diff_press = (advect.semi_lagrangian(v, v, dt) - v) / dt
        adv_diff_press += (diffuse.explicit(v, visc, dt) - v) / dt
        adv_diff_press -= field.spatial_gradient(p, type=self.gridtype)
        return adv_diff_press

    def pt_phi_flow(self, v, p, dt_=dt):
        v, delta_p = fluid.make_incompressible(v)
        p += delta_p / dt_
        return v, p


    def adp_high_ord(self, v, p):
        adv_diff_press = (advect.finite_difference(v, v, dt, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12))) - v) / dt
        adv_diff_press += (diffuse.finite_difference(v, visc, dt, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12))) - v) / dt
        adv_diff_press -= field.spatial_gradient(p, type=self.gridtype, scheme=Scheme(4))
        return adv_diff_press

    def pt_high_ord(self, v, p, dt_=dt):
        v, delta_p = \
            fluid.make_incompressible(v, scheme=Scheme(4),
                                      solve=math.Solve('auto', 1e-12, 1e-12, gradient_solve=math.Solve('auto', 1e-12, 1e-12)))
        p += delta_p / dt_
        return v, p


    def adp_mid_ord(self, v, p):
        adv_diff_press = (advect.finite_difference(v, v, dt, scheme=Scheme(4)) - v) / dt
        adv_diff_press += (diffuse.finite_difference(v, visc, dt, scheme=Scheme(4)) - v) / dt
        adv_diff_press -= field.spatial_gradient(p, type=self.gridtype, scheme=Scheme(4))
        return adv_diff_press

    def pt_mid_ord(self, v, p, dt_=dt):
        v, delta_p = \
            fluid.make_incompressible(v,
                                      solve=math.Solve('auto', 1e-12, 1e-12, gradient_solve=math.Solve('auto', 1e-12, 1e-12)),
                                      scheme=Scheme(4))
        p += delta_p / dt_
        return v, p


    def adp_low_ord(self, v, p):
        adv_diff_press = advect.finite_difference(v, v, dt) - v
        adv_diff_press += (diffuse.finite_difference(v, visc, dt) - v) / dt
        adv_diff_press -= field.spatial_gradient(p, type=self.gridtype)
        return adv_diff_press

    def pt_low_ord(self, v, p, dt_=dt):
        v, delta_p = \
            fluid.make_incompressible(v,
                                      solve=math.Solve('auto', 1e-12, 1e-12, gradient_solve=math.Solve('auto', 1e-12, 1e-12)))
        p += delta_p / dt_
        return v, p


    def run(self, jit_compile=True, t_num=0, freq=100):
        print(f"run {self.name}:")

        os.mkdir(f"data/{self.name}")

        if jit_compile:
            timestepper = math.jit_compile(self.timestep)
        else:
            timestepper = self.timestep

        if t_num > 0:
            self.t_num = t_num

        for xy_num in self.xy_nums:

            print(f"xy_num: {xy_num}")

            DOMAIN = dict(extrapolation=extrapolation.PERIODIC,
                          bounds=Box['x,y', 0:2 * math.pi, 0:2 * math.pi], x=xy_num, y=xy_num)

            pressure = CenteredGrid(partial(tgv_pressure, vis=visc, t=0), **DOMAIN)
            velocity = self.gridtype(partial(tgv_velocity, vis=visc, t=0), **DOMAIN)

            v_data = []
            p_data = []

            for i in range(self.t_num):

                if i % freq == 0:
                    print(f"timestep: {i} of {self.t_num}")

                    v_data.append(velocity.values.numpy(('x', 'y', 'vector')))
                    p_data.append(pressure.values.numpy(('x', 'y')))

                velocity, pressure = timestepper(velocity, pressure)

            np.savez(f"data/{self.name}/res_{xy_num}", v=np.array(v_data),
                     p=np.array(p_data),
                     xy_num=xy_num, t_num=self.t_num, dt=dt, visc=visc, freq=freq)

            print()


    def calc_errors(self, relative_error=False):
        print(f"error {self.name}:")
        data_dir = f"data/{self.name}"

        print(self.name + ":")

        errors_by_res = []

        for xy_num in self.xy_nums:
            print(f"{xy_num}:")

            file = np.load(f"{data_dir}/res_{xy_num}.npz")

            v = file['v']
            p = file['p']
            xy_num = file['xy_num'].item()
            t_num = file['t_num'].item()
            dt = file['dt'].item()
            visc = file['visc'].item()
            freq = file['freq'].item()

            DOMAIN = dict(extrapolation=extrapolation.PERIODIC,
                          bounds=Box['x,y', 0:2 * math.pi, 0:2 * math.pi], x=xy_num, y=xy_num)

            def anal_sol(t):
                return self.gridtype(partial(tgv_velocity, vis=visc, t=t), **DOMAIN)

            errors = []
            for i in range(int(t_num/freq)):

                t = tensor(i * freq * dt)
                ana_sol_values = anal_sol(t).values
                velocity = tensor(v[i], spatial('x'), spatial('y'), channel('vector'))
                if relative_error:
                    error = math.sqrt(math.mean((ana_sol_values - velocity) ** 2)) / math.mean(abs(ana_sol_values))
                else:
                    error = math.sum(math.abs(ana_sol_values - velocity)) / xy_num ** 2
                error_point = math.stack([t, error], channel('vector'))
                errors.append(error_point)
                print(f"v error: {error} at {i * dt}s")

            errors_by_res.append(math.stack(errors, channel("t")))
            print()

        np.savez(f"errors/{self.name}", data=math.stack(errors_by_res, channel("res")).numpy(('res', 'vector', 't')),
                 xy_nums=self.xy_nums)

        print()
        print()


tges = 5

xy_nums = [5, 15, 45, 95, 155, 255]
xy_nums2 = [5, 15, 25, 35, 45, 55, 65]

xy_nums_ = [5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 155, 185, 215, 255]
xy_nums2_ = [3, 5, 7, 10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]

small_test = [5]


# test = TestRun(xy_nums, tges, CenteredGrid, "fourth_ord_runge_kutta", "adp_phi_flow", "pt_phi_flow", "test")
# test.run(jit_compile=False)
# test.calc_errors(relative_error=True)


#
# pt_phi_flow = TestRun(xy_nums, tges, CenteredGrid, "fourth_ord_runge_kutta", "adp_phi_flow", "pt_phi_flow", "phi_flow")
# pt_phi_flow.run(jit_compile=True)
# pt_phi_flow.calc_errors(relative_error=True)
#
#
# pt_phi_flow_stagg = TestRun(xy_nums, tges, StaggeredGrid, "fourth_ord_runge_kutta", "adp_phi_flow", "pt_phi_flow", "phi_flow_stagg")
# pt_phi_flow_stagg.run(jit_compile=True)
# pt_phi_flow_stagg.calc_errors(relative_error=True)
#
#
# low_order = TestRun(xy_nums, tges, CenteredGrid, "fourth_ord_runge_kutta", "adp_low_ord", "pt_low_ord", "low_order")
# low_order.run(jit_compile=True)
# low_order.calc_errors(relative_error=True)
#
# low_order_stagg = TestRun(xy_nums, tges, StaggeredGrid, "fourth_ord_runge_kutta", "adp_low_ord", "pt_low_ord", "low_order_stagg")
# low_order_stagg.run(jit_compile=True)
# low_order_stagg.calc_errors(relative_error=True)
#
#
# mid_order = TestRun(xy_nums2, tges, CenteredGrid, "fourth_ord_runge_kutta", "adp_mid_ord", "pt_mid_ord", "mid_order")
# mid_order.run(jit_compile=True)
# mid_order.calc_errors(relative_error=True)
#
# mid_order_stagg = TestRun(xy_nums2, tges, StaggeredGrid, "fourth_ord_runge_kutta", "adp_mid_ord", "pt_mid_ord", "mid_order_stagg")
# mid_order_stagg.run(jit_compile=True)
# mid_order_stagg.calc_errors(relative_error=True)
#
#
# high_order = TestRun(xy_nums2, tges, CenteredGrid, "fourth_ord_runge_kutta", "adp_high_ord", "pt_high_ord", "high_order")
# high_order.run(jit_compile=False)
# high_order.calc_errors(relative_error=True)

high_order_stagg = TestRun(xy_nums2, tges, StaggeredGrid, "fourth_ord_runge_kutta", "adp_high_ord", "pt_high_ord", "high_order_stagg")
high_order_stagg.run(jit_compile=False)
high_order_stagg.calc_errors(relative_error=True)


print("done")

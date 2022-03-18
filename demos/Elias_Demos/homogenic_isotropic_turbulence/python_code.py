# from phi.flow import *
import os

from phi.jax.flow import *
from functools import partial

math.set_global_precision(64)

dt = 1 / 16 * 2 * math.pi / 1026
visc = 0.001


class TestRun:

    def __init__(self, sampling_rates, tges, pressure_treatment, name=None, entrance_step=2500):
        if name is None:
            self.name = f"{pressure_treatment}"
        else:
            self.name = name

        self.sampling_rates = sampling_rates
        self.pressure_treatment = getattr(self, pressure_treatment)
        self.t_num = int(math.ceil(tges / dt))
        self.timestep = self.fourth_ord_runge_kutta
        self.adv_diff_press = self.adp_ammount_laizet_sixth_ord
        self.gridtype = StaggeredGrid
        self.entrance_step = entrance_step



    def fourth_ord_runge_kutta(self, velocity, pressure):
        v_1, p_1 = velocity, pressure

        rhs_1 = self.adv_diff_press(v_1, p_1)
        v_2_old = velocity + (dt / 2) * rhs_1
        v_2, p_2 = self.pressure_treatment(v_2_old, p_1, dt/2)

        rhs_2 = self.adv_diff_press(v_2, p_2)
        v_3_old = velocity + (dt / 2) * rhs_2
        v_3, p_3 = self.pressure_treatment(v_3_old, p_2, dt/2)

        rhs_3 = self.adv_diff_press(v_3, p_3)
        v_4_old = velocity + dt * rhs_2
        v_4, p_4 = self.pressure_treatment(v_4_old, p_3, dt)

        rhs_4 = self.adv_diff_press(v_4, p_4)
        v_p1_old = velocity + (dt / 6) * (rhs_1 + 2 * rhs_2 + 2 * rhs_3 + rhs_4)
        p_p1_old = (1 / 6) * (p_1 + 2 * p_2 + 2 * p_3 + p_4)
        v_p1, p_p1 = self.pressure_treatment(v_p1_old, p_p1_old, dt)

        return v_p1, p_p1


    def adp_ammount_laizet_sixth_ord(self, v, p):
        adv_diff_press = advect.laiz(v, v, dt, return_amount=True)
        adv_diff_press += diffuse.laiz(v, visc, dt, return_amount=True)
        adv_diff_press -= field.spatial_gradient_laiz(p, type=self.gridtype)
        return adv_diff_press

    def pt_jaxsingle_kamp_sixth_ord_small_stencil(self, v, p, dt_):
        return fluid.make_incompressible_kamp_iter_jax(v, p, dt_, max_iter=1)

    def pt_jaxsingle_kamp_sixth_ord_big_stencil(self, v, p, dt_):
        return fluid.make_incompressible_kamp_iter_jax(v, p, dt_, max_iter=1, big_lap_stencil=True)



    def run(self, jit_compile=True, t_num=0, load_origin_data=False,
            save_vorticity_plot=False,
            freq=100, create_folder_new=True, save_per_step=False):

        if create_folder_new:
            os.mkdir(f"data/{self.name}")

        if jit_compile:
            timestepper = math.jit_compile(self.timestep)
        else:
            timestepper = self.timestep

        if t_num > 0:
            self.t_num = t_num

        xy_nums = []

        for sampling_rate in self.sampling_rates:

            if load_origin_data:
                # data from björns file
                data = np.load("Sayin_2048x2048_1.npz")
                data_tensor = math.tensor([data["arr_0"][::sampling_rate, ::sampling_rate],
                                           data["arr_1"][::sampling_rate, ::sampling_rate]],
                                          math.channel('vector'), math.spatial('x'), math.spatial('y'))
            else:
                # data from high res
                data = np.load(f"data/High_res_big_data/step_{self.entrance_step}.npz")
                data_tensor = math.tensor(data['v'][::sampling_rate, ::sampling_rate],
                                          math.spatial('x'), math.spatial('y'), math.channel('vector'))

            xy_num = data_tensor.x.size
            xy_nums.append(xy_num)
            print(f"sampling_rate: {sampling_rate} - xy_num: {xy_num}")


            DOMAIN = dict(extrapolation=extrapolation.PERIODIC,
                          bounds=Box['x,y', 0:2 * math.pi, 0:2 * math.pi], scheme=True, x=xy_num, y=xy_num)

            if load_origin_data:
                velocity = CenteredGrid(data_tensor, **DOMAIN).at(StaggeredGrid(data_tensor, **DOMAIN), scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))
            else:
                velocity = StaggeredGrid(data_tensor, **DOMAIN)

            if load_origin_data:
                # pressure by make incompressible
                pressure = CenteredGrid(0, **DOMAIN)
                velocity, pressure = self.pressure_treatment(velocity, pressure, 1)
            else:
                # pressure from data
                data_tensor_p = math.tensor(data['p'][::sampling_rate, ::sampling_rate],
                                            math.spatial('x'), math.spatial('y'))
                pressure = CenteredGrid(data_tensor_p, **DOMAIN)


            # errors = []
            v_data = []
            p_data = []

            for i in range(self.t_num):

                if i % freq == 0:
                    print(f"timestep: {i}")

                    t = tensor(i * dt)

                    if save_per_step:
                        # save data to file
                        np.savez(f"data/{self.name}/step_{self.entrance_step + i}" if len(self.sampling_rates) == 1 else
                                 f"data/{self.name}/sampling_rate_{sampling_rate}_step_{self.entrance_step + i}",
                                 v=velocity.values.numpy(('x', 'y', 'vector')),
                                 p=pressure.values.numpy(('x', 'y')), t=t)
                    else:
                        v_data.append(velocity.values.numpy(('x', 'y', 'vector')))
                        p_data.append(pressure.values.numpy(('x', 'y')))
                        # file = np.load(f"High_res_big_data/step_{i}.npz")
                        # v_high_res_data = file['v']
                        # v_high_res_data_resampled = v_high_res_data[::sampling_rate, ::sampling_rate]
                        # v_high_res_data_resampled_tensor = tensor(v_high_res_data_resampled,
                        #                                           math.spatial("x"), math.spatial("y"),
                        #                                           math.channel("vector"))
                        # error = math.sum(math.abs(velocity.values - v_high_res_data_resampled_tensor)) / xy_num ** 2
                        # error_point = math.stack([t, error], channel('vector'))
                        # errors.append(error_point)
                        # print(f"v error: {error} at {i * dt}s")

                    if save_vorticity_plot:
                        # save vorticity plot to file
                            vorticity = field.spatial_gradient_laiz(velocity.vector[1]).vector[0] - \
                                        field.spatial_gradient_laiz(velocity.vector[0]).vector[1]
                            vis.savefig(f"plots/{xy_num}_at_{i*dt}s.jpg", figure=vis.plot(vorticity))

                velocity, pressure = timestepper(velocity, pressure)


            print()

            if not save_per_step:
                np.savez(f"data/{self.name}/sampling_rate_{sampling_rate}", v=np.array(v_data),
                         p=np.array(p_data),
                         xy_num=xy_num, t_num=self.t_num, dt=dt, visc=visc, freq=freq,
                         sampling_rate=sampling_rate)


    def calc_errors(self, relative_error=False):
        data_dir = f"data/{self.name}"

        print(self.name + ":")

        errors_by_res = []
        xy_nums = []

        for sampling_rate in self.sampling_rates:
            print(f"{sampling_rate}:")

            file = np.load(f"{data_dir}/sampling_rate_{sampling_rate}.npz")

            v = file['v']
            p = file['p']
            xy_num = file['xy_num'].item()
            xy_nums.append(xy_num)
            t_num = file['t_num'].item()
            dt = file['dt'].item()
            visc = file['visc'].item()
            freq = file['freq'].item()
            sampling_rate = file['sampling_rate'].item()

            errors = []
            for i in range(int(t_num/freq)):
                t = tensor(dt * i * freq)
                timestep = i * freq
                file = np.load(f"data/High_res_big_data/step_{self.entrance_step + timestep}.npz")
                v_high_res_data = file['v']
                v_high_res_data_resampled = v_high_res_data[::sampling_rate, ::sampling_rate]
                v_high_res_data_resampled_tensor = tensor(v_high_res_data_resampled,
                                                          math.spatial("x"), math.spatial("y"),
                                                          math.channel("vector"))
                velocity = tensor(v[i], spatial('x'), spatial('y'), channel('vector'))
                if relative_error:
                    error = math.sqrt(math.mean((v_high_res_data_resampled_tensor - velocity) ** 2)) / \
                            math.mean(abs(v_high_res_data_resampled_tensor))
                else:
                    error = math.sum(math.abs(v_high_res_data_resampled_tensor - velocity)) / xy_num ** 2

                error_point = math.stack([t, error], channel('vector'))
                errors.append(error_point)
                print(f"v error: {error} at {i * dt}s")

            errors_by_res.append(math.stack(errors, channel("t")))
            print()

        np.savez(f"errors/{self.name}", data=math.stack(errors_by_res, channel("res")).numpy(('res', 'vector', 't')),
                 xy_nums=xy_nums)

        print()
        print()



tges = 5

# sampling_rates = [32, 28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2]
# sampling_rates = [128, 64]
sampling_rates = [30, 25, 20, 15, 10, 7, 5, 2]

# my_rk4_ho_step = TestRun(sampling_rates, tges, "pt_jaxsingle_kamp_sixth_ord_big_stencil", "test")
# my_rk4_ho_step.run()

# my_rk4_ho_step = TestRun(sampling_rates, tges, "pt_jaxsingle_kamp_sixth_ord_small_stencil", "big_stencil_against_high_res")
# my_rk4_ho_step.run()
# #
# small_stencil_excerpt_ = TestRun(sampling_rates, tges, "pt_jaxsingle_kamp_sixth_ord_big_stencil", "small_stencil_excerpt_")
# small_stencil_excerpt_.run(t_num=2, freq=1)
# small_stencil_excerpt_.calc_errors(relative_error=True)

# small_stencil_excerpt = TestRun(sampling_rates, tges, "pt_jaxsingle_kamp_sixth_ord_big_stencil", "small_stencil_excerpt")
# small_stencil_excerpt.run(t_num=2500)
# small_stencil_excerpt.calc_errors(relative_error=True)

highres_gen_1000 = TestRun([1], tges, "pt_jaxsingle_kamp_sixth_ord_big_stencil", "High_res_big_data")
highres_gen_1000.run(load_origin_data=False, save_per_step=True, t_num=100, freq=1, create_folder_new=False)

print("done")

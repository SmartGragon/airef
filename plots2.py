import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm, colors, colorbar
from matplotlib import gridspec, pyplot as plt
import numpy as np
import data_utils
import train
import random
import pandas as pd
import accloss
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import importlib

dir_path = os.path.dirname(os.path.realpath(__file__))
vir_white = colors.ListedColormap([[1.,1.,1.]]+cm.viridis.colors)
pick_model = "cs_modis_cgan"
jet = plt.get_cmap('Set1') 
modis_colors = {
    "tau_c": "#e41a1c",
    "p_top": "#377eb8",
    "twp": "#4daf4a",
    "r_e": "#984ea3"
}
mycvals  = [0,1,2,3,4,5,6,7,8]
mycolors = ["#ffffff","#ff0000","#ff6347","#ffff00","#00ff00","#008000","#00ffff","#0000ff","#9a0eea"]

norm1=plt.Normalize(min(mycvals),max(mycvals))
mycmap = matplotlib.colors.ListedColormap(mycolors)

# mycvals  = [1,2,3,4,5,6,7,8]
# mycolors = ["#ff0000","#ff6347","#ffff00","#00ff00","#008000","#00ffff","#0000ff","#9909e9"]
# mynorm=plt.Normalize(min(mycvals),max(mycvals))
# tuples = list(zip(map(mynorm,mycvals), mycolors))
# mycmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

dBZ_norm = colors.Normalize(vmin=-27, vmax=20)

def plot_scene(ax, img, pix_extent=(0.24,1.09), scene_size=64):
    (pix_height, pix_width) = pix_extent
    ax.imshow(img, interpolation='nearest', aspect='auto',
        extent=[0, pix_width*scene_size, 0, pix_height*scene_size],
        cmap=vir_white, norm=dBZ_norm)
    ax.set_xticks([0,20,40,60])
    ax.set_yticks([0,4,8,12])

def plot_label(ax, img, pix_extent=(0.24,1.09), scene_size=64):
    (pix_height, pix_width) = pix_extent
    ax.imshow(img, 
              interpolation='nearest', aspect='auto',
        extent=[0, pix_width*scene_size, 0, pix_height*scene_size],
         cmap=mycmap, norm=norm1)
    ax.set_xticks([0,20,40,60])
    ax.set_yticks([0,4,8,12])
    
def plot_column(ax, y1, y2, pix_extent=(0.24,1.09), 
    scene_size=64, c1='r', c2='b',
    range1=None, range2=None, inv1=False, inv2=False, 
    ticks1=None, ticks2=None,
    log1=False, log2=False):

    pix_width = pix_extent[1]
    x = (np.arange(scene_size)+0.5)*pix_width
    ax.plot(x, y1, color=c1)
    ax.set_xlim((0,scene_size*pix_width))
    ax.set_ylim(*range1)
    ax.tick_params(colors=c1)
    if inv1:
        ax.invert_yaxis()
    if log1:
        ax.set_yscale('log')
    if ticks1 is not None:
        ax.set_yticks(ticks1[0])
        ax.set_yticklabels(ticks1[1])
    ax.set_xticks([0,20,40,60])

    ax2 = ax.twinx()
    ax2.plot(x, y2, color=c2)
    ax2.set_xlim((0,scene_size*pix_width))
    ax2.set_ylim(*range2)
    ax2.tick_params(colors=c2)
    if inv2:
        ax2.invert_yaxis()
    if log2:
        ax2.set_yscale('log')
    if ticks2 is not None:
        ax2.set_yticks(ticks2[0])
        ax2.set_yticklabels(ticks2[1])
    return ax2


def add_dBZ_colorbar(fig, pos):
    cax = fig.add_axes(pos)
    colorbar.ColorbarBase(cax, norm=dBZ_norm,
        cmap=vir_white)
    cax.set_ylabel("Reflectivity [dBZ]")
    cax.yaxis.set_label_position("right")
    return cax

def add_label_colorbar(fig, pos):
    cax = fig.add_axes(pos)
    
    bounds = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]
    bounds2 = [0,1,2,3,4,5,6,7,8]
    # ticklabels=['asc','casc','casca','casca','casc','dg','ewf']
    colorbar.ColorbarBase(cax, cmap=mycmap,
                                norm=norm1,
                                orientation='vertical',
                                boundaries=bounds,
                                ticks = bounds2,
                                spacing = 'uniform')
    
    # colorbar.set_ticklabels(ticklabels , update_ticks=True )
    cax.set_xlabel("         Cloud type")
    cax.yaxis.set_label_position("right")
    cax.set_yticklabels(['No cloud','Cirrus','Altostratus','Altocumulus',
                         'St','Sc','Cumulus','Ns','Deep Convection'])
    return cax

def generate_scenes(gen, modis_vars, modis_mask, noise_dim=64, rng_seed=None,
    zero_noise=False, noise_scale=1.0):

    batch_size = modis_vars.shape[0]
    if zero_noise:
        noise = np.zeros((batch_size, noise_dim), dtype=np.float32)
    else:
        prng = np.random.RandomState(rng_seed)
        noise = prng.normal(scale=noise_scale, size=(batch_size, noise_dim))
    scene_gen = gen.predict([noise, modis_vars, modis_mask])
    return scene_gen


def plot_samples_cmp(gen, scene_real,labels_real, modis_vars, modis_mask, noise_dim=64,
    num_gen=2, rng_seeds=[20183,417662,783924], 
    pix_extent=(0.24,1.09), first_column_num=1):

    scene_gen = [None]*num_gen
    for k in range(num_gen-1):
        scene_gen[k] = generate_scenes(gen, modis_vars, modis_mask, 
            noise_dim=noise_dim, 
            rng_seed=(rng_seeds[k-1] if k>0 else 0),
            zero_noise=(k==0))
        scene_gen[k] = data_utils.rescale_scene_gen(scene_gen[k])
    scene_real = data_utils.rescale_scene(scene_real)

    num_samples = scene_real.shape[0]
    modis_vars_real = data_utils.decode_modis_vars(modis_vars, modis_mask)
    scene_size = scene_real.shape[1]
    tau_c = modis_vars_real["tau_c"][:,:]
    tau_c_range = (1, 150)
    p_top = modis_vars_real["p_top"][:,:]
    p_top_range = (100, 1024)
    r_e = modis_vars_real["r_e"][:,:]
    r_e_range = (0, 70)
    twp = modis_vars_real["twp"][:,:]
    twp_range = (0.25, 25)

    gs = gridspec.GridSpec(num_gen+3,num_samples,
        height_ratios=(1,1)+(2,)*(num_gen+1),hspace=0.1,wspace=0.1)
    fig = plt.figure(figsize=(num_samples*1.5, 2.0+num_gen*1.6))

    for i in range(num_samples):
        ax_tau = plt.subplot(gs[0,i])
        ax_tau.set_title(str(i+first_column_num))
        ax_p = plot_column(ax_tau, tau_c[i,:], p_top[i,:], 
            pix_extent, scene_size, range1=tau_c_range, 
            range2=p_top_range, inv2=True, log1=True, log2=True,
            ticks1=([1, 10, 100], ["1", "10", "100"]),
            ticks2=([1000, 300, 100], ["1000", "300", "100"]),
            c1=modis_colors["tau_c"], c2=modis_colors["p_top"])
        ax_tau.tick_params(labelbottom=False, labelleft=(i==0))
        if i==0:
            ax_tau.set_ylabel("$\\tau_c$", color=modis_colors["tau_c"])
        ax_p.tick_params(labelbottom=False, labelright=(i==num_samples-1))
        if i == num_samples-1:
            ax_p.set_ylabel("$P_\\mathrm{top}$\n$\\mathrm{[hPa]}$", 
                color=modis_colors["p_top"])

        ax_twp = plt.subplot(gs[1,i])
        ax_re = plot_column(ax_twp, twp[i,:], r_e[i,:], 
            pix_extent, scene_size, range1=twp_range, 
            range2=r_e_range, log1=True, log2=False,
            ticks1=([0.25, 1, 4, 16], ["0.25", "1", "4", "16"]),
            ticks2=([0, 20, 40, 60], ["0", "20", "40", "60"]),
            c1=modis_colors["twp"], c2=modis_colors["r_e"])
        ax_twp.tick_params(labelbottom=False, labelleft=(i==0))
        if i==0:
            ax_twp.set_ylabel("$\\mathrm{CWP}$\n$\\mathrm{[g\\,m^{-2}]}$", 
                color=modis_colors["twp"])
        ax_re.tick_params(labelbottom=False, labelright=(i==num_samples-1))
        if i == num_samples-1:
            ax_re.set_ylabel("$r_e$\n$\\mathrm{[\\mu m]}$", 
                color=modis_colors["r_e"])

        for k in range(num_gen):
            
            ax_gen = plt.subplot(gs[2+k,i])
            if (k==0):
                plot_scene(ax_gen, scene_gen[k][i,:,:,0], pix_extent, scene_size)
            else:
                plot_scene(ax_gen, scene_real[i,:,:,0], pix_extent, scene_size)
                
            ax_gen.tick_params(labelbottom=False, labelleft=(i==0))
            if i==0:
                if k==0:
                    label = "Gen. ($\\mathbf{z}=\\mathbf{0}$)\nAltitude [km]"
                # elif k!=num_gen-1:
                #     label = "Generated\nAltitude [km]"
                else:
                    label = "Real\nAltitude [km]"
                ax_gen.set_ylabel(label)
            
                
 
        ax_real = plt.subplot(gs[-1,i])
        plot_label(ax_real, labels_real[i,:,:,0], pix_extent, scene_size)
        ax_real.tick_params(labelleft=(i==0))
        if i==0:
            ax_real.set_ylabel("Real cloud type\nAltitude [km]")
        ax_real.set_xlabel("Distance [km]")

    add_dBZ_colorbar(fig, [0.91, 0.32, 0.018, 0.34])
    add_label_colorbar(fig, [0.91, 0.06, 0.018, 0.24])
    return fig


def plot_samples_cmp_all(gen, scene_real, modis_vars, modis_mask,labels_real):
    # these have been hand-picked to illustrate specific cases
    samples_sel_1 = [2472, 591,3302,2511 , 3422, 4859, 2088,5391 ]
    samples_sel_2 = [2896, 5220,2450 ,2152 ,2080, 79,925 ,3145 ]
    # samples_sel_1 = [3175, 4864,5316,3022 , 3803, 4173, 1468,3462 ]
    # samples_sel_2 = [2896, 5220,2450 ,2152 ,2080, 79,925 ,3145 ]
    # these were selected randomly from the validation dataset
    samples_rnd = []
    for i in range(8):
        samples_rnd.append(random.sample(range(1, 900), 8))
    np.set_printoptions(threshold=10000)
    # print(labels_real[855,:,:])
    # print(labels_real[307,:,:])


    samples = [samples_sel_1, samples_sel_2] + samples_rnd
    plot_names = ["real_gen_cmp_sel-1", "real_gen_cmp_sel-2"] + \
        ["real_gen_cmp_rnd-{}".format(i) for i in range(len(samples_rnd))]
    first_column_num = [1]*len(plot_names)
    first_column_num[1] = 9

    for (s, fn, fcn) in zip(samples, plot_names, first_column_num):
        plot_samples_cmp(gen, scene_real[s,...], labels_real[s,...],
            modis_vars[s,...], modis_mask[s,...], first_column_num=fcn)
        plt.savefig("../figures/{}.svg".format(fn), bbox_inches='tight', dpi=1200, format='svg')
        plt.close()

    dist_plot_names = \
        ["gen_dist-{}.svg".format(i) for i in range(len(samples_sel_2))]
    for (s,fn) in zip(samples_sel_2, dist_plot_names):
        plot_distribution(gen, modis_vars, modis_mask, s)
        plt.savefig("../figures/{}".format(fn), bbox_inches='tight', dpi=1200, format='svg')
        plt.close()


def plot_distribution(gen, modis_vars, modis_mask, sample_num,
    noise_dim=64, grid_size=(8,8)):

    gs = gridspec.GridSpec(grid_size[0], grid_size[1],
        hspace=0.1,wspace=0.1)
    fig = plt.figure(figsize=(grid_size[1]*1.5, grid_size[0]*1.5))

    num_samples = grid_size[0]*grid_size[1]
    ind_array = [sample_num]*num_samples
    modis_vars_s = modis_vars[ind_array,...]
    modis_mask_s = modis_mask[ind_array,...]

    scene_gen = generate_scenes(gen, modis_vars_s, modis_mask_s, 
        noise_dim=noise_dim, noise_scale=2.0,
        rng_seed=335573)
    scene_gen = data_utils.rescale_scene_gen(scene_gen)

    k = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            ax_gen = plt.subplot(gs[i,j])
            plot_scene(ax_gen, scene_gen[k,:,:,0])
            k += 1
            if j==0:
                ax_gen.set_ylabel("Altitude [km]")
            if i==grid_size[0]-1:
                ax_gen.set_xlabel("Distance [km]")
            ax_gen.tick_params(labelbottom=(i==grid_size[0]-1), 
                labelleft=(j==0))

    add_dBZ_colorbar(fig, [0.91, 0.41, 0.018, 0.4])


def plot_gen_vary(gen, modis_vars, modis_mask,
    scene_size=64, vary_space=(-2,2,9), noise_dim=64):
    # s_vary = 10939

    modis_var_dim = modis_vars.shape[-1]
    mask_bool = modis_mask[...,0].astype(bool)
    means = [modis_vars[...,i][mask_bool].mean() for 
        i in range(modis_var_dim)]
    stds = [modis_vars[...,i][mask_bool].std() for 
        i in range(modis_var_dim)]

    grid_size = (modis_var_dim, vary_space[2])
    N = grid_size[0]*grid_size[1]
    gs = gridspec.GridSpec(grid_size[0], grid_size[1],
        hspace=0.1,wspace=0.1)
    fig = plt.figure(figsize=(grid_size[1]*1.5, grid_size[0]*1.5))

    vary_samples = np.linspace(*vary_space)
    num_samples = len(vary_samples)

    modis_vars = np.zeros((N,scene_size,modis_var_dim),
        dtype=np.float32)
    modis_mask = np.ones((N,scene_size,1), dtype=np.float32)

    k = 0
    for i in range(modis_var_dim):
        modis_vars[...,i] = means[i]
        for s in vary_samples:
            modis_vars[k,:,i] = means[i]+stds[i]*s
            k += 1
    #modis_vars[0,:,0] = -1

    scene_gen = np.zeros((N,64,64,1), dtype=np.float32)
    #for i in range(N):
    #    scene_gen[i:i+1,...] = generate_scenes(gen,
    #        modis_vars[i:i+1,...], modis_mask[i:i+1,...],
    #        noise_dim=noise_dim, zero_noise=True)
    scene_gen = generate_scenes(gen, modis_vars, modis_mask, 
        noise_dim=noise_dim, zero_noise=True,
        rng_seed=743708)
    scene_gen = data_utils.rescale_scene_gen(scene_gen)

    #return scene_gen

    var_labels = {
        0: "$\\tau_c'$",
        1: "$P_\\mathrm{top}'$",
        2: "$r_e'$",
        3: "$\\mathrm{CWP}'$"
    } 

    k=0
    for i in range(modis_var_dim):
        for j in range(num_samples):
            ax_gen = plt.subplot(gs[i,j])
            plot_scene(ax_gen, scene_gen[k,:,:,0])
            k += 1
            if j==0:
                ax_gen.set_ylabel(var_labels[i]+"\nAltitude [km]")
            ax_gen.tick_params(labelbottom=(i==modis_var_dim-1), 
                labelleft=(j==0))
            if i==grid_size[0]-1:
                sig_label = "${:+.1f}\\sigma$".format(vary_samples[j])
                if vary_samples[j]==0:
                    sig_label = sig_label.replace('+','')
                ax_gen.set_xlabel("Distance [km]\n"+sig_label)

    add_dBZ_colorbar(fig, [0.91, 0.11, 0.018, 0.77])


def make_dBZ_hist(batch, dBZ_range=(-30,20.1,1)):
    dBZ_bins = np.arange(*dBZ_range)
    hist_shape = (
        batch.shape[1],
        len(dBZ_bins)-1
    )
    hist = np.zeros(hist_shape)
    for i in range(hist_shape[0]):
        data = batch[:,i,...].ravel()
        data = data[np.isfinite(data)]
        hist_height = np.histogram(data, dBZ_bins)[0]
        hist[i,:] = hist_height

    return hist


def generated_hist(gen, modis_vars, modis_mask,labels_real,scene_real, scenes_fn,
    batch_size=1, noise_dim=64):
    df = pd.DataFrame(columns=["Threshold","result","Class"])
    df2 = pd.DataFrame(columns=["cloud_type","POD","size","index"])
    df3 = pd.DataFrame(columns=["cloud_type","POD","size","index"])
    hist = None
    N = modis_vars.shape[0]
    for i in range(0,N,batch_size):
        vars_batch = modis_vars[i:i+batch_size,...]
        mask_batch = modis_mask[i:i+batch_size,...]
        labels_batch = labels_real[i:i+batch_size,...]
        scene_real_batch0 = scene_real[i:i+batch_size,...]
        gen_batch0 = generate_scenes(gen, vars_batch, mask_batch,
            noise_dim=noise_dim, rng_seed=493411+i)
        # N = vars_batch.shape[0]
        glb_dct = sys.modules['__main__'].__dict__
        # glb_dct["THRESHOLD"] = -25
        # importlib.reload(accloss)
        # for j in range(vars_batch.shape[0]):
        #     for k,bb in [(1,'Cirrus'),(2,'Altostratus'),(3,'Altocumulus'),(4,'St'),
        #               (5,'Sc'),(6,'Cumulus'),(7,'Ns'),(8,'Deep Convection')]:
        #         if (labels_batch[j,...].__contains__(k)):
        #             # if (k==4):
        #             #     # np.set_printoptions(threshold=10000)
        #             #     print(i+j)
        #             #     # figtmp = plt.figure()
        #             #     # ax = plt.subplot(111)
        #             #     # plot_label(ax,labels_batch[j,...], (0.24,1.09), 64)
        #             #     # figtmp.savefig("./{}.jpg".format(j))
        #             pixels = np.where(labels_batch[j,...]==k,1,0).sum()
        #             if (pixels>30):
        #                 sel_gen = data_utils.rescale_scene_gen(gen_batch0)
        #                 sel_real = data_utils.rescale_scene(scene_real_batch0)
        #                 sel_gen1  = np.where(labels_batch[j,...]==k,sel_gen,-26.5)
        #                 np.nan_to_num(sel_gen1,nan=-99,copy=False)
        #                 sel_real1 = np.where(labels_batch[j,...]==k,sel_real,-26)
                        
        #                 rmse41 = accloss.POD(sel_real1,sel_gen1)
        #                 # rmse41 = accloss.POD(sel_real1,sel_gen1)*10/(math.sqrt(pixels))
        #                 df2.loc[len(df2)] = [bb,rmse41,pixels,i+j]
        #                 # print(j,rmse41,">30,-25")
        #             else:
        #                 df2.loc[len(df2)] = [bb,np.nan,pixels,i+j]
        #         else:
        #             continue
        glb_dct["THRESHOLD"] = 0
        importlib.reload(accloss)
        for j in range(vars_batch.shape[0]):
            for k,bb in [(8,'Deep Convection'),(2,'Altostratus'),(3,'Altocumulus'),(4,'St'),
                      (5,'Sc'),(6,'Cumulus'),(7,'Ns'),(1,'Cirrus')]:
                if (labels_batch[j,...].__contains__(k)):
                    # if (k==4):
                    #     # np.set_printoptions(threshold=10000)
                    #     print(i+j)
                    #     # figtmp = plt.figure()
                    #     # ax = plt.subplot(111)
                    #     # plot_label(ax,labels_batch[j,...], (0.24,1.09), 64)
                    #     # figtmp.savefig("./{}.jpg".format(j))
                    pixels = np.where(labels_batch[j,...]==k,1,0).sum()
                    if (pixels>30):
                        sel_gen = data_utils.rescale_scene_gen(gen_batch0)
                        sel_real = data_utils.rescale_scene(scene_real_batch0)
                        sel_gen1  = np.where(labels_batch[j,...]==k,sel_gen,-26.5)
                        np.nan_to_num(sel_gen1,nan=-99,copy=False)
                        sel_real1 = np.where(labels_batch[j,...]==k,sel_real,-26)
                        
                        # rmse41 = accloss.POD(sel_real1,sel_gen1)*10/(math.sqrt(pixels))
                        rmse41 = accloss.POD(sel_real1,sel_gen1)
                        df3.loc[len(df3)] = [bb,rmse41,pixels,i+j]
                        # print(j,rmse41,">30,5")
                    else:
                        # print(j,"<30")
                        df3.loc[len(df3)] = [bb,np.nan,pixels,i+j]
                else:
                    continue
        # scene_real_batch = data_utils.rescale_scene(scene_real_batch0)
        # gen_batch = data_utils.rescale_scene_gen(gen_batch0)
        # glb_dct = sys.modules['__main__'].__dict__
        # glb_dct["THRESHOLD"] = -25
        # importlib.reload(accloss)
        # rmse,ts,far,pod,hss=accloss.calculate\
        #     (scene_real_batch,gen_batch)
        # # df.loc[len(df)] = ['-25dBZ',rmse,'RMSE']
        # df.loc[len(df)] = ['-25dBZ',ts,'TS']
        # # df.loc[len(df)] = ['-25dBZ',ets,'ETS']
        # df.loc[len(df)] = ['-25dBZ',far,'FAR']
        # # df.loc[len(df)] = ['-25dBZ',mar,'MAR']
        # df.loc[len(df)] = ['-25dBZ',pod,'POD']
        # df.loc[len(df)] = ['-25dBZ',hss,'HSS']
        # # df.loc[len(df)] = ['-25dBZ',bss,'BSS']
        
        # glb_dct["THRESHOLD"] = -15
        # importlib.reload(accloss)
        # # print(glb_dct["THRESHOLD"])
        # rmse,ts,far,pod,hss=accloss.calculate\
        #     (scene_real_batch,gen_batch)
        # # df.loc[len(df)] = ['-15dBZ',rmse,'RMSE']
        # df.loc[len(df)] = ['-15dBZ',ts,'TS']
        # # df.loc[len(df)] = ['-15dBZ',ets,'ETS']
        # df.loc[len(df)] = ['-15dBZ',far,'FAR']
        # # df.loc[len(df)] = ['-15dBZ',mar,'MAR']
        # df.loc[len(df)] = ['-15dBZ',pod,'POD']
        # df.loc[len(df)] = ['-15dBZ',hss,'HSS']
        # df.loc[len(df)] = ['-15dBZ',bss,'BSS']
        
        # glb_dct["THRESHOLD"] = -5
        # importlib.reload(accloss)
        # rmse,ts,far,pod,hss=accloss.calculate\
        #     (scene_real_batch,gen_batch)
        # # df.loc[len(df)] = ['-5dBZ',rmse,'RMSE']
        # df.loc[len(df)] = ['-5dBZ',ts,'TS']
        # # df.loc[len(df)] = ['-5dBZ',ets,'ETS']
        # df.loc[len(df)] = ['-5dBZ',far,'FAR']
        # # df.loc[len(df)] = ['-5dBZ',mar,'MAR']
        # df.loc[len(df)] = ['-5dBZ',pod,'POD']
        # df.loc[len(df)] = ['-5dBZ',hss,'HSS']
        # # df.loc[len(df)] = ['-5dBZ',bss,'BSS']
        
        # glb_dct["THRESHOLD"] = 5
        # importlib.reload(accloss)
        # rmse,ts,far,pod,hss=accloss.calculate\
        #     (scene_real_batch,gen_batch)
        # # df.loc[len(df)] = ['5dBZ',rmse,'RMSE']
        # df.loc[len(df)] = ['5dBZ',ts,'TS']
        # # df.loc[len(df)] = ['5dBZ',ets,'ETS']
        # df.loc[len(df)] = ['5dBZ',far,'FAR']
        # # df.loc[len(df)] = ['5dBZ',mar,'MAR']
        # df.loc[len(df)] = ['5dBZ',pod,'POD']
        # df.loc[len(df)] = ['5dBZ',hss,'HSS']
        # # df.loc[len(df)] = ['5dBZ',bss,'BSS']
        
        # glb_dct["THRESHOLD"] = 15
        # importlib.reload(accloss)
        # rmse,ts,far,pod,hss=accloss.calculate\
        #     (scene_real_batch,gen_batch)
        # # df.loc[len(df)] = ['15dBZ',rmse,'RMSE']
        # df.loc[len(df)] = ['15dBZ',ts,'TS']
        # # df.loc[len(df)] = ['15dBZ',ets,'ETS']
        # df.loc[len(df)] = ['15dBZ',far,'FAR']
        # # df.loc[len(df)] = ['15dBZ',mar,'MAR']
        # df.loc[len(df)] = ['15dBZ',pod,'POD']
        # df.loc[len(df)] = ['15dBZ',hss,'HSS']
        # # df.loc[len(df)] = ['15dBZ',bss,'BSS']
        # rmse = accloss.calculatermse(scene_real_batch0,gen_batch0)
        # # glb_dct["THRESHOLD"] = 0.6
        # # importlib.reload(accloss)
        # # rmse,ts,far,pod,hss=accloss.calculate\
        # #     (scene_real_batch,gen_batch)
        # # df.loc[len(df)] = [0.6,rmse,'RMSE']
        # # df.loc[len(df)] = [0.6,ts,'TS']
        # # # df.loc[len(df)] = [0.6,ets,'ETS']
        # # df.loc[len(df)] = [0.6,far,'FAR']
        # # # df.loc[len(df)] = [0.6,mar,'MAR']
        # # df.loc[len(df)] = [0.6,pod,'POD']
        # # df.loc[len(df)] = [0.6,hss,'HSS']
        # # df.loc[len(df)] = [0.6,bss,'BSS']
        
        # #meanRMSE = df[df['Class']=='RMSE'][['result']].mean().iloc[0,0]
        # #meanTS = df[df['Class']=='TS'][['result']].mean().iloc[0,0]
        # #meanETS = df[df['Class']=='ETS'][['result']].mean().iloc[0,0]
        # #meanFAR = df[df['Class']=='FAR'][['result']].mean().iloc[0,0]
        # #meanMAR = df[df['Class']=='MAR'][['result']].mean().iloc[0,0]
        # #meanPOD = df[df['Class']=='POD'][['result']].mean().iloc[0,0]
        # #meanHSS = df[df['Class']=='HSS'][['result']].mean().iloc[0,0]
        # #meanBSS = df[df['Class']=='BSS'][['result']].mean().iloc[0,0]
        # # meanRMSE = df[df['Class']=='RMSE'][['result']].mean()
        # meanTS = df[df['Class']=='TS'][['result']].mean()
        # # meanETS = df[df['Class']=='ETS'][['result']].mean()
        # meanFAR = df[df['Class']=='FAR'][['result']].mean()
        # # meanMAR = df[df['Class']=='MAR'][['result']].mean()
        # meanPOD = df[df['Class']=='POD'][['result']].mean()
        # meanHSS = df[df['Class']=='HSS'][['result']].mean()
        # # meanBSS = df[df['Class']=='BSS'][['result']].mean()
        # # rank = rmse
        
        # df.to_csv("../data/acc_data.csv")  
        
        # df2.to_csv("../data/cloudtype_data-25.csv")
        df3.to_csv("../data/highcloudtype_data0.csv")
        
        # large = 22; med = 16; small = 12
        # params = {'axes.titlesize': large,
        #           'legend.fontsize': med,
        #           'figure.figsize': (16, 10),
        #           'axes.labelsize': med,
        #           'axes.titlesize': med,
        #           'xtick.labelsize': med,
        #           'ytick.labelsize': med,
        #           'figure.titlesize': large}
        # plt.rcParams.update(params)
        # plt.style.use('seaborn-whitegrid')
        # sns.set_style("white")
        # # Draw Plot
        # plt.figure(figsize=(13,10), dpi= 80)
        # sns.boxplot(x='Class', y='result', data=df, hue='Threshold')
        # #sns.stripplot(x='Class', y='result', data=df, color='black', size=3, jitter=1)
        
        # for i in range(len(df['Class'].unique())-1):
        #     plt.vlines(i+.5, 0.1, 1, linestyles='solid', colors='gray', alpha=0.2)
        
        # # Decoration
        # plt.title('GAN Box Plot(model:'+pick_model+'dataset:'+scenes_fn.split('/')[-1]+')', fontsize=22)
        # plt.legend(title='Threshold')
        # ax=plt.gca()
        # # Rank = rank.item()
        # plt.text(0.05, 0.95,f"RMSE= {rmse:.2f}", color='red', transform=ax.transAxes)
        # gen_batch = data_utils.rescale_scene_gen(gen_batch)
        # plt.savefig("../data/accuracy.jpg")
        
        # hist_batch = make_dBZ_hist(gen_batch)
        # if hist is None:
        #     hist = hist_batch
        # else:
        #     hist += hist_batch

    return hist


def real_hist(scenes_real):
    scenes_real = data_utils.rescale_scene(scenes_real)
    return make_dBZ_hist(scenes_real)


def plot_hist(hist_real, hist_gen, pix_extent=(0.24,1.09)):
    gs = gridspec.GridSpec(1, 3,
        hspace=0.1,wspace=0.15)
    fig = plt.figure(figsize=(9,4))

    scene_size = hist_real.shape[0]
    hist_real_norm = hist_real / hist_real.sum()
    hist_gen_norm = hist_gen / hist_gen.sum()
    hist_diff = hist_gen_norm - hist_real_norm
    norm = colors.Normalize(0,
        max(hist_real_norm.max(),hist_gen_norm.max()))
    max_diff = abs(hist_diff).max()
    norm_diff = colors.Normalize(-max_diff,max_diff)

    ax = plt.subplot(gs[0,0])
    plt.imshow(hist_real_norm, aspect='auto', norm=norm,
        extent=[-30,20,0,scene_size*pix_extent[0]],
        cmap=vir_white)
    plt.xlabel("Reflectivity [dBZ]")
    plt.ylabel("Altitude [km]")
    ax.set_xticks([-20,-10,0,10,20])
    ax.set_yticks([0,4,8,12])
    cb = plt.colorbar(orientation='horizontal')
    cb.ax.set_xlabel("real")
    cb.ax.tick_params(labelsize=8)

    ax = plt.subplot(gs[0,1])
    plt.imshow(hist_gen_norm, aspect='auto', norm=norm,
        extent=[-30,20,0,scene_size*pix_extent[0]],
        cmap=vir_white)
    plt.xlabel("Reflectivity [dBZ]")
    ax.set_xticks([-20,-10,0,10,20])
    ax.set_yticks([0,4,8,12])
    cb = plt.colorbar(orientation='horizontal')
    cb.ax.set_xlabel("generated")
    cb.ax.tick_params(labelsize=8)

    ax = plt.subplot(gs[0,2])
    plt.imshow(hist_diff, aspect='auto', norm=norm_diff,
        extent=[-30,20,0,scene_size*pix_extent[0]],
        cmap="RdBu_r")
    plt.xlabel("Reflectivity [dBZ]")
    ax.set_xticks([-20,-10,0,10,20])
    ax.set_yticks([0,4,8,12])
    cb = plt.colorbar(orientation='horizontal')
    cb.ax.set_xlabel("Occurrence bias (gen. - real)")
    cb.ax.tick_params(labelsize=8)


def load_data_and_models(scenes_fn, model_name=pick_model, 
    epoch=45, scene_size=64, modis_var_dim=4, noise_dim=64, 
    lr_disc=0.0001, lr_gan=0.0002):

    scenes = data_utils.load_cloudsat_scenes(scenes_fn, shuffle_seed=214101)
    

    (gen, disc, gan, opt_disc, opt_gan) = train.create_models(
        scene_size, modis_var_dim, noise_dim, lr_disc, lr_gan)

    train.load_model_state(gen, disc, gan, model_name, epoch)

    return (scenes, gen, disc, gan)


def plot_all(scenes_fn, model_name=pick_model):
    (scenes, gen, disc, gan) = load_data_and_models(scenes_fn,
        model_name=model_name)
    labels = data_utils.load_cloudsat_labels(scenes_fn, shuffle_seed=214101)
    (scene_real, modis_vars, modis_mask) = scenes["validate"]
    labels_real = labels["validate"]
    
    plot_samples_cmp_all(gen, scene_real, modis_vars, modis_mask,labels_real)

    # plot_gen_vary(gen, modis_vars, modis_mask)
    # plt.savefig("../figures/gen_vary.pdf", bbox_inches='tight')
    # plt.close()

    # hist_gen = generated_hist(gen, modis_vars, modis_mask,labels_real,scene_real,scenes_fn)
    # hist_real = real_hist(scene_real)
    # plot_hist(hist_real, hist_gen)
    # plt.savefig("../figures/real_gen_hist.pdf", bbox_inches='tight')
    # plt.close()




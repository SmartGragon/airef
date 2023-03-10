import argparse
# import h5py
if __name__ == "__main__":
    glb_dct = globals()
    glb_dct["THRESHOLD"] = 0      
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="train or plot")
    parser.add_argument('--scenes_file', type=str, 
        help="CloudSat scenes file")
    parser.add_argument('--run_name', type=str, default="",
        help="Suffix to use for this training run")
    
    args = parser.parse_args()
    mode = args.mode
    scenes_fn = args.scenes_file

    if mode == "train":
        import train
        train.train_cs_modis_cgan_full(scenes_fn)
    elif mode == "plot":
        import plots
        plots.plot_all(scenes_fn)
    elif mode == "plot2":
        import plots2
        plots2.plot_all(scenes_fn)
    elif mode == "plot3":
        import plots3
        plots3.plot_all(scenes_fn)
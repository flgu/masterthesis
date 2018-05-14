#%% import
import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
import os

def main():

    path = r"D:\masterthesis\implementations\model_2\reaction_boundary_conditions"

    if not os.getcwd() == path:
        print("change to reaction boundary condition working directory")
        os.chdir(os.path.join(path))
        print("successfully changed to:",os.getcwd())

    # import solutions from BV voltage sweeps
    name_list = ["sweep_matrix.npy", "sweep_matrix_1.npy", "sweep_matrix_3.npy", "desktop_sweep_D_0.npy", "desktop_sweep_D_1.npy"]
    sol_list = []

    for name in name_list:
        print(name)
        sol_list.append(np.load(name))


    return sol_list, name_list

def plotting( sol_list, name_list):
    
    # list of colors
    colors = ["blue", "red", "blue", "red", "green","green", "yellow"]
    linestyles = ["-", "-", "--", "--", "-", "--"]

    # plot
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    


    # preallocate arrays for searching min and max elements
    min_element_x = np.zeros(sol_list.__len__(), dtype = np.float64)
    max_element_x = np.zeros(sol_list.__len__(), dtype = np.float64)
    min_element_y = np.zeros(sol_list.__len__(), dtype = np.float64)
    max_element_y = np.zeros(sol_list.__len__(), dtype = np.float64)

    for i in range(sol_list.__len__()):


        ax.plot(sol_list[i][:,0], sol_list[i][:,1], color = colors[i], label = name_list[i], ls = linestyles[i])
        
        # get min element x
        min_element_x[i] = sol_list[i][:,0].min()
        max_element_x[i] = sol_list[i][:,0].max()
        min_element_y[i] = sol_list[i][:,1:].min()
        max_element_y[i] = sol_list[i][:,1:].max()
    
    # show x, y axis
    ax.plot([min_element_x.min() + 0.1 * min_element_x.min(),
            max_element_x.max() + 0.1 *  max_element_x.max()], [0,0], color = "black", ls = "--", lw = 0.8,)
    ax.plot([0,0], [min_element_y.min() + 0.1 * min_element_y.min(),
            max_element_y.max() + 0.1 *  max_element_y.max()], color = "black", ls = "--", lw = 0.8)

    # set xlimits, ylimits
    ax.set_xlim([min_element_x.min() + 0.1 * min_element_x.min(), max_element_x.max() + 0.1 *  max_element_x.max()]);
    ax.set_ylim([min_element_y.min() + 0.1 * min_element_y.min(), max_element_y.max() + 0.1 *  max_element_y.max()]);

    # grid
    ax.grid(b = True, which = "major", axis = "both")

    # get legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    # tight layout
    fig.tight_layout()

    plt.show()



if __name__ == "__main__":

    sol_list, name_list = main()
    plotting( sol_list, name_list)
    
    

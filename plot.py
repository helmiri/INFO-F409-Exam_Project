import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize) #allows to print the whole array in the terminal
plt.rc("font", **{"size" : 7}) #change the general font size

def plot(data):
    """
    General function to plot our data.

    :param dta: The data to plot.
    """
    games = ["Prisoner's Dilemma", "Chicken", "Stag Hunt"]
    for game in range(len(data)):
        plt.subplot(3, 1, game+1)
        plt.ylim(0, 1.1)
        plt.title(games[game], fontweight='bold')
        plt.ylabel("Cooperation rate")
        plt.xlabel("Iterations")
        plt.plot(data[game])
    plt.tight_layout()
    plt.show()

def plot_SRE_over_A0(data):
    """
    Plot for the impact of the aspiration level on the SRE rate and
    a second plot with greed and fear.

    :param dta: The data to plot.
    """
    games = ["Prisoner's Dilemma", "Chicken", "Stag Hunt"]
    x = np.linspace(0, 4, 40)
    for game in range(len(data)):
        plt.subplot(3, 1, game+1)
        plt.ylim(0, 1.1)
        plt.title(games[game], fontweight='bold')
        plt.ylabel("SRE rate")
        plt.xlabel("A0")
        plt.plot(x, data[game][0])
    plt.tight_layout()
    plt.show()

    #Greed and fear plot
    for game in range(len(data)):
        plt.subplot(3, 1, game+1)
        plt.ylim(0, 1.1)
        plt.plot(x, data[game][0], color='black', linestyle='dashed', label='classic', alpha=0.6)
        if game != 1:
            plt.plot(x, data[game][1], color='blue', label='fear', alpha=0.6)
        if game != 2:
            plt.plot(x, data[game][2], color='red', label='greed', alpha=0.6)
        plt.title(games[game], fontweight='bold')
        plt.ylabel("SRE rate")
        plt.xlabel("A0")
        plt.legend()
    plt.tight_layout()
    plt.show()

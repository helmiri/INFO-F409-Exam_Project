import sys

import matplotlib.pyplot as plt
import numpy as np

from runner import compute_propo_coop_mut, read_data


class Plot:
    def __init__(self):
        np.set_printoptions(threshold=sys.maxsize)  # allows to print the whole array in the terminal
        plt.rc("font", **{"size": 7})  # change the general font size

    def plot_cooperation(self, data, parameters_name):
        """
        General function to plot our data.

        :param dta: The data to plot.
        """
        games = ["Prisoner's Dilemma", "Chicken", "Stag Hunt"]
        for game in range(len(data)):
            plt.subplot(3, 1, game + 1)
            plt.ylim(0, 1.1)
            plt.xlim(0, 500)
            plt.title(games[game], fontweight='bold')
            plt.ylabel(parameters_name)
            plt.xlabel("Iterations")
            plt.plot(data[game])
        plt.tight_layout()
        plt.show()

    def plot_SRE_over_A0(self, data):
        """
        Plot for the impact of the aspiration level on the SRE rate and
        a second plot with greed and fear.

        :param dta: The data to plot.
        """
        games = ["Prisoner's Dilemma", "Chicken", "Stag Hunt"]
        x = np.linspace(0, 4, 40)
        for game in range(len(data)):
            plt.subplot(3, 1, game + 1)
            plt.ylim(0, 1.1)
            plt.title(games[game], fontweight='bold')
            plt.ylabel("SRE rate")
            plt.xlabel("A0")
            plt.plot(x, data[game][0])
        plt.tight_layout()
        plt.show()

        # Greed and fear plot
        for game in range(len(data)):
            plt.subplot(3, 1, game + 1)
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


def main():
    parameters_list = list()
    if len(sys.argv) == 7:
        for i in range(len(sys.argv[1:])):
            sys.argv[i] = sys.argv[i].replace(".", "-")
        parameters_list.append(sys.argv[1:])
    elif len(sys.argv) == 2:
        with open(sys.argv[1], "r") as parameters_file:
            lines = parameters_file.readlines()
        for line in lines:
            line = line.replace(".", "-").strip("\n")
            parameters_list.append(line.split(" "))
    plot = Plot()
    coop_by_game = []
    for parameters in parameters_list:
        for game_name in ["PD", "SG", "CH"]:
            agt = read_data("data/agent_act_probs_{0}_{1}_{2}_{3}_{4}_{5}_{6}.p".format(game_name, *parameters))
            coop_by_game.append(agt[10])
            print(game_name + " convergence rate: " + str(compute_propo_coop_mut(agt)))
        plot.plot_cooperation(coop_by_game, "Proba. of cooperation")
        coop_by_game.clear()


if __name__ == '__main__':
    main()

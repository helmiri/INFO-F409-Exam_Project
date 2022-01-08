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

    def plot_SRE_aspiration(self, coop_PD, coop_CH, coop_SG):
        """
        Plot for the impact of the aspiration level on the SRE rate and
        a second plot with greed and fear.

        :param dta: The data to plot.
        """
        games = ["Prisoner's Dilemma", "Chicken", "Stag Hunt"]
        x = np.linspace(0, 4, 41)
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.ylim(0, 1.1)
            plt.title(games[i], fontweight='bold')
            plt.ylabel("%SRE")
            plt.xlabel("Aspiration")
            if i == 0:
                plt.plot(x, coop_PD)
            elif i == 1:
                plt.plot(x, coop_CH)
            else:
                plt.plot(x, coop_SG)
        plt.tight_layout()
        plt.show()

    def plot_SRE_greed_fear(self, PD_classic, PD_fear, PD_greed, SG_classic, SG_fear, SG_greed, CH_classic, CH_fear,
                             CH_greed):
        """
        Plot for the impact of the aspiration level on the SRE rate and
        a second plot with greed and fear.

        :param dta: The data to plot.
        """
        games = ["Prisoner's Dilemma", "Chicken", "Stag Hunt"]
        x = np.linspace(0, 4, 41)
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.ylim(0, 1.1)
            plt.title(games[i], fontweight='bold')
            plt.ylabel("%SRE")
            plt.xlabel("Aspiration")
            if i == 0:
                plt.plot(x, PD_classic, label="classic", linestyle="dashed")
                plt.plot(x, PD_fear, label="fear")
                plt.plot(x, PD_greed, label="greed")
                plt.legend(["classic", "fear", "greed"])
            elif i == 1:
                plt.plot(x, CH_classic, label="classic", linestyle="dashed")
                plt.plot(x, CH_fear, label="fear")
                plt.plot(x, CH_greed, label="greed")
                plt.legend(["classic", "fear", "greed"])
            else:
                plt.plot(x, SG_classic, label="classic", linestyle="dashed")
                plt.plot(x, SG_fear, label="fear")
                plt.plot(x, SG_greed, label="greed")
                plt.legend(["classic", "fear", "greed"])
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
        print("PD : ", sum(coop_by_game[0]) / len(coop_by_game[0]))
        print("SG : ", sum(coop_by_game[1]) / len(coop_by_game[1]))
        print("CH : ", sum(coop_by_game[2]) / len(coop_by_game[2]))
        plot.plot_cooperation(coop_by_game, "Proba. of cooperation")
        coop_by_game.clear()


def mainSRE():
    plot = Plot()
    coop_PD = []
    coop_SG = []
    coop_CH = []
    aspirations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
        , 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3
        , 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
    for i in range(41):
        for game_name in ["PD", "SG", "CH"]:
            if i % 10 == 0:
                agt = read_data(
                    "data/agent_act_probs_" + game_name + "_classic_0_" + str(int(aspirations[i])) + "_0-5_1000_250.p")
            else:
                aspirations[i] = str(aspirations[i]).replace(".", "-")
                agt = read_data(
                    "data/agent_act_probs_" + game_name + "_classic_0_" + str(aspirations[i]) + "_0-5_1000_250.p")
            if game_name == "PD":
                coop_PD.append(compute_propo_coop_mut(agt))
            elif game_name == "SG":
                coop_SG.append(compute_propo_coop_mut(agt))
            else:
                coop_CH.append(compute_propo_coop_mut(agt))
            print(game_name + " convergence rate: " + str(compute_propo_coop_mut(agt)))
    plot.plot_SRE_aspiration(coop_PD, coop_CH, coop_SG)
    coop_CH.clear()
    coop_SG.clear()
    coop_PD.clear()


def mainGreedFearSRE():
    plot = Plot()
    PD_classic = []
    PD_fear = []
    PD_greed = []
    SG_classic = []
    SG_fear = []
    SG_greed = []
    CH_classic = []
    CH_fear = []
    CH_greed = []
    aspirations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
        , 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3
        , 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
    for i in range(41):
        for game_name in ["PD", "SG", "CH"]:
            if i % 10 == 0:
                agtClassic = read_data(
                    "data/agent_act_probs_" + game_name + "_classic_0_" + str(int(aspirations[i])) + "_0-5_1000_250.p")
                agtFear = read_data(
                    "data/agent_act_probs_" + game_name + "_fear_0_" + str(int(aspirations[i])) + "_0-5_1000_250.p")
                agtGreed = read_data(
                    "data/agent_act_probs_" + game_name + "_greed_0_" + str(int(aspirations[i])) + "_0-5_1000_250.p")
            else:
                aspirations[i] = str(aspirations[i]).replace(".", "-")
                agtClassic = read_data(
                    "data/agent_act_probs_" + game_name + "_classic_0_" + str(aspirations[i]) + "_0-5_1000_250.p")
                agtFear = read_data(
                    "data/agent_act_probs_" + game_name + "_fear_0_" + str(aspirations[i]) + "_0-5_1000_250.p")
                agtGreed = read_data(
                    "data/agent_act_probs_" + game_name + "_greed_0_" + str(aspirations[i]) + "_0-5_1000_250.p")
            if game_name == "PD":
                PD_classic.append(compute_propo_coop_mut(agtClassic))
                PD_fear.append(compute_propo_coop_mut(agtFear))
                PD_greed.append(compute_propo_coop_mut(agtGreed))
            elif game_name == "SG":
                SG_classic.append(compute_propo_coop_mut(agtClassic))
                SG_fear.append(compute_propo_coop_mut(agtFear))
                SG_greed.append(compute_propo_coop_mut(agtGreed))
            else:
                CH_classic.append(compute_propo_coop_mut(agtClassic))
                CH_fear.append(compute_propo_coop_mut(agtFear))
                CH_greed.append(compute_propo_coop_mut(agtGreed))
            print(game_name + " - classic - convergence rate: " + str(compute_propo_coop_mut(agtClassic)))
            print(game_name + " - fear - convergence rate: " + str(compute_propo_coop_mut(agtFear)))
            print(game_name + " - greed - convergence rate: " + str(compute_propo_coop_mut(agtGreed)))
    plot.plot_SRE_greed_fear(PD_classic, PD_fear, PD_greed, SG_classic, SG_fear, SG_greed, CH_classic, CH_fear,
                             CH_greed)


if __name__ == '__main__':
    mainGreedFearSRE()

import os
import pickle
import sys
from typing import Tuple, List

import numpy
from numpy import ndarray

from agent import Agent
from matrix_payoffs import Matrix_Payoffs
from model import Bush_Mosteller
from plot import Plot


def get_payoffs_vector(game: str = "PD", fear=False, greed=False) -> List[int]:
    """
    Returns the payoff vector for one of three games: Prisoner's Dilemma, Stag Hunt, Chicken
    [optional] Apply fear or greed modifiers
    :param game: string: "PD"=Prisoner's Dilemma, "SH"=Stag Hunt, "CH"=Chicken. Default: "PD"
    :param fear: True to apply a fear modifier. Default: False
    :param greed: True to apply a greed modifier. Default: False
    :return: The payoff vector
    """
    temptation = 0
    reward = 1
    punishment = 2
    sucker = 3
    pd = [4, 3, 1, 0]  # Prisoner's Dilemma: T > R > P > S
    if game == "SG":
        pd = [4, 3, 0, 1]  # Stag Hunt: R > T > P > S
    elif game == "CH":
        pd = [3, 4, 1, 0]   # Chicken: T > R > S > P
    if fear:
        pd[sucker] -= 1
    elif greed:
        pd[reward] += 1
    return pd


def save_data(data: ndarray, filename: str) -> None:
    """
    Serializes a training data set
    :param data: The data set
    :param filename: The identifier of the file
    """
    zipped = tuple(zip(*data))
    for i, agent_data in enumerate(zipped):
        filename = filename.replace("\n", "")
        with open("data/agent_{0}_{1}.p".format(str(i), filename), "wb") as dest:
            pickle.dump(agent_data, dest)


def read_data(path: str) -> Tuple:
    """
    Deserializes stored data
    :param path: Path to destination
    :return: A tuple containing the data
    """
    with open(path, "rb") as source:
        data = pickle.load(source)
    return data


def compute_average_evolution(by_agent: Tuple) -> ndarray:
    """
    Computes the average evolution in time of the data
    :param by_agent: A Tuple where the ith element contains the data of agent i. The elements are a List of repetitions
    where each repetition contains the results of the a training set
    :return: A list containing the averaged results across agents and repetitions.
    """
    out = numpy.zeros(len(by_agent[0][0]))
    for i in range(len(by_agent)):
        for repetition in by_agent[i]:
            out = numpy.add(out, repetition)
    return (out / len(by_agent)) / len(by_agent[0])


def compute_propo_coop_mut(by_agent: Tuple) -> ndarray:
    num_of_mut_coop = numpy.zeros(len(by_agent[0][0]))
    num_of_no_coop = numpy.zeros(len(by_agent[0][0]))
    for i in range(len(by_agent[0])): # 1000
        for j in range(len(by_agent[0][0])): # 100
            # When prob of action 0 for agents 0 and 1 is bigger than
            if(by_agent[0][i][j] > 0.9 and by_agent[1][i][j] > 0.9):
                num_of_mut_coop[j] += 1
            else:
                num_of_no_coop[j] += 1
    return (num_of_mut_coop / num_of_no_coop)


def train(game: Matrix_Payoffs, habituation: float, aspiration: float,
          learning_rate: float, nb_repetitions: int, nb_episodes: int) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Trains a set of agents on the given game
    :param game: The payoff matrix
    :param habituation: h
    :param aspiration: A
    :param learning_rate: l
    :param nb_repetitions: Number of times the training will be repeated
    :param nb_episodes: Number of training episodes in each repetition
    :return: A tuple containing: The probabilities of the actions of the agents, the aspirations of the agents and the
    stimuli during training
    """
    action_probabilities = numpy.empty(nb_repetitions, dtype=object)
    aspirations = numpy.empty(nb_repetitions, dtype=object)
    stimuli = numpy.empty(nb_repetitions, dtype=object)
    action = numpy.empty(nb_repetitions, dtype=object)
    for repetition in range(nb_repetitions):
        agents = [Agent(learning_rate, aspiration, habituation) for _ in range(game.num_agents)]
        model = Bush_Mosteller(agents, game)

        model.run(nb_episodes)
        action_probabilities[repetition] = model.get_action_probabilities()
        aspirations[repetition] = model.get_aspirations()
        stimuli[repetition] = model.get_stimuli_agent()
        action[repetition] = model.get_action()
    return action_probabilities, aspirations, stimuli, action


def main() -> None:
    parameters_list = list()
    if len(sys.argv) == 8:
        parameters_list.append(sys.argv[1:])
    elif len(sys.argv) == 2:
        with open(sys.argv[1], "r") as parameters_file:
            lines = parameters_file.readlines()
        for line in lines:
            parameters_list.append(line.split(" "))
    else:
        print("Invalid arguments")
        print("Usage: python runner.py [game] [mode] habituation aspiration learning_rate nb_repetitions "
              "nb_episodes\n\tOR\n       python runner.py source_file\n\t- game:\n\t\t- PD: Prisoner's Dilemma\n\t\t- "
              "SG: Stag Hunt\n\t\t- CH: Chicken\n\t- mode:\n\t\t- classic\n\t\t- fear\n\t\t- greed\t\n\t- "
              "source_file: File where each line contains a set of arguments")
        return

    if not os.path.exists("data/"):
        os.makedirs("data/")

    for i, parameters in enumerate(parameters_list):
        print("({0}/{1}) Training: game={2} mode={3} "
              "h={4} A={5} l={6} reps={7} eps={8}".format(str(i + 1), len(parameters_list), *parameters))
        floats = numpy.asarray(parameters[2:5], dtype=float)
        ints = numpy.asarray(parameters[5:], dtype=int)
        game = Matrix_Payoffs(get_payoffs_vector(parameters[0], "fear" == parameters[1], "greed" == parameters[1]))
        action_probabilities, aspirations, stimuli, action = train(game, *floats, *ints)
        filename = "_".join(parameters)
        filename = filename.replace(".", "-")
        save_data(action_probabilities, "act_probs_" + filename)
        save_data(aspirations, "asp_" + filename)
        save_data(stimuli, "stim_" + filename)
        save_data(action, "actions_" + filename)


def plot():
    plt = Plot()
    # For h = 0
    for aspiration in ["0-5", "2"]:
        avg_coop_by_game = []
        for game_name in ["PD","CH","SG"]: # Habi, Aspi, Learning Rate
            agt0 = read_data("data/agent_0_act_probs_"+game_name+"_classic_0_"+ aspiration +"_0-5_1000_100.p")
            agt1 = read_data("data/agent_1_act_probs_"+game_name+"_classic_0_"+ aspiration +"_0-5_1000_100.p")
            avg_coop_by_game.append(compute_average_evolution((agt0, agt1)))
        plt.plot(avg_coop_by_game, "Proba. of cooperation")
    # For h = 0.2
    for aspiration in ["2", "3"]:
        avg_coop_by_game = []
        for game_name in ["PD","CH","SG"]: # Habi, Aspi, Learning Rate
            agt0 = read_data("data/agent_0_act_probs_"+game_name+"_classic_0-2_"+ aspiration +"_0-5_1000_100.p")
            agt1 = read_data("data/agent_1_act_probs_"+game_name+"_classic_0-2_"+ aspiration +"_0-5_1000_100.p")
            avg_coop_by_game.append(compute_propo_coop_mut((agt0, agt1)))
        plt.plot(avg_coop_by_game, "Proba. of cooperation")

if __name__ == '__main__':
    #main()
    plot()

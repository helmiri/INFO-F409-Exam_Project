import os
import pickle
import sys
from typing import Tuple, List

import numpy
from numpy import ndarray

from agent import Agent
from matrix_payoffs import Matrix_Payoffs
from model import Bush_Mosteller

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
        pd = [3, 4, 1, 0]  # Stag Hunt: R > T > P > S
    elif game == "CH":
        pd = [4, 3, 0, 1]  # Chicken: T > R > S > P
    if fear:
        pd[sucker] = -1
    elif greed:
        pd[reward] = 5
    return pd


def save_data(data: ndarray, filename: str) -> None:
    """
    Serializes a training data set
    :param data: The data set
    :param filename: The identifier of the file
    """
    zipped = tuple(zip(*data))
    filename = filename.replace("\n", "")
    with open(f'data/agent_{filename}.p', "wb") as dest:
        pickle.dump(zipped[0], dest)


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


def compute_propo_coop_mut(agent: Tuple) -> float:
    count = 0
    for repetition in agent:
        count += 1 if repetition[len(repetition)-1] > 0.99 else 0
    return count / len(agent)


def train(game: Matrix_Payoffs, habituation: float, aspiration: float,
          learning_rate: float, nb_repetitions: int, nb_episodes: int) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
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


def main() -> List[List[str]]:
    parameters_list = list()
    if len(sys.argv) == 7:
        parameters_list.append(sys.argv[1:])
    elif len(sys.argv) == 2:
        with open(sys.argv[1], "r") as parameters_file:
            lines = parameters_file.readlines()
        for line in lines:
            parameters_list.append(line.split(" "))
    else:
        print("Invalid arguments")
        print("Usage: python runner.py [mode] habituation aspiration learning_rate nb_repetitions "
              "nb_episodes\n\tOR\n       python runner.py source_file\n\t- mode:\n\t\t- classic\n\t\t- fear\n\t\t- "
              "greed\t\n\t- source_file: File where each line contains a set of arguments")
        return []

    if not os.path.exists("data/"):
        os.makedirs("data/")

    count = 0
    for i, parameters in enumerate(parameters_list):
        for j, game_name in enumerate(["PD", "SG", "CH"]):
            count += 1
            print("({0}/{1}) Training: game={2} mode={3} "
                  "h={4} A={5} l={6} reps={7} eps={8}".format(str(count), len(parameters_list) * 3, game_name,
                                                              *parameters))
            floats = numpy.asarray(parameters[1:4], dtype=float)
            ints = numpy.asarray(parameters[4:], dtype=int)
            game = Matrix_Payoffs(get_payoffs_vector(game_name, "fear" == parameters[1], "greed" == parameters[1]))
            action_probabilities, aspirations, stimuli, action = train(game, *floats, *ints)
            filename = game_name + "_" + "_".join(parameters)
            filename = filename.replace(".", "-")
            save_data(action_probabilities, "act_probs_" + filename)
            save_data(aspirations, "asp_" + filename)
            save_data(stimuli, "stim_" + filename)
            save_data(action, "actions_" + filename)
    return parameters_list


if __name__ == '__main__':
    main()

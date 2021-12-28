import numpy as np

class Agent:
    def __init__(self, learning_rate=0.5, aspiration=2, habituation=0, probab_init=0.5):
        self.proba = np.full(shape=2, fill_value=probab_init)
        self.leara = learning_rate
        self.aspi = aspiration
        self.habi = habituation

    def learn(self, stimu, act):
        """
        Update the probabilities of each action based on the stimulus and the learning rate.

        :param stimu: The stimuli obtained.
        :param act: The current action of the agent.
        :return: The new probabilities of each action.
        """
        if(stimu >= 0):
            newprob = self.proba[act] + (1-self.proba[act]) * self.leara * stimu
        else:
            newprob = self.proba[act] + self.proba[act] * self.leara * stimu
        self.proba[act] = newprob
        self.proba[1-act] = 1 - newprob
        return self.proba

    def updt_aspi(self, payoff):
        """
        Update the aspiration value of the agent.

        :return: The new aspiration value.
        """
        self.aspi = (1-self.habi) * self.aspi + self.habi * payoff
        return self.aspi

    def act(self):
        """
        Choose an action with a probability p.

        :return: The action choosen.
        """
        return np.random.choice([0, 1], p=self.proba)

    def cmpt_stimu(self, payoff, sup):
        """
        Compute the value of the stimuli.

        :return: The value of the stimuli.
        """
        return (payoff - self.aspi) / sup

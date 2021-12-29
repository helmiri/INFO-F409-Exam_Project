# INFO-F409 - Learning dynamics in social dilemmas
The goal of this project, being the exam of the **Learning dynamics course**, was to reproduce the experiments performed in a scientific article related to the course. This article was entitled ***Learning dynamics in social dilemmas*** and was written by *Michael W. Macy* and *Andreas Flache*. This project was produced by the following contributors : *Damien Decleire*, *Anthony Zhou*, *Hamza El Miri* and *Julien Baudru*.

## Requirements
To run this project you have to install the following libraries :
```
numpy
matplotlib
```

## Run
This project can be executed as follows :
```
python runner.py [game] [mode] habituation aspiration learning_rate nb_repetitions nb_episodes
```
or
```
python runner.py source_file
```
where:
```
- game:
    - PD: Prisoner's Dilemma
    - SG: Stag Hunt
    - CH: Chicken
- mode:
    - classic
    - fear
    - greed
    
- source_file: File where each line contains a set of arguments following the format in the first execution option
               The training of the agents will be performed with each parameter set in the file
```
The results of each agent's training will be saved in the ```data/``` folder. The filename follows the format:
```
agent_[num]_[data type]_[game]_[mode]_[h]_[A]_[l]_[nb_reps]_[nb_eps].p

    - num: 0-n agents
    - data type:
        - asp: aspirations at each timestep for every repetition
        - act_probs: cooperation probability at each timestep for every repetition
        - stim: stimulation received at each timestep for every repetition
    - h: habituation
    - A: Aspiration
    - l: learning rate
    - nb_reps: number of repetitions
    - nb_eps: number of episodes 
```




This webpage contains the results of experiments on PEG when applied to pixel inputs.
Note that the goal sampling strategy used for pixel inputs is not MPPI. 
Goals are sampled from the replay buffer and the goal with the highest exploratory value is used.

PEG was trained on two different environments: 3-Block Stacking, NES Super Mario Bros.\
For the 3-Block Stacking environment, Image PEG, Image LEXA and Vector PEG are compared.\
For the NES Super Mario Bros environment only Image PEG is trained.

# 3-Block Stacking

## Success rate comparison
  This plot shows the average success rates of Image PEG, Image LEXA and Vector PEG when attempting to reach a predefined set of goals.

  Image PEG and LEXA perform similarly, Vector PEG performs best.

  <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/images/ts_success_rate.png" width=600px>

## Goal conditioned behaviour

  | Model | Behaviour |
  |---|---|
  | Image PEG | <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/ts_peg_6M.gif" width=960px> |
  | Image LEXA | <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/ts_lexa_6M.gif" width=960px> |
  | Vector PEG | <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/ts_vecpeg_1M.gif" width=960px> |

## World model error
  These videos show the error of the world model predictions. 
  
  Rows top to bottom show ground truth, prediction, and error. 
  Errors mainly concern block dynamics.

  | Image PEG | Image LEXA|
  | --- | --- |
  | <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/ts_peg_6M_wm.gif"> | <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/ts_lexa_6M_wm.gif"> |

# NES Super Mario Bros
Unless specified, videos refer to the agent behaviour after 10M training steps.

## Goal conditioned behaviour
  Agent behaviour is observed when tasked with achieving a set of predefined goals.
  
  <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/mario_10M.gif" width=960px>

## World model error

  This video shows the world model errors.
  Some notable errors are the desync of the level horizontal scrolling.
  Changes in agent speed, or colliding with walls is not always correctly predicted by the model.

  <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/mario_10M_wm.gif" width=500px>

## Training policy observation
  Observation of the Go-Explore mechanism of the agent at 5M and 10M steps.
  The agent first attempts to achieve a chosen goal, then explores from the reached position.
  The following videos show the agent behaviour at 5M and 10M training steps respectively.

  | 5M steps | 10M steps |
  | --- | --- |
  | <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/mario_5M_goexp.gif" width=320px> | <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/mario_10M_goexp.gif" width=320px> |

  The agent chooses far goals to achieve high exploratory value, and then continues exploring to the right. \
  Later iterations seem to focus on colliding with walls instead. 
  Indeed, the effect of these collisions is difficult to model, and may correspond to high epistemic uncertainty/exploratory value for the agent.

  Moreover, in the 10M step video the agent has learned how to control the game camera to decenter Mario from the screen, adding another layer of complexity for computing collision and subsequent scrolling speed changes.
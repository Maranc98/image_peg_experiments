This webpage contains the results of experiments on PEG when applied to pixel inputs.
Note that the goal sampling strategy used for pixel inputs is not MPPI. 
Goals are sampled from the replay buffer and the goal with the highest exploratory value is used.

PEG was trained on two different environments: 3-Block Stacking, NES Super Mario Bros.\\
For the 3-Block Stacking environment, Image PEG, Image LEXA and Vector PEG are compared.\\
For the NES Super Mario Bros environment only Image PEG is trained.

# 3-Block Stacking

## Success rate comparison
  This plot shows the average success rates of Image PEG, Image LEXA and Vector PEG when attempting to reach a predefined set of goals.

  Image PEG and LEXA perform similarly, Vector PEG performs best.

  <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/images/ts_success_rate.png" width=600px>

## Goal conditioned behaviour

  Image PEG

  ![GC behaviour](/docs/videos/ts_peg_6M.gif?raw=true)

  Image LEXA

  ![GC behaviour](/docs/videos/ts_lexa_6M.gif?raw=true)

  Vector PEG

  <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/ts_vecpeg_1M.gif" width=960px>

## World model error
  These videos show the error of the world model predictions. 
  
  Rows top to bottom show ground truth, prediction, and error. 
  Errors mainly concern block dynamics.

  Image PEG

  ![WM error](/docs/videos/ts_peg_6M_wm.gif)

  Image LEXA
  
  ![WM error](/docs/videos/ts_lexa_6M_wm.gif)

# NES Super Mario Bros
Unless specified, videos refer to the agent behaviour after 10M training steps.

## Goal conditioned behaviour

  ![GC behaviour](/docs/videos/mario_10M.gif)

## World model error

  ![WM error](/docs/videos/mario_10M_wm.gif)

## Training policy observation
  Observation of the Go-Explore mechanism of the agent at 5M and 10M steps.
  
  <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/mario_5M_goexp.gif" width=320px>
  <img src="https://raw.githubusercontent.com/Maranc98/image_peg_experiments/master/docs/videos/mario_10M_goexp.gif" width=320px>


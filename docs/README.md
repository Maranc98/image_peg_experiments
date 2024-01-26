This webpage contains the results of experiments on PEG when applied to pixel inputs.
Note that the goal sampling strategy used for pixel inputs is not MPPI. 
Goals are sampled from the replay buffer and the goal with the highest exploratory value is used.
PEG was trained on three different environments: 3-Block Stacking, NES Super Mario Bros and 2d Point Maze.

# 3-Block Stacking

## Success rate comparison
  This plot shows the average success rates of Image PEG, Image LEXA and Vector PEG when attempting to reach a predefined set of goals.

  Image PEG and LEXA perform similarly, Vector PEG performs best.


## Goal conditioned behaviour

  Image PEG

  ![GC behaviour](/docs/videos/ts_peg_6M.gif?raw=true)

  Image LEXA

  ![GC behaviour](/docs/videos/ts_lexa_6M.gif?raw=true)

  Vector PEG

  ![GC behaviour](/docs/videos/ts_vecpeg_1M.gif?raw=true)

## World model error


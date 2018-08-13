# mLearning_2D
This is an ongoing project to create a machine learning model for data in the form of 2D arrays. The modules in
this repo reflect my current progress. For a more up-to-date (but less stable) version, see other branches.

# To run in a Docker container:
  $ git clone https://github.com/rockstardotb/mLearning_2D.git
  
  $cd mLearning_2D
  
  $ docker build -t waleedka/modern-deep-learning .
  
  $ docker run -i -t --name mlearn waleedka/modern-deep-learning:latest
  
# To run without Docker container:
  $ git clone https://github.com/rockstardotb/mLearning_2D.git
  
  $ cd mLearning_2D
  
  $ pip install -e .

% main.m
clear;clc

Nx = 50;  % point along x-direction
Nt = 50;  % point along t-direction
Nint = 1500; % inner point

% nn
numLayers = 9;
numNeurons = 20;

% lbfgs
maxIterations = 20000;

set_network

init_data

train_solution

calculate_L2_Error

drawaverage
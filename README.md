# SmallNetworkPlusPlus
A small network in keras with additional technics.

## Table of contents
* [Topology](#Topoloty)
* [Technologies added](#technologies-added)
* [Possible improvements](#possible-improvements)

## Topology
This project is simple network low on parameters (18,211)

=======Block#1=======

Input((100,100,1))

MaxPooling2D(pool_size=(2,2))

=======Block#2=======

Conv2D (16, kernel_size=(3,3), strides=1, padding='same')

ReLU()

Conv2D (16, kernel_size=(3,3), strides=1, padding='same')

ReLU()

AveragePooling2D(pool_size=(2, 2))

=======Block#3=======

Conv2D (24, kernel_size=(3,3), strides=1, padding='same')

ReLU()

Conv2D (24, kernel_size=(3,3), strides=1, padding='same')

ReLU()

AveragePooling2D(pool_size=(2, 2))

=======Block#5=======

Conv2D (32, kernel_size=(3,3), strides=1, padding='same')

GlobalAveragePooling2D()

Flatten()

Dense(3)

Softmax()
	
## Technologies added
Project is created with adding:
* Weight decay
* SGD with momentum
* Gradnorm = 1.0
* Small batchsize 16, 32

## Possible improvements
* Own learning loop
* Sample hardness estimation
* Including hardest examples into next loop

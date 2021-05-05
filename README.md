# LiteChess

## By Team Trojan Knights

There are three parts to this project:
- Neural network model for evaluating positions (**litechess-model**)
- Chess engine backend that makes moves after positions are evaluated (**litechess-engine-backend**)
- Litechess Website that hosts the chess engine (**litechess-website**)

### Litechess Model
The litechess model is based on the DeepChess neural network model that has two parts :
- An autoencoder for identifying features relevant to position evaluation
- A Siamese network that evaluates positions

### Litechess Engine Backend
The litechess engine backend is a modified version of Sunfish's chess engine backend. [Sunfish](https://github.com/thomasahle/sunfish) is a popular chess engine written in Python whose evaluation function is a simple piece-square mapping. We change this evaluation function to our litechess neural network.

### Litechess Website
The [litechess website](https://litechess.vercel.app) is developed in [Next.JS](https://nextjs.org/) (a React JS framework) and has several features that make it interactive and suitable for both beginners and professionals:
- Playing a game from the starting position
- Playing a game from a position provided by the user (in theform of a FEN string)
- Drawing arrows on the board in order to aid play
- Download played game (as a PGN file).

The website is deployed on [Vercel](https://vercel.com/), a serverless platform that makes it convenient to serve Next.JS projects in a highly efficient and scalable manner.
